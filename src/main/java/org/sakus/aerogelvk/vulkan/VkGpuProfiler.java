package org.sakus.aerogelvk.vulkan;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkQueryPoolCreateInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.lwjgl.vulkan.VK10.*;

/**
 * Optional per-pass Vulkan GPU timestamp profiler.
 * Disabled by default; enable with {@code -Daerogelvk.gpuProfile=true}.
 */
public final class VkGpuProfiler {

    private static final Logger LOGGER = LoggerFactory.getLogger(VkGpuProfiler.class);
    private static final boolean ENABLED = Boolean.getBoolean("aerogelvk.gpuProfile");
    private static final int MAX_QUERIES = 128; // up to 64 begin/end pass pairs per frame

    private static final class PassRecord {
        final String name;
        final int beginQuery;
        final int endQuery;

        PassRecord(String name, int beginQuery, int endQuery) {
            this.name = name;
            this.beginQuery = beginQuery;
            this.endQuery = endQuery;
        }
    }

    private final long[] queryPools = new long[VkContext.FRAMES_IN_FLIGHT];
    private final int[] slotQueryCount = new int[VkContext.FRAMES_IN_FLIGHT];
    private final boolean[] slotHasData = new boolean[VkContext.FRAMES_IN_FLIGHT];
    private final String[] slotMode = new String[VkContext.FRAMES_IN_FLIGHT];
    @SuppressWarnings("unchecked")
    private final List<PassRecord>[] slotPasses = new List[VkContext.FRAMES_IN_FLIGHT];

    private final Map<String, double[]> accumMs = new LinkedHashMap<>();
    private String accumMode = "";
    private int framesCollected = 0;
    private long lastLogNanos = 0L;

    private double timestampPeriodNs = 0.0;
    private boolean initialized = false;
    private boolean supported = false;

    public VkGpuProfiler() {
        for (int i = 0; i < VkContext.FRAMES_IN_FLIGHT; i++) {
            slotPasses[i] = new ArrayList<>();
            slotMode[i] = "";
        }
    }

    public static boolean isEnabled() {
        return ENABLED;
    }

    public synchronized void init(VkContext ctx) {
        if (!ENABLED || initialized || ctx == null || !ctx.isInitialized()) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(stack);
            vkGetPhysicalDeviceProperties(ctx.getPhysicalDevice(), props);
            this.timestampPeriodNs = props.limits().timestampPeriod();
            if (this.timestampPeriodNs <= 0.0) {
                LOGGER.warn("[Valkyrie][GpuProfile] Physical device reports timestampPeriod={}; GPU profiling disabled.",
                        this.timestampPeriodNs);
                this.initialized = true;
                this.supported = false;
                return;
            }

            VkQueryPoolCreateInfo poolInfo = VkQueryPoolCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO)
                    .queryType(VK_QUERY_TYPE_TIMESTAMP)
                    .queryCount(MAX_QUERIES);

            LongBuffer pPool = stack.mallocLong(1);
            for (int i = 0; i < VkContext.FRAMES_IN_FLIGHT; i++) {
                int res = vkCreateQueryPool(ctx.getVkDevice(), poolInfo, null, pPool);
                if (res != VK_SUCCESS) {
                    LOGGER.warn("[Valkyrie][GpuProfile] Failed to create VkQueryPool[{}] (err={}); GPU profiling disabled.", i, res);
                    close(ctx.getVkDevice());
                    this.initialized = true;
                    this.supported = false;
                    return;
                }
                queryPools[i] = pPool.get(0);
                slotQueryCount[i] = 0;
                slotHasData[i] = false;
                slotPasses[i].clear();
            }
            this.lastLogNanos = System.nanoTime();
            this.supported = true;
            this.initialized = true;
            LOGGER.info("[Valkyrie][GpuProfile] Enabled Vulkan timestamp profiling (timestampPeriod={} ns, maxPasses={}).",
                    String.format(Locale.ROOT, "%.3f", timestampPeriodNs), MAX_QUERIES / 2);
        }
    }

    /**
     * Collects finished results from the prior use of {@code slot} (after its fence has been waited),
     * logs a 1-second summary when due, and resets the query pool for the upcoming frame.
     */
    public synchronized void beginFrame(VkDevice device, int slot, VkCommandBuffer resetCmd, String modeLabel) {
        if (!ENABLED || !supported || device == null || resetCmd == null) {
            return;
        }
        int s = Math.floorMod(slot, VkContext.FRAMES_IN_FLIGHT);
        if (slotHasData[s] && slotQueryCount[s] > 0) {
            collectSlotResultsLocked(device, s);
        }

        if (!modeLabel.equals(accumMode)) {
            if (framesCollected > 0) {
                logSummaryLocked();
            }
            accumMs.clear();
            framesCollected = 0;
            accumMode = modeLabel;
            lastLogNanos = System.nanoTime();
        } else {
            long now = System.nanoTime();
            if (framesCollected > 0 && now - lastLogNanos >= 1_000_000_000L) {
                logSummaryLocked();
                accumMs.clear();
                framesCollected = 0;
                lastLogNanos = now;
            }
        }

        slotQueryCount[s] = 0;
        slotPasses[s].clear();
        slotMode[s] = modeLabel;
        slotHasData[s] = true;
        vkCmdResetQueryPool(resetCmd, queryPools[s], 0, MAX_QUERIES);
    }

    /**
     * Allocates a [begin, end] query index pair for {@code passName} in {@code slot} and writes the begin timestamp.
     * Returns the {@code endQuery} index (or {@code -1} if disabled or query capacity is reached).
     */
    public synchronized int beginPass(VkCommandBuffer cmd, int slot, String passName) {
        if (!ENABLED || !supported || cmd == null) {
            return -1;
        }
        int s = Math.floorMod(slot, VkContext.FRAMES_IN_FLIGHT);
        int qBegin = slotQueryCount[s];
        if (qBegin + 2 > MAX_QUERIES) {
            return -1;
        }
        int qEnd = qBegin + 1;
        slotQueryCount[s] = qEnd + 1;
        slotPasses[s].add(new PassRecord(passName, qBegin, qEnd));
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[s], qBegin);
        return qEnd;
    }

    /**
     * Writes the end timestamp for a pass started with {@link #beginPass}.
     */
    public void endPass(VkCommandBuffer cmd, int slot, int endQuery) {
        if (!ENABLED || !supported || cmd == null || endQuery < 0) {
            return;
        }
        int s = Math.floorMod(slot, VkContext.FRAMES_IN_FLIGHT);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[s], endQuery);
    }

    private void collectSlotResultsLocked(VkDevice device, int s) {
        int count = slotQueryCount[s];
        if (count <= 0 || slotPasses[s].isEmpty()) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer data = stack.mallocLong(count);
            int res = vkGetQueryPoolResults(
                    device,
                    queryPools[s],
                    0,
                    count,
                    data,
                    Long.BYTES,
                    VK_QUERY_RESULT_64_BIT
            );
            if (res != VK_SUCCESS) {
                return;
            }
            for (PassRecord rec : slotPasses[s]) {
                long t0 = data.get(rec.beginQuery);
                long t1 = data.get(rec.endQuery);
                if (t1 >= t0) {
                    double ms = (double) (t1 - t0) * timestampPeriodNs / 1_000_000.0;
                    double[] stat = accumMs.computeIfAbsent(rec.name, k -> new double[2]);
                    stat[0] += ms;
                    stat[1] += 1.0;
                }
            }
            framesCollected++;
        }
    }

    private void logSummaryLocked() {
        if (accumMs.isEmpty() || framesCollected <= 0) {
            return;
        }
        double totalMs = 0.0;
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, double[]> entry : accumMs.entrySet()) {
            double[] stat = entry.getValue();
            double avg = stat[0] / Math.max(1.0, stat[1]);
            totalMs += avg;
            if (sb.length() > 0) {
                sb.append(' ');
            }
            sb.append(entry.getKey()).append('=').append(String.format(Locale.ROOT, "%.3f", avg)).append("ms");
        }
        LOGGER.info("[Valkyrie][GpuProfile] mode={} frames={} total={}ms | {}",
                accumMode.isEmpty() ? "unknown" : accumMode,
                framesCollected,
                String.format(Locale.ROOT, "%.3f", totalMs),
                sb);
    }

    public synchronized void close(VkDevice device) {
        if (framesCollected > 0) {
            logSummaryLocked();
            accumMs.clear();
            framesCollected = 0;
        }
        if (device != null) {
            for (int i = 0; i < VkContext.FRAMES_IN_FLIGHT; i++) {
                if (queryPools[i] != VK_NULL_HANDLE) {
                    vkDestroyQueryPool(device, queryPools[i], null);
                    queryPools[i] = VK_NULL_HANDLE;
                }
                slotQueryCount[i] = 0;
                slotHasData[i] = false;
                slotPasses[i].clear();
            }
        }
        initialized = false;
        supported = false;
    }
}
