package org.sakus.aerogelvk;

import org.sakus.aerogelvk.compat.CompatManager;
import org.sakus.aerogelvk.config.ValkyrieConfig;
import org.sakus.aerogelvk.meshing.AtlasUVTable;
import org.sakus.aerogelvk.meshing.GreedyMesher;
import org.sakus.aerogelvk.shaderpack.ShaderUniformLayout;
import org.sakus.aerogelvk.shaderpack.VkShaderPackPipeline;
import org.sakus.aerogelvk.vulkan.VkContext;
import org.sakus.aerogelvk.vulkan.VkGpuProfiler;
import org.sakus.aerogelvk.vulkan.VkInteropManager;
import org.sakus.aerogelvk.vulkan.VkMegaBuffer;
import org.sakus.aerogelvk.vulkan.VkPipelineManager;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.material.FogType;
import net.minecraft.world.phys.Vec3;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK12.vkCmdDrawIndexedIndirectCount;
import static org.lwjgl.vulkan.VK13.*;

/**
 * Master Renderer & Orchestrator for ValkyrieRender.
 * Coordinates VkContext, VkInteropManager, VkMegaBuffer, VkPipelineManager,
 * AtlasUVTable, and GreedyMesher to deliver a GPU-driven deferred rendering pipeline.
 */
public final class ValkyrieRenderer {

    private static final Logger LOGGER = LoggerFactory.getLogger(ValkyrieRenderer.class);
    private static ValkyrieRenderer instance;

    public static synchronized ValkyrieRenderer getInstance() {
        if (instance == null) {
            instance = new ValkyrieRenderer();
        }
        return instance;
    }

    private VkContext vkContext;
    private VkInteropManager vkInteropManager;
    private VkMegaBuffer vkMegaBuffer;
    private VkPipelineManager vkPipelineManager;
    private VkShaderPackPipeline vkShaderPackPipeline;
    private final VkGpuProfiler gpuProfiler = new VkGpuProfiler();
    private AtlasUVTable atlasUVTable;
    private GreedyMesher greedyMesher;
    private volatile boolean pendingShaderPackReload = false;
    private volatile boolean explicitPackLoaded = false;
    private String lastLoggedRenderPipeline = "";
    private float smoothEyeBrightnessX = 240.0f;
    private float smoothEyeBrightnessY = 240.0f;
    private boolean hasSmoothedEyeBrightness = false;
    private float smoothedWetness = 0.0f;
    private boolean hasSmoothedWetness = false;
    private long lastShaderPackFrameNanos = 0L;
    private float shaderPackFrameTimeCounter = 0.0f;

    private static final int FRAMES_IN_FLIGHT = 2;
    private static final int MAX_SECTION_CAPTURES_PER_FRAME = 64;
    private static final long MAX_DRAIN_NANOS_STORM = 12_000_000L; // 12 ms CPU budget during initial load storm
    private static final long MAX_DRAIN_NANOS_STEADY = 4_000_000L; // 4 ms CPU budget in steady state
    private static final int STORM_DIRTY_THRESHOLD = 500;
    private static final int MAX_IN_FLIGHT_MESHES = 128;
    private static final int MAX_SECTION_UPLOADS_PER_FRAME = 64;
    private static final int DISTANCE_BUCKETS = 33;
    private static final boolean CHUNK_PROFILE = Boolean.getBoolean("aerogelvk.chunkProfile");
    private long chunkProfStartNs = 0L;
    private long chunkProfLastLogNs = 0L;
    private long chunkProfDrainNs = 0L;
    private long chunkProfPollNs = 0L;
    private long chunkProfFenceNs = 0L;
    private long chunkProfUploadNs = 0L;
    private long chunkProfUploadBytes = 0L;
    private int chunkProfCaptured = 0;
    private int chunkProfUploaded = 0;
    private int chunkProfFrames = 0;
    private boolean chunkProfSettled = false;
    private static final int RECORD_WORKER_COUNT = Math.max(1, Math.min(3,
            Runtime.getRuntime().availableProcessors() - Math.max(1, Runtime.getRuntime().availableProcessors() / 2)));
    private static final int MAX_PARALLEL_CMDS_PER_WORKER = 4;

    private final Set<Long> dirtySections = ConcurrentHashMap.newKeySet();

    private final VkCommandBuffer[] commandBuffers = new VkCommandBuffer[FRAMES_IN_FLIGHT];
    private final VkCommandBuffer[] postCommandBuffers = new VkCommandBuffer[FRAMES_IN_FLIGHT];
    private final VkCommandBuffer[] transferCommandBuffers = new VkCommandBuffer[FRAMES_IN_FLIGHT];
    private final VkCommandBuffer[] cullComputeCommandBuffers = new VkCommandBuffer[FRAMES_IN_FLIGHT];
    private final VkCommandBuffer[] hizComputeCommandBuffers = new VkCommandBuffer[FRAMES_IN_FLIGHT];
    private final long[] transferCompleteSemaphores = new long[FRAMES_IN_FLIGHT];
    private final long[] cullCompleteSemaphores = new long[FRAMES_IN_FLIGHT];
    private final long[] gbufferDoneSemaphores = new long[FRAMES_IN_FLIGHT];
    private long hizReadySemaphore = VK_NULL_HANDLE;
    private boolean hizSemaphorePending = false;
    private final long[][] workerCommandPools = new long[RECORD_WORKER_COUNT][FRAMES_IN_FLIGHT];
    private final VkCommandBuffer[][][] workerCommandBuffers = new VkCommandBuffer[RECORD_WORKER_COUNT][FRAMES_IN_FLIGHT][MAX_PARALLEL_CMDS_PER_WORKER];
    private ExecutorService recordExecutor;
    private final long[] frameFences = new long[FRAMES_IN_FLIGHT];
    private final boolean[] fenceWaited = new boolean[FRAMES_IN_FLIGHT];
    private int currentFrameSlot = 0;
    private int lastSubmittedSlot = 0;

    private int currentWidth = 0;
    private int currentHeight = 0;
    private long frameIndex = 0;
    private boolean dirtyShadow = true;
    private boolean hasPreviousFrameHiZ = false;
    private boolean initialized = false;
    private volatile boolean unavailable = false;
    private String unavailableReason = null;

    private final Matrix4f prevViewProj = new Matrix4f();
    private final Matrix4f[] csmMatrices = new Matrix4f[VkPipelineManager.CSM_CASCADE_COUNT];
    private final Matrix4f[] renderedCsmMatrices = new Matrix4f[VkPipelineManager.CSM_CASCADE_COUNT];
    private final boolean[] csmCascadeInitialized = new boolean[VkPipelineManager.CSM_CASCADE_COUNT];

    // Static-frame shadow skip state
    private static final float SHADOW_STATIC_EPSILON = 1e-4f;
    private boolean shadowStaticStateValid = false;
    private boolean shadowStaticDirtyThisFrame = true;
    private final Vector3f prevShadowCameraPos = new Vector3f();
    private final Matrix4f prevShadowViewMatrix = new Matrix4f();
    private final Vector3f prevShadowSunDir = new Vector3f();
    private long prevShadowDayTime = Long.MIN_VALUE;
    private float prevShadowRainLevel = Float.NaN;
    private int prevShadowLevelId = 0;
    private String prevShadowDimensionId = "";
    private int prevShadowEntityCount = -1;
    private long prevShadowEntitySum = 0L;
    private long prevShadowEntityXor = 0L;
    private boolean prevCfgCsmEnabled = true;
    private int prevCfgCsmCascades = 3;
    private int prevCfgCsmResolution = 1024;
    private int prevCfgCsmUpdateInterval = 6;
    private boolean prevCfgVanillaLighting = false;
    private boolean prevCfgShadowSkipStaticFrames = true;
    private int prevCfgShadowResCap = 0;
    private boolean prevCfgShaderPackActive = false;

    private ValkyrieRenderer() {
        this.atlasUVTable = AtlasUVTable.getInstance();
        this.greedyMesher = GreedyMesher.getInstance();
        for (int i = 0; i < csmMatrices.length; i++) {
            csmMatrices[i] = new Matrix4f();
            renderedCsmMatrices[i] = new Matrix4f();
        }
    }

    public synchronized void ensureInitialized(int width, int height) {
        if (unavailable) {
            return;
        }
        if (width <= 0) width = 1280;
        if (height <= 0) height = 720;

        if (initialized) {
            if (currentWidth != width || currentHeight != height) {
                onResize(width, height);
            }
            return;
        }

        LOGGER.info("[Valkyrie] Initializing ValkyrieRenderer ({}x{})...", width, height);

        this.currentWidth = width;
        this.currentHeight = height;

        try {
            // 1. Vulkan 1.3 Context
            this.vkContext = VkContext.getInstance();
            if (!vkContext.isInitialized()) {
                vkContext.init();
            }
            this.gpuProfiler.init(vkContext);

            // 2. Zero-Copy OpenGL Interop Manager
            this.vkInteropManager = VkInteropManager.getInstance();
            vkInteropManager.recreate(width, height);

            // 3. GPU Mega-Buffer
            this.vkMegaBuffer = VkMegaBuffer.getInstance();
            if (!vkMegaBuffer.isInitialized()) {
                vkMegaBuffer.init();
            }

            // 4. Block Atlas UV Table
            this.atlasUVTable = AtlasUVTable.getInstance();
            atlasUVTable.rebuildFromBlockAtlas();

            // 5. Greedy Mesher Worker Pool
            this.greedyMesher = GreedyMesher.getInstance();

            // 6. Pipeline Manager
            this.vkPipelineManager = VkPipelineManager.getInstance();
            if (!vkPipelineManager.isInitialized()) {
                vkPipelineManager.init(
                        width,
                        height,
                        vkInteropManager.getSharedColorImageView(),
                        vkInteropManager.getSharedDepthImageView()
                );
            }

            // 7. Upload initial Atlas UV table & Block Atlas texture, then update Descriptor Set 0
            syncAtlasResources();
            updateDescriptorSet0Buffers();

            // 7b. Native Vulkan Shaderpack Pipeline (active only when ValkyrieConfig enables a pack or loadShaderPack() was called)
            if (this.vkShaderPackPipeline == null) {
                this.vkShaderPackPipeline = new VkShaderPackPipeline(vkContext, vkMegaBuffer);
            }
            if (!this.vkShaderPackPipeline.isInitialized()) {
                this.vkShaderPackPipeline.init(width, height, VK_FORMAT_R8G8B8A8_UNORM);
            }
            syncShaderPackAtlas();
            if (!explicitPackLoaded && (CompatManager.isNativeShaderPackAllowed() || pendingShaderPackReload)) {
                pendingShaderPackReload = false;
                reloadShaderPackLocked();
            }

            // 8. Allocate Double-Buffered Command Buffers, Transfer Semaphores & Frame Synchronization Fences
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
                        .flags(VK_FENCE_CREATE_SIGNALED_BIT);
                VkSemaphoreCreateInfo semInfo = VkSemaphoreCreateInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);

                LongBuffer pFence = stack.mallocLong(1);
                LongBuffer pSem = stack.mallocLong(1);
                for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
                    this.commandBuffers[i] = vkContext.allocateCommandBuffer();
                    this.postCommandBuffers[i] = vkContext.allocateCommandBuffer();
                    this.transferCommandBuffers[i] = vkContext.allocateTransferCommandBuffer();
                    this.cullComputeCommandBuffers[i] = vkContext.allocateComputeCommandBuffer();
                    this.hizComputeCommandBuffers[i] = vkContext.allocateComputeCommandBuffer();

                    int semRes = vkCreateSemaphore(vkContext.getVkDevice(), semInfo, null, pSem);
                    if (semRes != VK_SUCCESS) {
                        throw new RuntimeException("Failed to create transfer semaphore[" + i + "]: " + semRes);
                    }
                    this.transferCompleteSemaphores[i] = pSem.get(0);

                    semRes = vkCreateSemaphore(vkContext.getVkDevice(), semInfo, null, pSem);
                    if (semRes != VK_SUCCESS) {
                        throw new RuntimeException("Failed to create cull semaphore[" + i + "]: " + semRes);
                    }
                    this.cullCompleteSemaphores[i] = pSem.get(0);

                    semRes = vkCreateSemaphore(vkContext.getVkDevice(), semInfo, null, pSem);
                    if (semRes != VK_SUCCESS) {
                        throw new RuntimeException("Failed to create gbuffer-done semaphore[" + i + "]: " + semRes);
                    }
                    this.gbufferDoneSemaphores[i] = pSem.get(0);

                    int res = vkCreateFence(vkContext.getVkDevice(), fenceInfo, null, pFence);
                    if (res != VK_SUCCESS) {
                        throw new RuntimeException("Failed to create frame fence[" + i + "]: " + res);
                    }
                    this.frameFences[i] = pFence.get(0);
                    this.fenceWaited[i] = true;
                }
                int hizSemRes = vkCreateSemaphore(vkContext.getVkDevice(), semInfo, null, pSem);
                if (hizSemRes != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create Hi-Z semaphore: " + hizSemRes);
                }
                this.hizReadySemaphore = pSem.get(0);
                this.hizSemaphorePending = false;

                // Stage B: Per-worker, per-slot VkCommandPools & pre-allocated primary command buffers
                VkCommandPoolCreateInfo workerPoolInfo = VkCommandPoolCreateInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                        .flags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT)
                        .queueFamilyIndex(vkContext.getQueueFamilyIndex());
                LongBuffer pPool = stack.mallocLong(1);
                PointerBuffer pWorkerCmds = stack.mallocPointer(MAX_PARALLEL_CMDS_PER_WORKER);
                for (int w = 0; w < RECORD_WORKER_COUNT; w++) {
                    for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
                        int poolRes = vkCreateCommandPool(vkContext.getVkDevice(), workerPoolInfo, null, pPool);
                        if (poolRes != VK_SUCCESS) {
                            throw new RuntimeException("Failed to create worker command pool[" + w + "][" + i + "]: " + poolRes);
                        }
                        long pool = pPool.get(0);
                        workerCommandPools[w][i] = pool;

                        VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                                .commandPool(pool)
                                .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                                .commandBufferCount(MAX_PARALLEL_CMDS_PER_WORKER);
                        int allocRes = vkAllocateCommandBuffers(vkContext.getVkDevice(), allocInfo, pWorkerCmds);
                        if (allocRes != VK_SUCCESS) {
                            throw new RuntimeException("Failed to allocate worker command buffers[" + w + "][" + i + "]: " + allocRes);
                        }
                        for (int k = 0; k < MAX_PARALLEL_CMDS_PER_WORKER; k++) {
                            workerCommandBuffers[w][i][k] = new VkCommandBuffer(pWorkerCmds.get(k), vkContext.getVkDevice());
                        }
                    }
                }
                if (RECORD_WORKER_COUNT > 1 && (this.recordExecutor == null || this.recordExecutor.isShutdown())) {
                    this.recordExecutor = Executors.newFixedThreadPool(RECORD_WORKER_COUNT, r -> {
                        Thread t = new Thread(r, "valkyrie-cmd-recorder");
                        t.setDaemon(true);
                        return t;
                    });
                }
            }

            this.initialized = true;
            LOGGER.info("[Valkyrie] ValkyrieRenderer successfully initialized.");
        } catch (Throwable t) {
            markUnavailableLocked(t);
        }
    }

    private void markUnavailableLocked(Throwable t) {
        if (!this.unavailable) {
            this.unavailable = true;
            this.unavailableReason = (t.getMessage() != null && !t.getMessage().isBlank())
                    ? t.getMessage()
                    : t.getClass().getSimpleName();
            LOGGER.warn("[Valkyrie] Vulkan initialization failed ({}); falling back to Vanilla/Sodium OpenGL rendering for this session.",
                    this.unavailableReason, t);
        }
        this.initialized = false;
        try {
            destroyResourcesLocked();
        } catch (Throwable cleanupErr) {
            LOGGER.debug("[Valkyrie] Secondary error while cleaning up partial Vulkan resources.", cleanupErr);
        }
    }

    private void syncAtlasResources() {
        if (atlasUVTable.getSpriteCount() <= 1) {
            atlasUVTable.rebuildFromBlockAtlas();
        }
        if (atlasUVTable.getSpriteCount() > 0) {
            vkMegaBuffer.uploadAtlasUvTable(atlasUVTable.getUvTableBuffer());
        }
        AtlasUVTable.AtlasPixels atlasPixels = atlasUVTable.captureBlockAtlasPixels();
        if (atlasPixels != null) {
            try {
                vkPipelineManager.uploadBlockAtlas(atlasPixels.width(), atlasPixels.height(), atlasPixels.rgbaPixels());
            } finally {
                MemoryUtil.memFree(atlasPixels.rgbaPixels());
            }
        }
        atlasUVTable.clearDirty();
        syncShaderPackAtlas();
    }

    private void syncShaderPackAtlas() {
        if (vkShaderPackPipeline != null && vkShaderPackPipeline.isInitialized() && vkPipelineManager != null) {
            vkShaderPackPipeline.updateBlockAtlas(
                    vkPipelineManager.getBlockAtlasView(),
                    vkPipelineManager.getNearestSampler(),
                    atlasUVTable
            );
        }
    }

    private void updateDescriptorSet0Buffers() {
        vkPipelineManager.updateDescriptorSet0(
                vkMegaBuffer.getMegaVertexBuffer().buffer,
                vkMegaBuffer.getChunkMetaBuffer().buffer,
                vkMegaBuffer.getIndirectCmdBuffer().buffer,
                vkMegaBuffer.getDrawCountBuffer().buffer,
                vkMegaBuffer.getVisibleChunkIdBuffer().buffer,
                vkMegaBuffer.getAtlasUvBuffer().buffer,
                vkPipelineManager.getBlockAtlasView()
        );
    }

    /**
     * Converts an OpenGL perspective projection matrix (z_ndc in [-1, 1], Y-up)
     * into a Vulkan Reversed-Z projection matrix (z_ndc in [1, 0], Y-up).
     */
    public static Matrix4f toVulkanReversedZ(Matrix4f glProj) {
        Matrix4f vkProj = new Matrix4f(glProj);
        vkProj.m02(-0.5f * glProj.m02() + 0.5f * glProj.m03());
        vkProj.m12(-0.5f * glProj.m12() + 0.5f * glProj.m13());
        vkProj.m22(-0.5f * glProj.m22() + 0.5f * glProj.m23());
        vkProj.m32(-0.5f * glProj.m32() + 0.5f * glProj.m33());
        return vkProj;
    }

    /**
     * Primary rendering hook called from MixinLevelRenderer.
     */
    public void renderWorld(Camera camera, Matrix4f frustumMatrix, Matrix4f projectionMatrix, DeltaTracker deltaTracker) {
        if (unavailable) {
            return;
        }
        RenderTarget mainTarget = Minecraft.getInstance().getMainRenderTarget();
        int width = (mainTarget != null && mainTarget.width > 0) ? mainTarget.width : currentWidth;
        int height = (mainTarget != null && mainTarget.height > 0) ? mainTarget.height : currentHeight;

        ensureInitialized(width, height);
        if (!initialized || unavailable) {
            return;
        }

        if (pendingShaderPackReload) {
            synchronized (this) {
                if (pendingShaderPackReload) {
                    pendingShaderPackReload = false;
                    reloadShaderPackLocked();
                }
            }
        }

        Vector3f cameraPos = new Vector3f(
                (float) camera.getPosition().x,
                (float) camera.getPosition().y,
                (float) camera.getPosition().z
        );

        // 1. Stage D: Overlap CPU-side frame preparation with previous in-flight GPU work
        // before waiting on the current frame slot's fence.
        long cpT0 = CHUNK_PROFILE ? System.nanoTime() : 0L;
        if (CHUNK_PROFILE && chunkProfStartNs == 0L && (!dirtySections.isEmpty() || vkMegaBuffer.getActiveSectionCount() > 0)) {
            chunkProfStartNs = cpT0;
            chunkProfLastLogNs = cpT0;
        }
        int capturedThisFrame = drainDirtySections(Minecraft.getInstance().level, cameraPos);
        long cpT1 = CHUNK_PROFILE ? System.nanoTime() : 0L;

        List<GreedyMesher.MeshedSectionResult> completedMeshes = null;
        GreedyMesher.MeshedSectionResult polled;
        int polledCount = 0;
        while (polledCount < MAX_SECTION_UPLOADS_PER_FRAME && (polled = greedyMesher.pollCompleted()) != null) {
            if (completedMeshes == null) {
                completedMeshes = new ArrayList<>();
            }
            completedMeshes.add(polled);
            polledCount++;
        }
        long cpT2 = CHUNK_PROFILE ? System.nanoTime() : 0L;

        // Compute Matrices and Sun Direction on CPU while GPU finishes the prior slot
        ClientLevel level = Minecraft.getInstance().level;
        float partialTicks = deltaTracker != null ? deltaTracker.getGameTimeDeltaPartialTick(true) : 0f;
        float timeOfDay = (level != null) ? level.getTimeOfDay(partialTicks) : 0.25f;

        float sunAngle = timeOfDay * (float) (Math.PI * 2.0);
        Vector3f sunDir = new Vector3f((float) -Math.sin(sunAngle), (float) Math.cos(sunAngle), 0.25f).normalize();

        // frustumMatrix from GameRenderer already holds camera rotation inverse (world -> view rotation)
        Matrix4f viewMatrix = new Matrix4f(frustumMatrix).translate(-cameraPos.x, -cameraPos.y, -cameraPos.z);
        Matrix4f projMatrix = toVulkanReversedZ(projectionMatrix);
        Matrix4f viewProjMatrix = new Matrix4f(projMatrix).mul(viewMatrix);
        Matrix4f invViewProjMatrix = new Matrix4f(viewProjMatrix).invert();

        // Compute 3 CSM Cascade Matrices
        computeCsmMatrices(cameraPos, sunDir, viewMatrix);

        ShaderUniformLayout.UniformData shaderPackUniforms = null;
        if (isShaderPackActive()) {
            shaderPackUniforms = buildShaderPackUniformData(
                    level, camera, cameraPos, frustumMatrix, projectionMatrix, partialTicks
            );
        }

        // 2. Wait on this frame slot's fence ONLY if compositeToOpenGL didn't already wait for it
        int slot = currentFrameSlot;
        VkDevice device = vkContext.getVkDevice();
        long cpFenceT0 = CHUNK_PROFILE ? System.nanoTime() : 0L;
        if (!fenceWaited[slot]) {
            int waitRes = vkWaitForFences(device, frameFences[slot], true, 1_000_000_000L);
            if (waitRes != VK_SUCCESS) {
                LOGGER.error("[Valkyrie] vkWaitForFences failed/timed out on slot {} (err={}); skipping frame submit to avoid fence reset race.", slot, waitRes);
                return;
            }
            fenceWaited[slot] = true;
        }
        // B4: When GL semaphore wasn't signaled (or no interop), ensure the prior frame finished writing shared images before reusing them
        if (lastSubmittedSlot != slot && !fenceWaited[lastSubmittedSlot] && !vkInteropManager.isGlSemaphoreSignaled()) {
            int prevWaitRes = vkWaitForFences(device, frameFences[lastSubmittedSlot], true, 1_000_000_000L);
            if (prevWaitRes == VK_SUCCESS) {
                fenceWaited[lastSubmittedSlot] = true;
            }
        }
        long cpFenceT1 = CHUNK_PROFILE ? System.nanoTime() : 0L;
        vkResetFences(device, frameFences[slot]);
        for (int w = 0; w < RECORD_WORKER_COUNT; w++) {
            if (workerCommandPools[w][slot] != VK_NULL_HANDLE) {
                vkResetCommandPool(device, workerCommandPools[w][slot], 0);
            }
        }
        long cpUploadT0 = CHUNK_PROFILE ? System.nanoTime() : 0L;
        vkMegaBuffer.beginFrameSlot(slot, FRAMES_IN_FLIGHT);

        // 3. Stream polled meshing results into MegaBuffer (opaque, translucent, and half-scale VoxelShape sub-meshes)
        int uploadedThisFrame = 0;
        if (completedMeshes != null) {
            for (GreedyMesher.MeshedSectionResult result : completedMeshes) {
                int shadowFlag = result.flags() & 2;
                vkMegaBuffer.uploadSectionSubMesh(
                        result.sectionX(), result.sectionY(), result.sectionZ(), 0,
                        result.packedVertices(), result.quadCount(), result.aabbPacked(), shadowFlag
                );
                vkMegaBuffer.uploadSectionSubMesh(
                        result.sectionX(), result.sectionY(), result.sectionZ(), 1,
                        result.translucentVertices(), result.translucentQuadCount(), result.translucentAabbPacked(), 1 | shadowFlag
                );
                vkMegaBuffer.uploadSectionSubMesh(
                        result.sectionX(), result.sectionY(), result.sectionZ(), 2,
                        result.halfScaleOpaqueVertices(), result.halfScaleOpaqueQuadCount(), result.halfScaleOpaqueAabbPacked(), 4 | shadowFlag
                );
                vkMegaBuffer.uploadSectionSubMesh(
                        result.sectionX(), result.sectionY(), result.sectionZ(), 3,
                        result.halfScaleTranslucentVertices(), result.halfScaleTranslucentQuadCount(), result.halfScaleTranslucentAabbPacked(), 5 | shadowFlag
                );
                dirtyShadow = true;
                uploadedThisFrame++;
            }
        }
        if (CHUNK_PROFILE && chunkProfStartNs != 0L) {
            long cpNow = System.nanoTime();
            chunkProfDrainNs += (cpT1 - cpT0);
            chunkProfPollNs += (cpT2 - cpT1);
            chunkProfFenceNs += (cpFenceT1 - cpFenceT0);
            chunkProfUploadNs += (cpNow - cpUploadT0);
            chunkProfUploadBytes += vkMegaBuffer.getStagingUsedBytes(slot);
            chunkProfCaptured += capturedThisFrame;
            chunkProfUploaded += uploadedThisFrame;
            chunkProfFrames++;

            int dirtyRem = dirtySections.size();
            int inFlightRem = greedyMesher.getInFlightCount();
            int compRem = greedyMesher.getCompletedCount();
            int defRem = vkMegaBuffer.getDeferredUploadCount();
            boolean queuesEmpty = (dirtyRem == 0 && inFlightRem == 0 && compRem == 0 && defRem == 0);
            if (!chunkProfSettled && (cpNow - chunkProfLastLogNs >= 1_000_000_000L || (queuesEmpty && chunkProfCaptured > 0))) {
                double elapsedMs = (cpNow - chunkProfStartNs) / 1_000_000.0;
                double intervalSec = Math.max(0.001, (cpNow - chunkProfLastLogNs) / 1_000_000_000.0);
                double fps = chunkProfFrames / intervalSec;
                LOGGER.info("[ChunkProfile] t={}ms fps={} active={} dirty={} inFlight={} comp={} deferred={} | cap={} upl={} ({}MB) | drain={}ms poll={}ms fence={}ms upload={}ms",
                        String.format("%.0f", elapsedMs),
                        String.format("%.1f", fps),
                        vkMegaBuffer.getActiveSectionCount(),
                        dirtyRem, inFlightRem, compRem, defRem,
                        chunkProfCaptured, chunkProfUploaded,
                        String.format("%.1f", chunkProfUploadBytes / (1024.0 * 1024.0)),
                        String.format("%.1f", chunkProfDrainNs / 1_000_000.0),
                        String.format("%.1f", chunkProfPollNs / 1_000_000.0),
                        String.format("%.1f", chunkProfFenceNs / 1_000_000.0),
                        String.format("%.1f", chunkProfUploadNs / 1_000_000.0));
                chunkProfLastLogNs = cpNow;
                chunkProfFrames = 0;
                chunkProfDrainNs = 0L;
                chunkProfPollNs = 0L;
                chunkProfFenceNs = 0L;
                chunkProfUploadNs = 0L;
                if (queuesEmpty && chunkProfCaptured > 100) {
                    chunkProfSettled = true;
                }
            }
        }

        // 4. Refresh Block Atlas UVs & Texture if dirty
        if (atlasUVTable.isDirty()) {
            syncAtlasResources();
            updateDescriptorSet0Buffers();
        }

        // Determine CSM cascades to update this frame and keep renderedCsmMatrices locked to rendered depth
        boolean useBuiltinCsm = !isShaderPackActive()
                && vkMegaBuffer.getActiveSectionCount() > 0
                && vkMegaBuffer.getSlotDispatchLimit() > 0;
        this.shadowStaticDirtyThisFrame = evaluateStaticShadowState(
                level, cameraPos, viewMatrix, sunDir, slot, uploadedThisFrame > 0);
        if (this.shadowStaticDirtyThisFrame && isShaderPackActive() && vkShaderPackPipeline != null) {
            vkShaderPackPipeline.invalidateShadow();
        }
        List<Integer> csmCascadesToUpdate = collectCsmCascadesToUpdate(useBuiltinCsm);

        int dimFlags = 0;
        if (level != null) {
            if (level.dimension() == net.minecraft.world.level.Level.END) {
                dimFlags |= 16;
            } else if (!level.dimensionType().hasSkyLight()) {
                dimFlags |= 32;
            }
        }
        vkPipelineManager.setDimensionRenderFlags(dimFlags);

        // Update Frame Uniform Buffer (UBO) for this frame slot
        vkPipelineManager.updateFrameUbo(
                viewMatrix,
                projMatrix,
                viewProjMatrix,
                invViewProjMatrix,
                prevViewProj,
                renderedCsmMatrices,
                sunDir,
                cameraPos,
                width,
                height,
                frameIndex,
                (float) (frameIndex * 0.016666),
                slot
        );

        prevViewProj.set(viewProjMatrix);

        // 5. Stage A: Async Transfer Queue submission when dedicated transfer queue is available
        boolean asyncTransferSubmitted = false;
        if (vkContext.hasDedicatedTransferQueue() && vkMegaBuffer.hasPendingTransfers(slot)) {
            VkCommandBuffer transferCmd = transferCommandBuffers[slot];
            try (MemoryStack stack = MemoryStack.stackPush()) {
                vkResetCommandBuffer(transferCmd, 0);
                VkCommandBufferBeginInfo tBegin = VkCommandBufferBeginInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                        .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
                vkBeginCommandBuffer(transferCmd, tBegin);
                boolean recorded = vkMegaBuffer.recordPendingTransfersForSlot(
                        slot, transferCmd, vkContext.hasSeparateTransferQueueFamily());
                vkEndCommandBuffer(transferCmd);

                if (recorded) {
                    PointerBuffer pTransferCmd = stack.pointers(transferCmd);
                    LongBuffer pTransferSig = stack.longs(transferCompleteSemaphores[slot]);
                    VkSubmitInfo tSubmit = VkSubmitInfo.calloc(stack)
                            .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                            .pCommandBuffers(pTransferCmd)
                            .pSignalSemaphores(pTransferSig);
                    int tRes = vkQueueSubmit(vkContext.getTransferQueue(), tSubmit, VK_NULL_HANDLE);
                    if (tRes == VK_SUCCESS) {
                        asyncTransferSubmitted = true;
                    } else {
                        LOGGER.error("[Valkyrie] Transfer queue submit failed: {}", tRes);
                    }
                }
            }
        }

        // 5b. Stage C: Async Compute Queue chunk culling when dedicated compute queue is available
        int activeSections = vkMegaBuffer.getActiveSectionCount();
        int slotDispatchLimit = vkMegaBuffer.getSlotDispatchLimit();
        boolean useShaderPack = shaderPackUniforms != null && isShaderPackActive();
        boolean useBuiltinPbr = activeSections > 0 && slotDispatchLimit > 0 && !useShaderPack;
        boolean asyncComputeEnabled = vkContext.hasDedicatedComputeQueue() && useBuiltinPbr && !VkGpuProfiler.isEnabled();
        boolean asyncComputeCullSubmitted = false;
        ValkyrieConfig config = ValkyrieConfig.get();
        boolean useHiZ = hasPreviousFrameHiZ && config.isHiZOcclusionEnabled();

        if (asyncComputeEnabled) {
            VkCommandBuffer cullCmd = cullComputeCommandBuffers[slot];
            try (MemoryStack stack = MemoryStack.stackPush()) {
                vkResetCommandBuffer(cullCmd, 0);
                VkCommandBufferBeginInfo cBegin = VkCommandBufferBeginInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                        .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
                vkBeginCommandBuffer(cullCmd, cBegin);
                long pipeLayout = vkPipelineManager.getPipelineLayout();
                LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(slot), vkPipelineManager.getDescriptorSet1());
                recordChunkCullCompute(
                        cullCmd, pipeLayout, pSets, slotDispatchLimit, useHiZ, 1,
                        vkContext.hasSeparateComputeQueueFamily(), stack);
                vkEndCommandBuffer(cullCmd);

                PointerBuffer pCullCmd = stack.pointers(cullCmd);
                LongBuffer pCullSig = stack.longs(cullCompleteSemaphores[slot]);
                VkSubmitInfo cSubmit = VkSubmitInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                        .pCommandBuffers(pCullCmd)
                        .pSignalSemaphores(pCullSig);
                if (hizSemaphorePending) {
                    LongBuffer pWaitHiz = stack.longs(hizReadySemaphore);
                    IntBuffer pWaitMask = stack.ints(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                    cSubmit.waitSemaphoreCount(1)
                            .pWaitSemaphores(pWaitHiz)
                            .pWaitDstStageMask(pWaitMask);
                    hizSemaphorePending = false;
                }
                int cRes = vkQueueSubmit(vkContext.getComputeQueue(), cSubmit, VK_NULL_HANDLE);
                if (cRes == VK_SUCCESS) {
                    asyncComputeCullSubmitted = true;
                } else {
                    LOGGER.error("[Valkyrie] Compute queue cull submit failed: {}", cRes);
                }
            }
        }

        // 5c. Record Graphics Command Buffer(s) (Stage B: multi-threaded independent primary CBs for builtin PBR when >= 2 passes, else serial)
        boolean asyncHiZOnCompute = asyncComputeEnabled && config.isHiZOcclusionEnabled();
        List<VkCommandBuffer> submittedCmds = recordFrameCommands(
                slot, width, height, shaderPackUniforms, csmCascadesToUpdate, asyncTransferSubmitted,
                asyncComputeCullSubmitted, asyncHiZOnCompute);

        // B4: Track actual final Vulkan image layouts before GL waitSemaphoreEXT acquires them
        vkInteropManager.setVkExportedLayouts(
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        );

        // 6. Submit Command Buffer(s) (using external/transfer/compute semaphores when active, else fence sync in composite)
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer pCmd = stack.mallocPointer(submittedCmds.size());
            for (int i = 0; i < submittedCmds.size(); i++) {
                pCmd.put(i, submittedCmds.get(i).address());
            }

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(pCmd);

            boolean waitGl = vkInteropManager.usesExternalSemaphores() && vkInteropManager.consumeGlSemaphoreSignaled();
            int waitCount = (asyncTransferSubmitted ? 1 : 0) + (asyncComputeCullSubmitted ? 1 : 0) + (waitGl ? 1 : 0);
            if (waitCount > 0) {
                LongBuffer pWaitSemaphores = stack.mallocLong(waitCount);
                IntBuffer pWaitDstStageMask = stack.mallocInt(waitCount);
                int idx = 0;
                if (asyncTransferSubmitted) {
                    pWaitSemaphores.put(idx, transferCompleteSemaphores[slot]);
                    pWaitDstStageMask.put(idx, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                    idx++;
                }
                if (asyncComputeCullSubmitted) {
                    pWaitSemaphores.put(idx, cullCompleteSemaphores[slot]);
                    pWaitDstStageMask.put(idx, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
                    idx++;
                }
                if (waitGl) {
                    pWaitSemaphores.put(idx, vkInteropManager.getVkGlReadySemaphore());
                    pWaitDstStageMask.put(idx, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                            | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                            | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
                }
                submitInfo.waitSemaphoreCount(waitCount)
                        .pWaitSemaphores(pWaitSemaphores)
                        .pWaitDstStageMask(pWaitDstStageMask);
            }

            boolean sigExt = vkInteropManager.usesExternalSemaphores();
            int sigCount = (sigExt ? 1 : 0) + (asyncHiZOnCompute ? 1 : 0);
            if (sigCount > 0) {
                LongBuffer pSignalSemaphores = stack.mallocLong(sigCount);
                int sIdx = 0;
                if (sigExt) {
                    pSignalSemaphores.put(sIdx++, vkInteropManager.getVkRenderCompleteSemaphore());
                }
                if (asyncHiZOnCompute) {
                    pSignalSemaphores.put(sIdx, gbufferDoneSemaphores[slot]);
                }
                submitInfo.pSignalSemaphores(pSignalSemaphores);
            }

            int res = vkQueueSubmit(vkContext.getGraphicsQueue(), submitInfo, frameFences[slot]);
            if (res != VK_SUCCESS) {
                LOGGER.error("[Valkyrie] Failed to submit render command buffer: {}", res);
            } else {
                fenceWaited[slot] = false;
            }
        }

        // 6b. Stage C: Submit async Hi-Z downsample on computeQueue after graphicsQueue signals gbufferDoneSemaphores[slot]
        if (asyncHiZOnCompute) {
            VkCommandBuffer hizCmd = hizComputeCommandBuffers[slot];
            try (MemoryStack stack = MemoryStack.stackPush()) {
                vkResetCommandBuffer(hizCmd, 0);
                VkCommandBufferBeginInfo hBegin = VkCommandBufferBeginInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                        .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
                vkBeginCommandBuffer(hizCmd, hBegin);
                renderHiZPyramid(hizCmd, width, height, stack);
                vkEndCommandBuffer(hizCmd);

                PointerBuffer pHizCmd = stack.pointers(hizCmd);
                LongBuffer pWaitGfx = stack.longs(gbufferDoneSemaphores[slot]);
                IntBuffer pWaitMask = stack.ints(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                LongBuffer pSigHiz = stack.longs(hizReadySemaphore);
                VkSubmitInfo hSubmit = VkSubmitInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                        .waitSemaphoreCount(1)
                        .pWaitSemaphores(pWaitGfx)
                        .pWaitDstStageMask(pWaitMask)
                        .pCommandBuffers(pHizCmd)
                        .pSignalSemaphores(pSigHiz);
                int hRes = vkQueueSubmit(vkContext.getComputeQueue(), hSubmit, VK_NULL_HANDLE);
                if (hRes == VK_SUCCESS) {
                    hasPreviousFrameHiZ = true;
                    hizSemaphorePending = true;
                } else {
                    LOGGER.error("[Valkyrie] Compute queue Hi-Z submit failed: {}", hRes);
                }
            }
        }

        lastSubmittedSlot = slot;
        currentFrameSlot = (slot + 1) % FRAMES_IN_FLIGHT;
        dirtyShadow = false;
        frameIndex++;
    }

    private List<VkCommandBuffer> recordFrameCommands(
            int frameSlot,
            int width,
            int height,
            ShaderUniformLayout.UniformData shaderPackUniforms,
            List<Integer> csmCascades,
            boolean asyncTransferSubmitted,
            boolean asyncComputeCullSubmitted,
            boolean asyncHiZOnCompute
    ) {
        int activeSections = vkMegaBuffer.getActiveSectionCount();
        int slotDispatchLimit = vkMegaBuffer.getSlotDispatchLimit();
        boolean useShaderPack = shaderPackUniforms != null && isShaderPackActive();

        // Stage B: Parallelize independent CSM cascades + Opaque GBuffer into per-worker primary CBs
        if (activeSections > 0 && slotDispatchLimit > 0 && !useShaderPack && RECORD_WORKER_COUNT > 1 && recordExecutor != null) {
            // Only dispatch to worker threads when there are >= 2 independent passes
            int totalParallelPasses = csmCascades.size() + 1; // cascades + opaque GBuffer
            if (totalParallelPasses >= 2) {
                return recordBuiltinPbrParallel(
                        frameSlot, width, height, slotDispatchLimit,
                        csmCascades, asyncTransferSubmitted, asyncComputeCullSubmitted, asyncHiZOnCompute);
            }
        }

        // Serial fallback (shaderpack, phase2 fallback, or <2 independent passes)
        VkCommandBuffer commandBuffer = commandBuffers[frameSlot];
        recordFrameCommandsSerial(commandBuffer, width, height, shaderPackUniforms, csmCascades, frameSlot, asyncTransferSubmitted);
        return List.of(commandBuffer);
    }

    private List<VkCommandBuffer> recordBuiltinPbrParallel(
            int frameSlot,
            int width,
            int height,
            int slotDispatchLimit,
            List<Integer> csmCascades,
            boolean asyncTransferSubmitted,
            boolean asyncComputeCullSubmitted,
            boolean asyncHiZOnCompute
    ) {
        ValkyrieConfig config = ValkyrieConfig.get();
        boolean useHiZ = hasPreviousFrameHiZ && config.isHiZOcclusionEnabled();
        boolean hasTranslucent = vkMegaBuffer.hasTranslucentSections();
        long sharedColorImage = vkInteropManager.getSharedColorImage();
        long sharedDepthImage = vkInteropManager.getSharedDepthImage();
        long sharedDepthImageView = vkInteropManager.getSharedDepthImageView();

        logActivePipelineIfChanged("builtin-pbr");

        // 1. Pre-pass CB on main thread: transfer/acquire + opaque chunk cull (or compute cull acquire barrier)
        VkCommandBuffer preCmd = commandBuffers[frameSlot];
        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkResetCommandBuffer(preCmd, 0);
            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            vkBeginCommandBuffer(preCmd, beginInfo);
            gpuProfiler.beginFrame(vkContext.getVkDevice(), frameSlot, preCmd, "builtin-pbr");

            if (asyncTransferSubmitted) {
                vkMegaBuffer.recordTransferAcquireBarrier(preCmd);
            } else {
                vkMegaBuffer.recordPendingTransfersForSlot(frameSlot, preCmd, false);
            }

            if (asyncComputeCullSubmitted) {
                recordComputeCullAcquireBarrier(preCmd, stack);
            } else {
                long pipeLayout = vkPipelineManager.getPipelineLayout();
                LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(frameSlot), vkPipelineManager.getDescriptorSet1());
                int qCull = gpuProfiler.beginPass(preCmd, frameSlot, "chunk_cull");
                dispatchChunkCull(preCmd, pipeLayout, pSets, slotDispatchLimit, useHiZ, 1, stack);
                gpuProfiler.endPass(preCmd, frameSlot, qCull);
            }
            vkEndCommandBuffer(preCmd);
        }

        // 2. Build ordered independent pass descriptors (CSM cascades + Opaque GBuffer)
        int totalTasks = csmCascades.size() + 1;
        VkCommandBuffer[] orderedParallelCmds = new VkCommandBuffer[totalTasks];
        int usedWorkers = Math.min(RECORD_WORKER_COUNT, totalTasks);
        List<CompletableFuture<Void>> futures = new ArrayList<>(usedWorkers);

        for (int w = 0; w < usedWorkers; w++) {
            final int workerIdx = w;
            futures.add(CompletableFuture.runAsync(() -> {
                int localCmdIdx = 0;
                for (int t = workerIdx; t < totalTasks; t += usedWorkers) {
                    VkCommandBuffer wCmd = workerCommandBuffers[workerIdx][frameSlot][localCmdIdx++];
                    orderedParallelCmds[t] = wCmd;
                    try (MemoryStack stack = MemoryStack.stackPush()) {
                        VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
                        vkBeginCommandBuffer(wCmd, beginInfo);
                        if (t < csmCascades.size()) {
                            int cascade = csmCascades.get(t);
                            int qCsm = gpuProfiler.beginPass(wCmd, frameSlot, "csm_c" + cascade);
                            renderCsmCascade(wCmd, cascade, slotDispatchLimit, frameSlot, stack);
                            gpuProfiler.endPass(wCmd, frameSlot, qCsm);
                        } else {
                            int qGb = gpuProfiler.beginPass(wCmd, frameSlot, "gbuffer_opaque");
                            renderGBuffer(wCmd, width, height, slotDispatchLimit, frameSlot, false, !hasTranslucent, stack);
                            gpuProfiler.endPass(wCmd, frameSlot, qGb);
                        }
                        vkEndCommandBuffer(wCmd);
                    }
                }
            }, recordExecutor));
        }

        // 3. While workers record CSM + Opaque GBuffer, record post-GBuffer passes on main thread
        VkCommandBuffer postCmd = postCommandBuffers[frameSlot];
        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkResetCommandBuffer(postCmd, 0);
            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            vkBeginCommandBuffer(postCmd, beginInfo);

            long pipeLayout = vkPipelineManager.getPipelineLayout();
            LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(frameSlot), vkPipelineManager.getDescriptorSet1());
            if (hasTranslucent) {
                int qTrans = gpuProfiler.beginPass(postCmd, frameSlot, "gbuffer_translucent");
                dispatchChunkCull(postCmd, pipeLayout, pSets, slotDispatchLimit, useHiZ, 2, stack);
                renderGBuffer(postCmd, width, height, slotDispatchLimit, frameSlot, true, true, stack);
                gpuProfiler.endPass(postCmd, frameSlot, qTrans);
            }
            int qLight = gpuProfiler.beginPass(postCmd, frameSlot, "deferred_lighting");
            renderDeferredLighting(postCmd, width, height, frameSlot, stack);
            gpuProfiler.endPass(postCmd, frameSlot, qLight);
            if (!asyncHiZOnCompute) {
                if (config.isHiZOcclusionEnabled()) {
                    int qHiz = gpuProfiler.beginPass(postCmd, frameSlot, "hiz_downsample");
                    renderHiZPyramid(postCmd, width, height, stack);
                    gpuProfiler.endPass(postCmd, frameSlot, qHiz);
                    hasPreviousFrameHiZ = true;
                } else {
                    hasPreviousFrameHiZ = false;
                }
            }
            int qPost = gpuProfiler.beginPass(postCmd, frameSlot, "post_composite");
            renderPostComposite(postCmd, width, height, sharedColorImage, frameSlot, stack);
            gpuProfiler.endPass(postCmd, frameSlot, qPost);
            int qDepth = gpuProfiler.beginPass(postCmd, frameSlot, "depth_convert");
            renderDepthConvert(postCmd, width, height, sharedDepthImage, sharedDepthImageView, frameSlot, stack);
            gpuProfiler.endPass(postCmd, frameSlot, qDepth);
            vkEndCommandBuffer(postCmd);
        }

        for (CompletableFuture<Void> f : futures) {
            f.join();
        }

        List<VkCommandBuffer> result = new ArrayList<>(totalTasks + 2);
        result.add(preCmd);
        for (VkCommandBuffer c : orderedParallelCmds) {
            result.add(c);
        }
        result.add(postCmd);
        return result;
    }

    private void recordFrameCommandsSerial(
            VkCommandBuffer commandBuffer,
            int width,
            int height,
            ShaderUniformLayout.UniformData shaderPackUniforms,
            List<Integer> csmCascades,
            int frameSlot,
            boolean asyncTransferSubmitted
    ) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkResetCommandBuffer(commandBuffer, 0);

            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

            vkBeginCommandBuffer(commandBuffer, beginInfo);

            // Staging ring -> Device Local SSBO copy (either acquire from async transfer queue, or inline on single-queue fallback)
            if (asyncTransferSubmitted) {
                vkMegaBuffer.recordTransferAcquireBarrier(commandBuffer);
            } else {
                vkMegaBuffer.recordPendingTransfersForSlot(frameSlot, commandBuffer, false);
            }

            int activeSections = vkMegaBuffer.getActiveSectionCount();
            int slotDispatchLimit = vkMegaBuffer.getSlotDispatchLimit();

            if (activeSections > 0 && slotDispatchLimit > 0) {
                if (shaderPackUniforms != null && isShaderPackActive()) {
                    try {
                        String packLabel = "shaderpack:" + vkShaderPackPipeline.getActivePackName();
                        logActivePipelineIfChanged(packLabel);
                        gpuProfiler.beginFrame(vkContext.getVkDevice(), frameSlot, commandBuffer, packLabel);
                        vkShaderPackPipeline.setGpuProfiler(gpuProfiler, frameSlot);
                        vkShaderPackPipeline.recordFrame(
                                commandBuffer,
                                vkInteropManager.getSharedColorImage(),
                                vkInteropManager.getSharedColorImageView(),
                                vkInteropManager.getSharedDepthImage(),
                                width,
                                height,
                                shaderPackUniforms
                        );
                        hasPreviousFrameHiZ = false;
                    } catch (Throwable t) {
                        LOGGER.error("[Valkyrie] Native shaderpack frame failed; resetting command buffer and falling back to built-in deferred PBR pipeline.", t);
                        explicitPackLoaded = false;
                        vkShaderPackPipeline.unloadPack();
                        logActivePipelineIfChanged("builtin-pbr");
                        vkResetCommandBuffer(commandBuffer, 0);
                        vkBeginCommandBuffer(commandBuffer, beginInfo);
                        gpuProfiler.beginFrame(vkContext.getVkDevice(), frameSlot, commandBuffer, "builtin-pbr");
                        recordGpuDrivenTerrainPasses(
                                commandBuffer, width, height, slotDispatchLimit,
                                vkInteropManager.getSharedColorImage(),
                                vkInteropManager.getSharedDepthImage(),
                                vkInteropManager.getSharedDepthImageView(),
                                csmCascades,
                                frameSlot,
                                stack
                        );
                    }
                } else {
                    logActivePipelineIfChanged("builtin-pbr");
                    gpuProfiler.beginFrame(vkContext.getVkDevice(), frameSlot, commandBuffer, "builtin-pbr");
                    recordGpuDrivenTerrainPasses(
                            commandBuffer, width, height, slotDispatchLimit,
                            vkInteropManager.getSharedColorImage(),
                            vkInteropManager.getSharedDepthImage(),
                            vkInteropManager.getSharedDepthImageView(),
                            csmCascades,
                            frameSlot,
                            stack
                    );
                }
            } else {
                // Phase 2 fallback: animated sky/horizon procedural gradient
                hasPreviousFrameHiZ = false;
                logActivePipelineIfChanged("phase2-fallback");
                gpuProfiler.beginFrame(vkContext.getVkDevice(), frameSlot, commandBuffer, "phase2-fallback");
                recordPhase2FallbackPass(commandBuffer, width, height, stack);
            }

            vkEndCommandBuffer(commandBuffer);
        }
    }

    /**
     * Records a full GPU-driven deferred frame against caller-supplied color/depth images
     * (used by headless verification without an OpenGL interop context).
     */
    public synchronized void recordHeadlessGpuDrivenFrame(
            VkCommandBuffer cmd,
            int width,
            int height,
            Vector3f cameraPos,
            Vector3f lookTarget,
            Matrix4f glProjection,
            long sharedColorImage,
            long sharedDepthImage,
            long sharedDepthImageView,
            boolean resetTemporalState
    ) {
        recordHeadlessGpuDrivenFrame(
                cmd,
                width,
                height,
                cameraPos,
                lookTarget,
                glProjection,
                sharedColorImage,
                VK_NULL_HANDLE,
                sharedDepthImage,
                sharedDepthImageView,
                resetTemporalState
        );
    }

    /**
     * Overload accepting {@code sharedColorImageView} so headless verification can also exercise
     * an active {@link VkShaderPackPipeline} with automatic fallback to built-in deferred PBR.
     */
    public synchronized void recordHeadlessGpuDrivenFrame(
            VkCommandBuffer cmd,
            int width,
            int height,
            Vector3f cameraPos,
            Vector3f lookTarget,
            Matrix4f glProjection,
            long sharedColorImage,
            long sharedColorImageView,
            long sharedDepthImage,
            long sharedDepthImageView,
            boolean resetTemporalState
    ) {
        this.vkContext = VkContext.getInstance();
        this.vkMegaBuffer = VkMegaBuffer.getInstance();
        this.vkPipelineManager = VkPipelineManager.getInstance();
        if (resetTemporalState) {
            this.frameIndex = 0;
            this.hasPreviousFrameHiZ = false;
            this.dirtyShadow = true;
            this.hasSmoothedEyeBrightness = false;
            this.hasSmoothedWetness = false;
            this.lastShaderPackFrameNanos = 0L;
            this.shaderPackFrameTimeCounter = 0.0f;
            this.prevViewProj.identity();
            if (this.vkShaderPackPipeline != null && this.vkShaderPackPipeline.isInitialized()) {
                this.vkShaderPackPipeline.resetTemporalState();
            }
        }

        Vector3f sunDir = new Vector3f(0.4f, 0.85f, 0.35f).normalize();
        Matrix4f viewMatrix = new Matrix4f().lookAt(cameraPos, lookTarget, new Vector3f(0f, 1f, 0f));
        Matrix4f projMatrix = toVulkanReversedZ(glProjection);
        Matrix4f viewProjMatrix = new Matrix4f(projMatrix).mul(viewMatrix);
        Matrix4f invViewProjMatrix = new Matrix4f(viewProjMatrix).invert();
        if (resetTemporalState) {
            this.prevViewProj.set(viewProjMatrix);
        }

        computeCsmMatrices(cameraPos, sunDir, viewMatrix);
        List<Integer> csmCascadesToUpdate = collectCsmCascadesToUpdate(!isShaderPackActive());
        vkPipelineManager.updateFrameUbo(
                viewMatrix, projMatrix, viewProjMatrix, invViewProjMatrix, prevViewProj,
                renderedCsmMatrices, sunDir, cameraPos, width, height, frameIndex, (float) (frameIndex * 0.016666)
        );
        prevViewProj.set(viewProjMatrix);

        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkMegaBuffer.recordPendingTransfers(cmd);
            int slotDispatchLimit = vkMegaBuffer.getSlotDispatchLimit();
            if (slotDispatchLimit > 0) {
                if (sharedColorImageView != VK_NULL_HANDLE && isShaderPackActive()) {
                    try {
                        logActivePipelineIfChanged("shaderpack:" + vkShaderPackPipeline.getActivePackName());
                        ShaderUniformLayout.UniformData uniforms = new ShaderUniformLayout.UniformData();
                        uniforms.gbufferModelView.set(viewMatrix).m30(0.0f).m31(0.0f).m32(0.0f).m33(1.0f);
                        uniforms.gbufferProjection.set(glProjection);
                        uniforms.cameraPosition.set(cameraPos);
                        uniforms.near = 0.05f;
                        uniforms.far = 256.0f;
                        uniforms.sunAngle = 0.25f;
                        uniforms.shadowAngle = 0.25f;
                        uniforms.worldTime = 6000;
                        uniforms.worldDay = 0;
                        uniforms.moonPhase = 0;
                        uniforms.skyColor.set(0.47f, 0.67f, 1.0f);
                        uniforms.fogColor.set(0.75f, 0.85f, 1.0f);
                        uniforms.fogStart = 128.0f;
                        uniforms.fogEnd = 256.0f;
                        uniforms.fogDensity = 1.0f / 128.0f;
                        uniforms.eyeBrightness.set(240, 240);
                        uniforms.eyeBrightnessSmooth.set(240, 240);
                        uniforms.screenBrightness = 1.0f;
                        uniforms.frameCounter = (int) frameIndex;
                        uniforms.frameTime = 1.0f / 60.0f;
                        uniforms.frameTimeCounter = (float) (frameIndex * (1.0 / 60.0));
                        populateCelestialVectors(uniforms, uniforms.gbufferModelView, sunDir);
                        vkShaderPackPipeline.recordFrame(
                                cmd, sharedColorImage, sharedColorImageView, sharedDepthImage, width, height, uniforms
                        );
                        hasPreviousFrameHiZ = false;
                    } catch (Throwable t) {
                        LOGGER.error("[Valkyrie] Headless shaderpack frame failed; falling back to built-in deferred PBR pipeline.", t);
                        explicitPackLoaded = false;
                        vkShaderPackPipeline.unloadPack();
                        logActivePipelineIfChanged("builtin-pbr");
                        recordGpuDrivenTerrainPasses(
                                cmd, width, height, slotDispatchLimit,
                                sharedColorImage, sharedDepthImage, sharedDepthImageView,
                                csmCascadesToUpdate, 0, stack
                        );
                    }
                } else {
                    logActivePipelineIfChanged("builtin-pbr");
                    recordGpuDrivenTerrainPasses(
                            cmd, width, height, slotDispatchLimit,
                            sharedColorImage, sharedDepthImage, sharedDepthImageView,
                            csmCascadesToUpdate, 0, stack
                    );
                }
            }
        }
        dirtyShadow = false;
        frameIndex++;
    }

    private void recordGpuDrivenTerrainPasses(
            VkCommandBuffer cmd,
            int width,
            int height,
            int slotDispatchLimit,
            long sharedColorImage,
            long sharedDepthImage,
            long sharedDepthImageView,
            List<Integer> csmCascades,
            int frameSlot,
            MemoryStack stack
    ) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(frameSlot), vkPipelineManager.getDescriptorSet1());

        ValkyrieConfig config = ValkyrieConfig.get();
        boolean useHiZ = hasPreviousFrameHiZ && config.isHiZOcclusionEnabled();

        // --------------------------------------------------------------------
        // Pass 1: Clear DrawCountSSBO & Compute Frustum + Hi-Z Culling (Opaque passFlagFilter=1)
        // --------------------------------------------------------------------
        int qCull = gpuProfiler.beginPass(cmd, frameSlot, "chunk_cull");
        dispatchChunkCull(cmd, pipeLayout, pSets, slotDispatchLimit, useHiZ, 1, stack);
        gpuProfiler.endPass(cmd, frameSlot, qCull);

        // --------------------------------------------------------------------
        // Pass 2: Cascaded Shadow Map (CSM) Dynamic Rendering Pass
        // --------------------------------------------------------------------
        if (config.isCsmEnabled() && !config.isVanillaLighting() && csmCascades != null) {
            for (int c : csmCascades) {
                int qCsm = gpuProfiler.beginPass(cmd, frameSlot, "csm_c" + c);
                renderCsmCascade(cmd, c, slotDispatchLimit, frameSlot, stack);
                gpuProfiler.endPass(cmd, frameSlot, qCsm);
            }
        }

        boolean hasTranslucent = vkMegaBuffer.hasTranslucentSections();

        // --------------------------------------------------------------------
        // Pass 3a: Opaque G-Buffer Dynamic Rendering Pass (gbufferPipeline)
        // --------------------------------------------------------------------
        int qGbOpaque = gpuProfiler.beginPass(cmd, frameSlot, "gbuffer_opaque");
        renderGBuffer(cmd, width, height, slotDispatchLimit, frameSlot, false, !hasTranslucent, stack);
        gpuProfiler.endPass(cmd, frameSlot, qGbOpaque);

        // --------------------------------------------------------------------
        // Pass 3b: Translucent Culling & G-Buffer Pass (passFlagFilter=2, translucentGbufferPipeline)
        // --------------------------------------------------------------------
        if (hasTranslucent) {
            int qGbTrans = gpuProfiler.beginPass(cmd, frameSlot, "gbuffer_translucent");
            dispatchChunkCull(cmd, pipeLayout, pSets, slotDispatchLimit, useHiZ, 2, stack);
            renderGBuffer(cmd, width, height, slotDispatchLimit, frameSlot, true, true, stack);
            gpuProfiler.endPass(cmd, frameSlot, qGbTrans);
        }

        // --------------------------------------------------------------------
        // Pass 4: Deferred Lighting & Water SSR Compute Pass (deferred_lighting.comp)
        // --------------------------------------------------------------------
        int qLight = gpuProfiler.beginPass(cmd, frameSlot, "deferred_lighting");
        renderDeferredLighting(cmd, width, height, frameSlot, stack);
        gpuProfiler.endPass(cmd, frameSlot, qLight);

        // --------------------------------------------------------------------
        // Pass 4b: Hi-Z Occlusion Pyramid Downsample for Next Frame (hiz_downsample.comp)
        // --------------------------------------------------------------------
        if (config.isHiZOcclusionEnabled()) {
            int qHiz = gpuProfiler.beginPass(cmd, frameSlot, "hiz_downsample");
            renderHiZPyramid(cmd, width, height, stack);
            gpuProfiler.endPass(cmd, frameSlot, qHiz);
            hasPreviousFrameHiZ = true;
        } else {
            hasPreviousFrameHiZ = false;
        }

        // --------------------------------------------------------------------
        // Pass 5: Post-Process ACES Filmic Tonemapping (post_composite.comp)
        // --------------------------------------------------------------------
        int qPost = gpuProfiler.beginPass(cmd, frameSlot, "post_composite");
        renderPostComposite(cmd, width, height, sharedColorImage, frameSlot, stack);
        gpuProfiler.endPass(cmd, frameSlot, qPost);

        // --------------------------------------------------------------------
        // Pass 6: Convert Reversed-Z G-Buffer Depth -> Standard-Z Shared Depth
        // --------------------------------------------------------------------
        int qDepth = gpuProfiler.beginPass(cmd, frameSlot, "depth_convert");
        renderDepthConvert(cmd, width, height, sharedDepthImage, sharedDepthImageView, frameSlot, stack);
        gpuProfiler.endPass(cmd, frameSlot, qDepth);
    }

    private void dispatchChunkCull(
            VkCommandBuffer cmd,
            long pipeLayout,
            LongBuffer pSets,
            int slotDispatchLimit,
            boolean useHiZ,
            int passFlagFilter,
            MemoryStack stack
    ) {
        recordChunkCullCompute(cmd, pipeLayout, pSets, slotDispatchLimit, useHiZ, passFlagFilter, false, stack);
    }

    private void recordChunkCullCompute(
            VkCommandBuffer cmd,
            long pipeLayout,
            LongBuffer pSets,
            int slotDispatchLimit,
            boolean useHiZ,
            int passFlagFilter,
            boolean releaseToGraphics,
            MemoryStack stack
    ) {
        vkCmdFillBuffer(cmd, vkMegaBuffer.getDrawCountBuffer().buffer, 0, 4, 0);

        VkBufferMemoryBarrier.Buffer clearCountBarrier = VkBufferMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .buffer(vkMegaBuffer.getDrawCountBuffer().buffer)
                .offset(0).size(4);

        vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, null, clearCountBarrier, null);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipelineManager.getChunkCullPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, pSets, null);

        ByteBuffer pcCull = stack.malloc(12);
        pcCull.putInt(0, slotDispatchLimit);
        pcCull.putInt(4, useHiZ ? 1 : 0);
        pcCull.putInt(8, passFlagFilter);
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, pcCull);

        int groupCountX = (slotDispatchLimit + 63) / 64;
        vkCmdDispatch(cmd, groupCountX, 1, 1);

        if (releaseToGraphics && vkContext.hasSeparateComputeQueueFamily()) {
            VkBufferMemoryBarrier.Buffer releaseBarriers = VkBufferMemoryBarrier.calloc(3, stack);
            releaseBarriers.get(0).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                    .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                    .dstAccessMask(0)
                    .srcQueueFamilyIndex(vkContext.getComputeQueueFamilyIndex())
                    .dstQueueFamilyIndex(vkContext.getQueueFamilyIndex())
                    .buffer(vkMegaBuffer.getIndirectCmdBuffer().buffer)
                    .offset(0).size(VK_WHOLE_SIZE);
            releaseBarriers.get(1).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                    .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                    .dstAccessMask(0)
                    .srcQueueFamilyIndex(vkContext.getComputeQueueFamilyIndex())
                    .dstQueueFamilyIndex(vkContext.getQueueFamilyIndex())
                    .buffer(vkMegaBuffer.getDrawCountBuffer().buffer)
                    .offset(0).size(VK_WHOLE_SIZE);
            releaseBarriers.get(2).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                    .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                    .dstAccessMask(0)
                    .srcQueueFamilyIndex(vkContext.getComputeQueueFamilyIndex())
                    .dstQueueFamilyIndex(vkContext.getQueueFamilyIndex())
                    .buffer(vkMegaBuffer.getVisibleChunkIdBuffer().buffer)
                    .offset(0).size(VK_WHOLE_SIZE);
            vkCmdPipelineBarrier(cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    0, null, releaseBarriers, null);
        } else {
            VkBufferMemoryBarrier.Buffer indirectBarriers = VkBufferMemoryBarrier.calloc(2, stack);
            indirectBarriers.get(0).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                    .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                    .dstAccessMask(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .buffer(vkMegaBuffer.getIndirectCmdBuffer().buffer)
                    .offset(0).size(VK_WHOLE_SIZE);

            indirectBarriers.get(1).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                    .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                    .dstAccessMask(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .buffer(vkMegaBuffer.getDrawCountBuffer().buffer)
                    .offset(0).size(VK_WHOLE_SIZE);

            vkCmdPipelineBarrier(cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                    0, null, indirectBarriers, null);
        }
    }

    private void recordComputeCullAcquireBarrier(VkCommandBuffer graphicsCmd, MemoryStack stack) {
        if (!vkContext.hasSeparateComputeQueueFamily()) {
            return;
        }
        VkBufferMemoryBarrier.Buffer acquireBarriers = VkBufferMemoryBarrier.calloc(3, stack);
        acquireBarriers.get(0).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(0)
                .dstAccessMask(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
                .srcQueueFamilyIndex(vkContext.getComputeQueueFamilyIndex())
                .dstQueueFamilyIndex(vkContext.getQueueFamilyIndex())
                .buffer(vkMegaBuffer.getIndirectCmdBuffer().buffer)
                .offset(0).size(VK_WHOLE_SIZE);
        acquireBarriers.get(1).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(0)
                .dstAccessMask(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
                .srcQueueFamilyIndex(vkContext.getComputeQueueFamilyIndex())
                .dstQueueFamilyIndex(vkContext.getQueueFamilyIndex())
                .buffer(vkMegaBuffer.getDrawCountBuffer().buffer)
                .offset(0).size(VK_WHOLE_SIZE);
        acquireBarriers.get(2).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(0)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT)
                .srcQueueFamilyIndex(vkContext.getComputeQueueFamilyIndex())
                .dstQueueFamilyIndex(vkContext.getQueueFamilyIndex())
                .buffer(vkMegaBuffer.getVisibleChunkIdBuffer().buffer)
                .offset(0).size(VK_WHOLE_SIZE);
        vkCmdPipelineBarrier(graphicsCmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                0, null, acquireBarriers, null);
    }

    private void renderCsmCascade(VkCommandBuffer cmd, int cascade, int maxDrawCount, int frameSlot, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(frameSlot), vkPipelineManager.getDescriptorSet1());
        int csmRes = vkPipelineManager.getCsmResolution();

        transitionImageLayout(cmd, vkPipelineManager.getCsmArrayImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, cascade, 1,
                0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

        VkRenderingAttachmentInfo.Buffer depthAttach = VkRenderingAttachmentInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(vkPipelineManager.getCsmLayerView(cascade))
                .imageLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        depthAttach.clearValue().depthStencil().depth(1.0f);

        VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_INFO)
                .renderArea(ra -> ra.extent().set(csmRes, csmRes))
                .layerCount(1)
                .pDepthAttachment(depthAttach.get(0));

        vkCmdBeginRendering(cmd, renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineManager.getCsmShadowPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, pSets, null);

        // Push cascade index and squared max section distance for early out-of-range vertex rejection
        float maxRadius = switch (cascade) {
            case 0 -> 56.0f;   // 32m split + 24m section margin
            case 1 -> 120.0f;  // 96m split + 24m section margin
            default -> 216.0f; // 192m split + 24m section margin
        };
        ByteBuffer pcCsm = stack.malloc(8);
        pcCsm.putInt(0, cascade);
        pcCsm.putFloat(4, maxRadius * maxRadius);
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, pcCsm);

        VkViewport.Buffer vp = VkViewport.calloc(1, stack)
                .x(0).y(0).width(csmRes).height(csmRes).minDepth(0f).maxDepth(1f);
        vkCmdSetViewport(cmd, 0, vp);

        VkRect2D.Buffer sc = VkRect2D.calloc(1, stack);
        sc.extent().set(csmRes, csmRes);
        vkCmdSetScissor(cmd, 0, sc);

        vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 1.75f);

        vkCmdBindIndexBuffer(cmd, vkMegaBuffer.getGlobalQuadIndexBuffer().buffer, 0, VK_INDEX_TYPE_UINT32);

        // Core Vulkan 1.2 draw indexed indirect count
        vkCmdDrawIndexedIndirectCount(
                cmd,
                vkMegaBuffer.getIndirectCmdBuffer().buffer,
                0,
                vkMegaBuffer.getDrawCountBuffer().buffer,
                0,
                maxDrawCount,
                20
        );

        vkCmdEndRendering(cmd);

        transitionImageLayout(cmd, vkPipelineManager.getCsmArrayImage(),
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, cascade, 1,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    private void renderGBuffer(VkCommandBuffer cmd, int width, int height, int maxDrawCount, int frameSlot, boolean translucent, boolean transitionToShaderRead, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(frameSlot), vkPipelineManager.getDescriptorSet1());

        if (!translucent) {
            // Prepare RT0, RT1, and internal Reversed-Z gbufferDepthImage
            transitionImageLayout(cmd, vkPipelineManager.getRt0Image(),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

            transitionImageLayout(cmd, vkPipelineManager.getRt1Image(),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

            transitionImageLayout(cmd, vkPipelineManager.getGbufferDepthImage(),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                    0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
        }

        int colorLoadOp = translucent ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        int depthLoadOp = translucent ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;

        VkRenderingAttachmentInfo.Buffer colorAttaches = VkRenderingAttachmentInfo.calloc(2, stack);
        colorAttaches.get(0).sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(vkPipelineManager.getRt0View())
                .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                .loadOp(colorLoadOp)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        colorAttaches.get(0).clearValue().color().float32(0, 0f).float32(1, 0f).float32(2, 0f).float32(3, 0f);

        colorAttaches.get(1).sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(vkPipelineManager.getRt1View())
                .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                .loadOp(colorLoadOp)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        colorAttaches.get(1).clearValue().color().uint32(0, 0).uint32(1, 0).uint32(2, 0).uint32(3, 0);

        VkRenderingAttachmentInfo.Buffer depthAttach = VkRenderingAttachmentInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(vkPipelineManager.getGbufferDepthView())
                .imageLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                .loadOp(depthLoadOp)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        depthAttach.clearValue().depthStencil().depth(0.0f); // Reverse-Z: 0.0 is Far, 1.0 is Near

        VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_INFO)
                .renderArea(ra -> ra.extent().set(width, height))
                .layerCount(1)
                .pColorAttachments(colorAttaches)
                .pDepthAttachment(depthAttach.get(0));

        vkCmdBeginRendering(cmd, renderingInfo);

        long pipeline = translucent
                ? vkPipelineManager.getTranslucentGbufferPipeline()
                : vkPipelineManager.getGbufferPipeline();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, pSets, null);

        VkViewport.Buffer vp = VkViewport.calloc(1, stack).x(0).y(0).width(width).height(height).minDepth(0f).maxDepth(1f);
        vkCmdSetViewport(cmd, 0, vp);

        VkRect2D.Buffer sc = VkRect2D.calloc(1, stack);
        sc.extent().set(width, height);
        vkCmdSetScissor(cmd, 0, sc);

        vkCmdBindIndexBuffer(cmd, vkMegaBuffer.getGlobalQuadIndexBuffer().buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexedIndirectCount(
                cmd,
                vkMegaBuffer.getIndirectCmdBuffer().buffer,
                0,
                vkMegaBuffer.getDrawCountBuffer().buffer,
                0,
                maxDrawCount,
                20
        );

        vkCmdEndRendering(cmd);

        if (transitionToShaderRead) {
            // Transition G-Buffer outputs to SHADER_READ_ONLY_OPTIMAL for lighting & Hi-Z compute passes
            transitionImageLayout(cmd, vkPipelineManager.getRt0Image(),
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            transitionImageLayout(cmd, vkPipelineManager.getRt1Image(),
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            transitionImageLayout(cmd, vkPipelineManager.getGbufferDepthImage(),
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }
    }

    private void renderHiZPyramid(VkCommandBuffer cmd, int width, int height, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());
        int mipLevels = vkPipelineManager.getHiZMipLevels();
        long hiZImage = vkPipelineManager.getHiZImage();

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipelineManager.getHizDownsamplePipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, pSets, null);

        int srcW = width;
        int srcH = height;
        int dstW = width;
        int dstH = height;
        ByteBuffer pcHiZ = stack.malloc(16);

        for (int mip = 0; mip < mipLevels; mip++) {
            pcHiZ.putInt(0, mip);
            pcHiZ.putInt(4, srcW);
            pcHiZ.putInt(8, srcH);
            pcHiZ.putInt(12, dstW);
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, pcHiZ);

            vkCmdDispatch(cmd, (dstW + 7) / 8, (dstH + 7) / 8, 1);

            // Ensure this written mip in GENERAL layout is visible before next mip reads it
            transitionImageLayout(cmd, hiZImage,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            srcW = dstW;
            srcH = dstH;
            dstW = Math.max(1, dstW / 2);
            dstH = Math.max(1, dstH / 2);
        }
    }

    private void renderDeferredLighting(VkCommandBuffer cmd, int width, int height, int frameSlot, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(frameSlot), vkPipelineManager.getDescriptorSet1());

        transitionImageLayout(cmd, vkPipelineManager.getHdrColorImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipelineManager.getDeferredLightingPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, pSets, null);

        int groupsX = (width + 7) / 8;
        int groupsY = (height + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Barrier: HDR storage image write -> read in post composite
        VkMemoryBarrier.Buffer memBarrier = VkMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT);

        vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, memBarrier, null, null);
    }

    private void renderPostComposite(VkCommandBuffer cmd, int width, int height, long sharedColorImage, int frameSlot, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(frameSlot), vkPipelineManager.getDescriptorSet1());

        transitionImageLayout(cmd, sharedColorImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipelineManager.getPostCompositePipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, pSets, null);

        ValkyrieConfig config = ValkyrieConfig.get();
        ByteBuffer pcPost = stack.malloc(8);
        pcPost.putFloat(0, config.getExposure());
        pcPost.putInt(4, config.isFxaaEnabled() ? 1 : 0);
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, pcPost);

        int groupsX = (width + 7) / 8;
        int groupsY = (height + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Transition Shared Color to TRANSFER_SRC_OPTIMAL for OpenGL zero-copy Blit
        transitionImageLayout(cmd, sharedColorImage,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    private void renderDepthConvert(VkCommandBuffer cmd, int width, int height, long sharedDepthImage, long sharedDepthImageView, int frameSlot, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(frameSlot), vkPipelineManager.getDescriptorSet1());

        transitionImageLayout(cmd, sharedDepthImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

        VkRenderingAttachmentInfo.Buffer depthAttach = VkRenderingAttachmentInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(sharedDepthImageView)
                .imageLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                .loadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);

        VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_INFO)
                .renderArea(ra -> ra.extent().set(width, height))
                .layerCount(1)
                .pDepthAttachment(depthAttach.get(0));

        vkCmdBeginRendering(cmd, renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineManager.getDepthConvertPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, pSets, null);

        VkViewport.Buffer vp = VkViewport.calloc(1, stack).x(0).y(0).width(width).height(height).minDepth(0f).maxDepth(1f);
        vkCmdSetViewport(cmd, 0, vp);

        VkRect2D.Buffer sc = VkRect2D.calloc(1, stack);
        sc.extent().set(width, height);
        vkCmdSetScissor(cmd, 0, sc);

        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);

        // Transition Shared Depth to TRANSFER_SRC_OPTIMAL for OpenGL zero-copy Blit
        transitionImageLayout(cmd, sharedDepthImage,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    private void recordPhase2FallbackPass(VkCommandBuffer cmd, int width, int height, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

        transitionImageLayout(cmd, vkInteropManager.getSharedColorImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipelineManager.getTestGradientPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, pSets, null);

        ByteBuffer pc = stack.malloc(4);
        pc.putFloat(0, (float) (frameIndex * 0.016));
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);

        int groupsX = (width + 15) / 16;
        int groupsY = (height + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        transitionImageLayout(cmd, vkInteropManager.getSharedColorImage(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        // Clear shared depth buffer to 1.0f (Standard-Z far plane) before OpenGL blit
        transitionImageLayout(cmd, vkInteropManager.getSharedDepthImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkClearDepthStencilValue clearDepth = VkClearDepthStencilValue.calloc(stack).depth(1.0f).stencil(0);
        VkImageSubresourceRange.Buffer depthRange = VkImageSubresourceRange.calloc(1, stack)
                .aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
        vkCmdClearDepthStencilImage(cmd, vkInteropManager.getSharedDepthImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clearDepth, depthRange);

        transitionImageLayout(cmd, vkInteropManager.getSharedDepthImage(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    private void computeCsmMatrices(Vector3f cameraPos, Vector3f sunDir, Matrix4f viewMatrix) {
        float[] splits = { 32.0f, 96.0f, 192.0f };
        int csmRes = Math.max(1, vkPipelineManager != null ? vkPipelineManager.getCsmResolution() : 1024);
        boolean nearPole = Math.abs(sunDir.y) > 0.99f;
        float upX = 0f;
        float upY = nearPole ? 0f : 1f;
        float upZ = nearPole ? 1f : 0f;
        for (int i = 0; i < VkPipelineManager.CSM_CASCADE_COUNT; i++) {
            float dist = splits[i];
            Vector3f lightPos = new Vector3f(cameraPos).add(new Vector3f(sunDir).mul(dist * 0.5f));
            Matrix4f lightView = new Matrix4f().lookAt(
                    lightPos.x, lightPos.y, lightPos.z,
                    cameraPos.x, cameraPos.y, cameraPos.z,
                    upX, upY, upZ
            );
            // Snap light-space origin to shadow texel grid to prevent shadow edge shimmering
            float texelWorldSize = (2.0f * dist) / (float) csmRes;
            lightView.m30((float) Math.floor(lightView.m30() / texelWorldSize) * texelWorldSize);
            lightView.m31((float) Math.floor(lightView.m31() / texelWorldSize) * texelWorldSize);
            Matrix4f lightProj = new Matrix4f().ortho(-dist, dist, -dist, dist, -dist * 2f, dist * 2f, true);
            csmMatrices[i].set(lightProj).mul(lightView);
        }
    }

    /**
     * Collects the cascades to render this frame and synchronizes renderedCsmMatrices
     * so deferred lighting never samples a stale cascade with a newer frame's matrix.
     */
    private List<Integer> collectCsmCascadesToUpdate(boolean isBuiltinPipeline) {
        ValkyrieConfig config = ValkyrieConfig.get();
        if (!isBuiltinPipeline || !config.isCsmEnabled() || config.isVanillaLighting()) {
            java.util.Arrays.fill(csmCascadeInitialized, false);
            return List.of();
        }
        int activeCascades = Math.min(config.getCsmCascades(), VkPipelineManager.CSM_CASCADE_COUNT);
        if (config.isShadowSkipStaticFrames() && !shadowStaticDirtyThisFrame && !dirtyShadow) {
            boolean allInitialized = true;
            for (int c = 0; c < activeCascades; c++) {
                if (!csmCascadeInitialized[c]) {
                    allInitialized = false;
                    break;
                }
            }
            if (allInitialized) {
                return List.of();
            }
        }
        List<Integer> cascades = new ArrayList<>(activeCascades);
        for (int c = 0; c < activeCascades; c++) {
            if (!csmCascadeInitialized[c] || vkPipelineManager.shouldUpdateCascade(c, frameIndex, dirtyShadow)) {
                cascades.add(c);
                renderedCsmMatrices[c].set(csmMatrices[c]);
                csmCascadeInitialized[c] = true;
            }
        }
        return cascades;
    }

    /**
     * Evaluates whether camera, sun/time, scene geometry, entities, world/dimension, or shadow settings
     * changed since the previous frame. Conservative: returns true whenever any tracked state differs.
     */
    private boolean evaluateStaticShadowState(
            ClientLevel level,
            Vector3f cameraPos,
            Matrix4f viewMatrix,
            Vector3f sunDir,
            int slot,
            boolean uploadedMeshesThisFrame) {
        ValkyrieConfig cfg = ValkyrieConfig.get();
        boolean packActive = isShaderPackActive();

        int levelId = (level != null) ? System.identityHashCode(level) : 0;
        String dimId = (level != null && level.dimension() != null)
                ? level.dimension().location().toString()
                : "";
        long dayTime = (level != null) ? level.getDayTime() : 0L;
        float rainLevel = (level != null) ? level.getRainLevel(1.0f) : 0.0f;

        // Entity movement digest within shadow range (conservative 256m radius)
        int entityCount = 0;
        long entitySum = 0L;
        long entityXor = 0L;
        if (level != null) {
            final float shadowRange = 256.0f;
            final double shadowRangeSq = (double) shadowRange * (double) shadowRange;
            for (net.minecraft.world.entity.Entity e : level.entitiesForRendering()) {
                if (e.distanceToSqr(cameraPos.x, cameraPos.y, cameraPos.z) <= shadowRangeSq) {
                    long h = Double.doubleToLongBits(e.getX());
                    h = h * 31L + Double.doubleToLongBits(e.getY());
                    h = h * 31L + Double.doubleToLongBits(e.getZ());
                    h = h * 31L + Float.floatToIntBits(e.getYRot());
                    h = h * 31L + Float.floatToIntBits(e.getXRot());
                    h = h * 31L + e.getId();
                    entitySum += h;
                    entityXor ^= Long.rotateLeft(h, 17);
                    entityCount++;
                }
            }
        }

        boolean worldChanged = !shadowStaticStateValid
                || levelId != prevShadowLevelId
                || !dimId.equals(prevShadowDimensionId);

        boolean settingsChanged = !shadowStaticStateValid
                || cfg.isCsmEnabled() != prevCfgCsmEnabled
                || cfg.getCsmCascades() != prevCfgCsmCascades
                || cfg.getCsmResolution() != prevCfgCsmResolution
                || cfg.getCsmUpdateInterval() != prevCfgCsmUpdateInterval
                || cfg.isVanillaLighting() != prevCfgVanillaLighting
                || cfg.isShadowSkipStaticFrames() != prevCfgShadowSkipStaticFrames
                || cfg.getShaderPackShadowResolutionCap() != prevCfgShadowResCap
                || packActive != prevCfgShaderPackActive;

        if (worldChanged || settingsChanged) {
            java.util.Arrays.fill(csmCascadeInitialized, false);
        }

        boolean cameraChanged = !shadowStaticStateValid
                || Math.abs(cameraPos.x - prevShadowCameraPos.x) > SHADOW_STATIC_EPSILON
                || Math.abs(cameraPos.y - prevShadowCameraPos.y) > SHADOW_STATIC_EPSILON
                || Math.abs(cameraPos.z - prevShadowCameraPos.z) > SHADOW_STATIC_EPSILON
                || !matrixWithinEpsilon(viewMatrix, prevShadowViewMatrix, SHADOW_STATIC_EPSILON);

        boolean sunOrTimeChanged = !shadowStaticStateValid
                || dayTime != prevShadowDayTime
                || Math.abs(rainLevel - prevShadowRainLevel) > SHADOW_STATIC_EPSILON
                || Math.abs(sunDir.x - prevShadowSunDir.x) > SHADOW_STATIC_EPSILON
                || Math.abs(sunDir.y - prevShadowSunDir.y) > SHADOW_STATIC_EPSILON
                || Math.abs(sunDir.z - prevShadowSunDir.z) > SHADOW_STATIC_EPSILON;

        boolean sceneChanged = dirtyShadow
                || uploadedMeshesThisFrame
                || vkMegaBuffer.hasPendingTransfers(slot)
                || vkMegaBuffer.getDeferredUploadCount() > 0
                || entityCount != prevShadowEntityCount
                || entitySum != prevShadowEntitySum
                || entityXor != prevShadowEntityXor;

        prevShadowCameraPos.set(cameraPos);
        prevShadowViewMatrix.set(viewMatrix);
        prevShadowSunDir.set(sunDir);
        prevShadowDayTime = dayTime;
        prevShadowRainLevel = rainLevel;
        prevShadowLevelId = levelId;
        prevShadowDimensionId = dimId;
        prevShadowEntityCount = entityCount;
        prevShadowEntitySum = entitySum;
        prevShadowEntityXor = entityXor;
        prevCfgCsmEnabled = cfg.isCsmEnabled();
        prevCfgCsmCascades = cfg.getCsmCascades();
        prevCfgCsmResolution = cfg.getCsmResolution();
        prevCfgCsmUpdateInterval = cfg.getCsmUpdateInterval();
        prevCfgVanillaLighting = cfg.isVanillaLighting();
        prevCfgShadowSkipStaticFrames = cfg.isShadowSkipStaticFrames();
        prevCfgShadowResCap = cfg.getShaderPackShadowResolutionCap();
        prevCfgShaderPackActive = packActive;
        shadowStaticStateValid = true;

        return worldChanged || settingsChanged || cameraChanged || sunOrTimeChanged || sceneChanged;
    }

    private static boolean matrixWithinEpsilon(Matrix4f a, Matrix4f b, float eps) {
        return Math.abs(a.m00() - b.m00()) <= eps && Math.abs(a.m01() - b.m01()) <= eps
                && Math.abs(a.m02() - b.m02()) <= eps && Math.abs(a.m03() - b.m03()) <= eps
                && Math.abs(a.m10() - b.m10()) <= eps && Math.abs(a.m11() - b.m11()) <= eps
                && Math.abs(a.m12() - b.m12()) <= eps && Math.abs(a.m13() - b.m13()) <= eps
                && Math.abs(a.m20() - b.m20()) <= eps && Math.abs(a.m21() - b.m21()) <= eps
                && Math.abs(a.m22() - b.m22()) <= eps && Math.abs(a.m23() - b.m23()) <= eps
                && Math.abs(a.m30() - b.m30()) <= eps && Math.abs(a.m31() - b.m31()) <= eps
                && Math.abs(a.m32() - b.m32()) <= eps && Math.abs(a.m33() - b.m33()) <= eps;
    }

    private void transitionImageLayout(
            VkCommandBuffer cmd,
            long image,
            int oldLayout,
            int newLayout,
            int aspectMask,
            int baseMip,
            int mipCount,
            int baseLayer,
            int layerCount,
            int srcAccess,
            int dstAccess,
            int srcStage,
            int dstStage
    ) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                    .oldLayout(oldLayout)
                    .newLayout(newLayout)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .srcAccessMask(srcAccess)
                    .dstAccessMask(dstAccess);

            barrier.subresourceRange()
                    .aspectMask(aspectMask)
                    .baseMipLevel(baseMip)
                    .levelCount(mipCount)
                    .baseArrayLayer(baseLayer)
                    .layerCount(layerCount);

            vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, null, null, barrier);
        }
    }

    /**
     * Composites the rendered Vulkan frame buffer into Minecraft's main OpenGL RenderTarget.
     */
    public void compositeToOpenGL(RenderTarget mainTarget) {
        if (unavailable || !initialized) {
            return;
        }
        if (vkInteropManager != null) {
            if (!vkInteropManager.usesExternalSemaphores()
                    && vkContext != null && vkContext.isInitialized()) {
                int slot = lastSubmittedSlot;
                if (frameFences[slot] != VK_NULL_HANDLE && !fenceWaited[slot]) {
                    vkWaitForFences(vkContext.getVkDevice(), frameFences[slot], true, 1_000_000_000L);
                    fenceWaited[slot] = true;
                }
            }
            vkInteropManager.compositeToMainTarget(mainTarget);
        }
    }

    public boolean isInitialized() {
        return initialized && !unavailable;
    }

    public boolean isUnavailable() {
        return unavailable;
    }

    public boolean isChunkQueueIdle() {
        if (!initialized || unavailable || greedyMesher == null || vkMegaBuffer == null) {
            return true;
        }
        return dirtySections.isEmpty()
                && greedyMesher.getInFlightCount() == 0
                && greedyMesher.getCompletedCount() == 0
                && vkMegaBuffer.getDeferredUploadCount() == 0;
    }

    public boolean isShaderPackActive() {
        return vkShaderPackPipeline != null
                && vkShaderPackPipeline.isInitialized()
                && vkShaderPackPipeline.isPackReady()
                && !CompatManager.shouldYieldToExternalShaderPipeline()
                && (explicitPackLoaded || CompatManager.isNativeShaderPackAllowed());
    }

    public VkShaderPackPipeline getShaderPackPipeline() {
        return vkShaderPackPipeline;
    }

    private void logActivePipelineIfChanged(String mode) {
        if (!mode.equals(lastLoggedRenderPipeline)) {
            lastLoggedRenderPipeline = mode;
            LOGGER.info("[Valkyrie] Active render pipeline -> {}", mode);
        }
    }

    /**
     * Hot-reloads the configured shaderpack from {@link ValkyrieConfig} against Minecraft's game directory.
     * If the renderer is not yet initialized, queues the reload for initialization.
     */
    public synchronized boolean reloadShaderPack() {
        if (unavailable) {
            return false;
        }
        explicitPackLoaded = false;
        if (!initialized && (vkContext == null || !vkContext.isInitialized())) {
            pendingShaderPackReload = true;
            return false;
        }
        pendingShaderPackReload = false;
        return reloadShaderPackLocked();
    }

    /**
     * Hot-reloads the configured shaderpack from an explicit game directory (useful for headless tests).
     */
    public synchronized boolean reloadShaderPack(Path gameDir) {
        if (unavailable) {
            return false;
        }
        explicitPackLoaded = false;
        ValkyrieConfig config = ValkyrieConfig.get();
        if (!CompatManager.isNativeShaderPackAllowed()) {
            unloadShaderPack();
            return false;
        }
        ensureShaderPackPipelineInitialized();
        if (vkShaderPackPipeline == null || gameDir == null) {
            return false;
        }
        int w = currentWidth > 0 ? currentWidth : 1280;
        int h = currentHeight > 0 ? currentHeight : 720;
        try {
            syncShaderPackAtlas();
            boolean ok = vkShaderPackPipeline.loadShaderPack(gameDir, config.getShaderPack(), w, h);
            if (!ok) {
                LOGGER.warn("[Valkyrie] Shaderpack '{}' failed to activate; falling back to built-in deferred PBR pipeline.",
                        config.getShaderPack());
                vkShaderPackPipeline.unloadPack();
            } else {
                LOGGER.info("[Valkyrie] Loaded configured shaderpack '{}' (active={})",
                        vkShaderPackPipeline.getActivePackName(), isShaderPackActive());
            }
            return ok;
        } catch (Throwable t) {
            LOGGER.warn("[Valkyrie] Exception loading shaderpack '{}'; falling back to built-in deferred PBR pipeline.",
                    config.getShaderPack(), t);
            vkShaderPackPipeline.unloadPack();
            return false;
        }
    }

    /**
     * Loads and compiles a shaderpack directly from a {@code .zip} or folder {@link Path}.
     * Sets an explicit active override so direct API / smoke-sweep loads render even when
     * {@code config/valkyrie.properties} has {@code shaderPack=off}.
     */
    public synchronized boolean loadShaderPack(Path packPath) {
        if (unavailable) {
            return false;
        }
        try {
            ensureShaderPackPipelineInitialized();
        } catch (Throwable t) {
            markUnavailableLocked(t);
            return false;
        }
        if (vkShaderPackPipeline == null || packPath == null) {
            explicitPackLoaded = false;
            return false;
        }
        int w = currentWidth > 0 ? currentWidth : 1280;
        int h = currentHeight > 0 ? currentHeight : 720;
        try {
            syncShaderPackAtlas();
            boolean ok = vkShaderPackPipeline.loadShaderPack(packPath, w, h);
            explicitPackLoaded = ok;
            if (!ok) {
                LOGGER.warn("[Valkyrie] Shaderpack '{}' failed to compile; falling back to built-in deferred PBR pipeline.",
                        packPath);
                vkShaderPackPipeline.unloadPack();
            } else {
                LOGGER.info("[Valkyrie] Loaded explicit shaderpack '{}' (active={})",
                        vkShaderPackPipeline.getActivePackName(), isShaderPackActive());
            }
            return ok;
        } catch (Throwable t) {
            LOGGER.warn("[Valkyrie] Exception compiling shaderpack '{}'; falling back to built-in deferred PBR pipeline.",
                    packPath, t);
            explicitPackLoaded = false;
            vkShaderPackPipeline.unloadPack();
            return false;
        }
    }

    public synchronized void unloadShaderPack() {
        pendingShaderPackReload = false;
        explicitPackLoaded = false;
        if (vkShaderPackPipeline != null) {
            vkShaderPackPipeline.unloadPack();
        }
        LOGGER.info("[Valkyrie] Unloaded native shaderpack; built-in deferred PBR pipeline active.");
    }

    private boolean reloadShaderPackLocked() {
        if (!CompatManager.isNativeShaderPackAllowed()) {
            if (vkShaderPackPipeline != null) {
                vkShaderPackPipeline.unloadPack();
            }
            return false;
        }
        Path gameDir = resolveGameDirectory();
        if (gameDir == null) {
            LOGGER.warn("[Valkyrie] Cannot resolve game directory to load shaderpack '{}'; using built-in deferred PBR.",
                    ValkyrieConfig.get().getShaderPack());
            return false;
        }
        return reloadShaderPack(gameDir);
    }

    private void ensureShaderPackPipelineInitialized() {
        if (vkShaderPackPipeline != null && vkShaderPackPipeline.isInitialized()) {
            return;
        }
        if (this.vkContext == null) {
            this.vkContext = VkContext.getInstance();
        }
        if (!this.vkContext.isInitialized()) {
            this.vkContext.init();
        }
        if (this.vkMegaBuffer == null) {
            this.vkMegaBuffer = VkMegaBuffer.getInstance();
        }
        if (!this.vkMegaBuffer.isInitialized()) {
            this.vkMegaBuffer.init();
        }
        if (this.vkPipelineManager == null) {
            this.vkPipelineManager = VkPipelineManager.getInstance();
        }
        int w = currentWidth > 0 ? currentWidth : 1280;
        int h = currentHeight > 0 ? currentHeight : 720;
        if (this.vkShaderPackPipeline == null) {
            this.vkShaderPackPipeline = new VkShaderPackPipeline(vkContext, vkMegaBuffer);
        }
        if (!this.vkShaderPackPipeline.isInitialized()) {
            this.vkShaderPackPipeline.init(w, h, VK_FORMAT_R8G8B8A8_UNORM);
        }
        syncShaderPackAtlas();
    }

    private ShaderUniformLayout.UniformData buildShaderPackUniformData(
            ClientLevel level,
            Camera camera,
            Vector3f cameraPos,
            Matrix4f frustumMatrix,
            Matrix4f projectionMatrix,
            float partialTicks
    ) {
        ShaderUniformLayout.UniformData u = new ShaderUniformLayout.UniformData();
        u.gbufferModelView.set(frustumMatrix).m30(0.0f).m31(0.0f).m32(0.0f).m33(1.0f);
        u.gbufferProjection.set(projectionMatrix);
        u.cameraPosition.set(cameraPos);

        long nowNanos = System.nanoTime();
        float frameTime = 1.0f / 60.0f;
        if (lastShaderPackFrameNanos != 0L) {
            float dt = (float) ((nowNanos - lastShaderPackFrameNanos) * 1e-9);
            if (Float.isFinite(dt) && dt > 0.0f) {
                frameTime = Math.max(1e-4f, Math.min(0.25f, dt));
            }
        }
        lastShaderPackFrameNanos = nowNanos;
        shaderPackFrameTimeCounter += frameTime;
        u.frameCounter = (int) frameIndex;
        u.frameTime = frameTime;
        u.frameTimeCounter = shaderPackFrameTimeCounter;

        float far = 256.0f;
        Minecraft mc = null;
        try {
            mc = Minecraft.getInstance();
            if (mc != null && mc.gameRenderer != null) {
                far = Math.max(64.0f, mc.gameRenderer.getDepthFar());
            }
        } catch (Throwable ignored) {
        }
        u.near = 0.05f;
        u.far = far;

        if (level != null) {
            float skyAngle = level.getTimeOfDay(partialTicks);
            float optifineSunAngle = (skyAngle < 0.75f) ? (skyAngle + 0.25f) : (skyAngle - 0.75f);
            u.sunAngle = optifineSunAngle;
            u.shadowAngle = (optifineSunAngle < 0.5f) ? optifineSunAngle : (optifineSunAngle - 0.5f);

            long rawDayTime = level.getDayTime();
            int wrappedWorldTime = (int) (rawDayTime % 24000L);
            u.worldTime = wrappedWorldTime < 0 ? wrappedWorldTime + 24000 : wrappedWorldTime;
            u.worldDay = (int) Math.max(0L, rawDayTime / 24000L);
            u.moonPhase = level.getMoonPhase();

            float rain = level.getRainLevel(partialTicks);
            u.rainStrength = rain;
            if (!hasSmoothedWetness) {
                smoothedWetness = rain;
                hasSmoothedWetness = true;
            } else if (rain > smoothedWetness) {
                smoothedWetness = Math.min(1.0f, smoothedWetness + frameTime * 0.1f);
            } else {
                smoothedWetness = Math.max(rain, smoothedWetness - frameTime * 0.01f);
            }
            u.wetness = Math.max(rain, smoothedWetness);

            Vec3 skyPos = (camera != null)
                    ? camera.getPosition()
                    : new Vec3(cameraPos.x, cameraPos.y, cameraPos.z);
            Vec3 sky = level.getSkyColor(skyPos, partialTicks);
            u.skyColor.set((float) sky.x, (float) sky.y, (float) sky.z);

            try {
                float[] shaderFogColor = RenderSystem.getShaderFogColor();
                if (shaderFogColor != null && shaderFogColor.length >= 4 && shaderFogColor[3] > 0.001f) {
                    u.fogColor.set(shaderFogColor[0], shaderFogColor[1], shaderFogColor[2]);
                } else {
                    u.fogColor.set(u.skyColor);
                }
                float fogStart = RenderSystem.getShaderFogStart();
                float fogEnd = RenderSystem.getShaderFogEnd();
                u.fogStart = (Float.isFinite(fogStart) && fogStart > -1.0e5f && fogStart < 1.0e6f)
                        ? fogStart
                        : Math.max(16.0f, far * 0.5f);
                u.fogEnd = (Float.isFinite(fogEnd) && fogEnd > u.fogStart && fogEnd < 1.0e6f)
                        ? fogEnd
                        : far;
            } catch (Throwable ignored) {
                u.fogColor.set(u.skyColor);
                u.fogStart = Math.max(16.0f, far * 0.5f);
                u.fogEnd = far;
            }
            u.fogDensity = (u.fogEnd > u.fogStart) ? Math.min(1.0f, 1.0f / (u.fogEnd - u.fogStart)) : 0.0f;

            if (camera != null) {
                FogType fluid = camera.getFluidInCamera();
                if (fluid == FogType.WATER) {
                    u.isEyeInWater = 1;
                } else if (fluid == FogType.LAVA) {
                    u.isEyeInWater = 2;
                } else if (fluid == FogType.POWDER_SNOW) {
                    u.isEyeInWater = 3;
                } else {
                    u.isEyeInWater = 0;
                }
            }

            BlockPos eyePos = (camera != null)
                    ? camera.getBlockPosition()
                    : BlockPos.containing(cameraPos.x, cameraPos.y, cameraPos.z);
            int eyeX = Math.max(0, Math.min(240, level.getBrightness(LightLayer.BLOCK, eyePos) * 16));
            int eyeY = Math.max(0, Math.min(240, level.getBrightness(LightLayer.SKY, eyePos) * 16));
            u.eyeBrightness.set(eyeX, eyeY);
            if (!hasSmoothedEyeBrightness) {
                smoothEyeBrightnessX = eyeX;
                smoothEyeBrightnessY = eyeY;
                hasSmoothedEyeBrightness = true;
            } else {
                float alpha = Math.min(1.0f, frameTime * 10.0f);
                smoothEyeBrightnessX += (eyeX - smoothEyeBrightnessX) * alpha;
                smoothEyeBrightnessY += (eyeY - smoothEyeBrightnessY) * alpha;
            }
            u.eyeBrightnessSmooth.set(Math.round(smoothEyeBrightnessX), Math.round(smoothEyeBrightnessY));
        } else {
            // Sunlit midday defaults when no ClientLevel is present
            u.sunAngle = 0.25f;
            u.shadowAngle = 0.25f;
            u.worldTime = 6000;
            u.worldDay = 0;
            u.moonPhase = 0;
            u.rainStrength = 0.0f;
            u.wetness = 0.0f;
            u.skyColor.set(0.47f, 0.67f, 1.0f);
            u.fogColor.set(0.75f, 0.85f, 1.0f);
            u.fogStart = Math.max(16.0f, far * 0.5f);
            u.fogEnd = far;
            u.fogDensity = (u.fogEnd > u.fogStart) ? Math.min(1.0f, 1.0f / (u.fogEnd - u.fogStart)) : 0.0f;
            u.eyeBrightness.set(240, 240);
            u.eyeBrightnessSmooth.set(240, 240);
        }

        if (mc != null) {
            try {
                if (mc.player != null) {
                    if (mc.player.hasEffect(MobEffects.NIGHT_VISION) && mc.gameRenderer != null) {
                        u.nightVision = GameRenderer.getNightVisionScale(mc.player, partialTicks);
                    }
                    if (mc.player.hasEffect(MobEffects.BLINDNESS)) {
                        u.blindness = 1.0f;
                    }
                    if (mc.player.hasEffect(MobEffects.DARKNESS) && mc.gameRenderer != null) {
                        u.darknessFactor = mc.gameRenderer.getDarkenWorldAmount(partialTicks);
                    }
                }
                if (mc.options != null) {
                    u.screenBrightness = mc.options.gamma().get().floatValue();
                }
            } catch (Throwable ignored) {
            }
        }

        float sunPathRotationDeg = 0.0f;
        if (vkShaderPackPipeline != null && vkShaderPackPipeline.getActiveProperties() != null) {
            sunPathRotationDeg = vkShaderPackPipeline.getActiveProperties().getSunPathRotation();
        }
        double sunRad = u.sunAngle * (Math.PI * 2.0);
        double sunPathRad = Math.toRadians(sunPathRotationDeg);
        Vector3f sunWorldDir = new Vector3f(
                (float) Math.cos(sunRad),
                (float) (Math.cos(sunPathRad) * Math.sin(sunRad)),
                (float) (-Math.sin(sunPathRad) * Math.sin(sunRad))
        );
        populateCelestialVectors(u, u.gbufferModelView, sunWorldDir);
        return u;
    }

    private void populateCelestialVectors(
            ShaderUniformLayout.UniformData u,
            Matrix4f rotView,
            Vector3f sunWorldDir
    ) {
        Vector3f normSun = new Vector3f(sunWorldDir);
        if (normSun.lengthSquared() > 1e-6f) {
            normSun.normalize();
        } else {
            normSun.set(0.0f, 1.0f, 0.0f);
        }

        Vector3f sunViewDir = rotView.transformDirection(normSun, new Vector3f());
        if (sunViewDir.lengthSquared() > 1e-6f) {
            sunViewDir.normalize();
        } else {
            sunViewDir.set(0.0f, 1.0f, 0.0f);
        }
        u.sunPosition.set(sunViewDir).mul(100.0f);
        u.moonPosition.set(sunViewDir).negate().mul(100.0f);

        Vector3f shadowLightDir = normSun.y >= 0.0f ? sunViewDir : new Vector3f(sunViewDir).negate();
        u.shadowLightPosition.set(shadowLightDir).mul(100.0f);

        Vector3f upViewDir = rotView.transformDirection(new Vector3f(0.0f, 1.0f, 0.0f), new Vector3f());
        if (upViewDir.lengthSquared() > 1e-6f) {
            upViewDir.normalize();
        } else {
            upViewDir.set(0.0f, 1.0f, 0.0f);
        }
        u.upPosition.set(upViewDir).mul(100.0f);
    }

    private static Path resolveGameDirectory() {
        try {
            Minecraft mc = Minecraft.getInstance();
            if (mc != null && mc.gameDirectory != null) {
                return mc.gameDirectory.toPath();
            }
        } catch (Throwable ignored) {
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private final List<Long>[] dirtyDistanceBuckets = new List[DISTANCE_BUCKETS];

    private int drainDirtySections(ClientLevel level, Vector3f cameraPos) {
        if (level == null || greedyMesher == null || dirtySections.isEmpty()) {
            return 0;
        }
        int inFlight = greedyMesher.getInFlightCount();
        if (inFlight >= MAX_IN_FLIGHT_MESHES) {
            return 0;
        }
        if (atlasUVTable != null && atlasUVTable.getSpriteCount() <= 1) {
            atlasUVTable.rebuildFromBlockAtlas();
        }

        int camSX = SectionPos.blockToSectionCoord((int) Math.floor(cameraPos.x));
        int camSY = SectionPos.blockToSectionCoord((int) Math.floor(cameraPos.y));
        int camSZ = SectionPos.blockToSectionCoord((int) Math.floor(cameraPos.z));

        for (int b = 0; b < DISTANCE_BUCKETS; b++) {
            if (dirtyDistanceBuckets[b] == null) {
                dirtyDistanceBuckets[b] = new ArrayList<>();
            } else {
                dirtyDistanceBuckets[b].clear();
            }
        }
        for (Long key : dirtySections) {
            int sx = SectionPos.x(key);
            int sy = SectionPos.y(key);
            int sz = SectionPos.z(key);
            int dx = Math.abs(sx - camSX);
            int dy = Math.abs(sy - camSY);
            int dz = Math.abs(sz - camSZ);
            int bucket = Math.min(DISTANCE_BUCKETS - 1, Math.max(dx, Math.max(dy, dz)));
            dirtyDistanceBuckets[bucket].add(key);
        }

        long startNs = System.nanoTime();
        long budgetNs = (dirtySections.size() > STORM_DIRTY_THRESHOLD)
                ? MAX_DRAIN_NANOS_STORM
                : MAX_DRAIN_NANOS_STEADY;
        int dispatched = 0;
        int processed = 0;
        int maxDispatch = Math.min(MAX_SECTION_CAPTURES_PER_FRAME, MAX_IN_FLIGHT_MESHES - inFlight);
        for (int b = 0; b < DISTANCE_BUCKETS && dispatched < maxDispatch; b++) {
            List<Long> bucketList = dirtyDistanceBuckets[b];
            for (int i = 0; i < bucketList.size() && dispatched < maxDispatch; i++) {
                if (processed > 0 && (System.nanoTime() - startNs) >= budgetNs) {
                    return dispatched;
                }
                long key = bucketList.get(i);
                if (!dirtySections.remove(key)) {
                    continue;
                }
                processed++;
                int sx = SectionPos.x(key);
                int sy = SectionPos.y(key);
                int sz = SectionPos.z(key);
                if (level.getChunkSource().getChunkNow(sx, sz) == null) {
                    continue;
                }
                GreedyMesher.SectionSnapshot snapshot = GreedyMesher.SectionSnapshot.capture(level, sx, sy, sz);
                if (snapshot == null) {
                    if (vkMegaBuffer != null && vkMegaBuffer.isInitialized()) {
                        vkMegaBuffer.unloadSection(sx, sy, sz);
                    }
                } else {
                    greedyMesher.enqueue(snapshot);
                    dispatched++;
                }
            }
        }
        return dispatched;
    }

    /**
     * Queues a chunk section rebuild (deduplicated; snapshots are captured per frame to prevent light-update bursts).
     */
    public void scheduleSectionRebuild(int sectionX, int sectionY, int sectionZ) {
        if (unavailable || Minecraft.getInstance().level == null || greedyMesher == null) {
            return;
        }
        dirtySections.add(SectionPos.asLong(sectionX, sectionY, sectionZ));
    }

    /**
     * Unloads all chunk sections belonging to an unloaded chunk column.
     */
    public void unloadChunkColumn(int chunkX, int chunkZ, int minSection, int maxSection) {
        for (int sectionY = minSection; sectionY < maxSection; sectionY++) {
            dirtySections.remove(SectionPos.asLong(chunkX, sectionY, chunkZ));
        }
        if (unavailable || vkMegaBuffer == null || !vkMegaBuffer.isInitialized()) {
            return;
        }
        for (int sectionY = minSection; sectionY < maxSection; sectionY++) {
            vkMegaBuffer.unloadSection(chunkX, sectionY, chunkZ);
        }
        dirtyShadow = true;
    }

    public synchronized void onWorldReload() {
        if (unavailable || !initialized) {
            dirtySections.clear();
            return;
        }
        LOGGER.info("[Valkyrie] World reload event triggered.");
        dirtySections.clear();
        chunkProfStartNs = 0L;
        chunkProfLastLogNs = 0L;
        chunkProfDrainNs = 0L;
        chunkProfPollNs = 0L;
        chunkProfFenceNs = 0L;
        chunkProfUploadNs = 0L;
        chunkProfUploadBytes = 0L;
        chunkProfCaptured = 0;
        chunkProfUploaded = 0;
        chunkProfFrames = 0;
        chunkProfSettled = false;
        java.util.Arrays.fill(csmCascadeInitialized, false);
        dirtyShadow = true;
        if (vkContext != null && vkContext.isInitialized()) {
            vkDeviceWaitIdle(vkContext.getVkDevice());
        }
        if (vkMegaBuffer != null) {
            vkMegaBuffer.clearAllSections();
        }
        if (greedyMesher != null) {
            greedyMesher.clear();
        }
        if (atlasUVTable != null) {
            atlasUVTable.rebuildFromBlockAtlas();
        }
        if (vkShaderPackPipeline != null && vkShaderPackPipeline.isInitialized()) {
            vkShaderPackPipeline.resetTemporalState();
        }
        hasPreviousFrameHiZ = false;
        dirtyShadow = true;
    }

    public synchronized void onResize(int width, int height) {
        if (unavailable || !initialized) {
            return;
        }
        width = Math.max(1, width);
        height = Math.max(1, height);

        if (currentWidth == width && currentHeight == height) {
            return;
        }

        LOGGER.info("[Valkyrie] Resizing renderer viewport to {}x{}", width, height);
        this.currentWidth = width;
        this.currentHeight = height;
        this.hasPreviousFrameHiZ = false;

        if (vkContext != null && vkContext.isInitialized()) {
            vkDeviceWaitIdle(vkContext.getVkDevice());
        }

        if (vkInteropManager != null) {
            vkInteropManager.recreate(width, height);
        }

        if (vkPipelineManager != null && vkInteropManager != null) {
            vkPipelineManager.recreateSizeDependentResources(
                    width,
                    height,
                    vkInteropManager.getSharedColorImageView(),
                    vkInteropManager.getSharedDepthImageView()
            );
        }

        if (vkShaderPackPipeline != null && vkShaderPackPipeline.isInitialized()) {
            vkShaderPackPipeline.resize(width, height);
        }
    }

    public synchronized void shutdown() {
        if (!initialized && vkContext == null && vkInteropManager == null && vkMegaBuffer == null
                && vkPipelineManager == null && vkShaderPackPipeline == null) {
            return;
        }

        LOGGER.info("[Valkyrie] Shutting down ValkyrieRenderer...");
        destroyResourcesLocked();
        LOGGER.info("[Valkyrie] ValkyrieRenderer shut down cleanly.");
    }

    private void destroyResourcesLocked() {
        if (vkContext != null && vkContext.isInitialized()) {
            VkDevice device = vkContext.getVkDevice();
            vkDeviceWaitIdle(device);

            for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
                if (frameFences[i] != VK_NULL_HANDLE) {
                    vkDestroyFence(device, frameFences[i], null);
                    frameFences[i] = VK_NULL_HANDLE;
                }
                if (transferCompleteSemaphores[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device, transferCompleteSemaphores[i], null);
                    transferCompleteSemaphores[i] = VK_NULL_HANDLE;
                }
                if (cullCompleteSemaphores[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device, cullCompleteSemaphores[i], null);
                    cullCompleteSemaphores[i] = VK_NULL_HANDLE;
                }
                if (gbufferDoneSemaphores[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device, gbufferDoneSemaphores[i], null);
                    gbufferDoneSemaphores[i] = VK_NULL_HANDLE;
                }
                if (commandBuffers[i] != null) {
                    vkContext.freeCommandBuffer(commandBuffers[i]);
                    commandBuffers[i] = null;
                }
                if (postCommandBuffers[i] != null) {
                    vkContext.freeCommandBuffer(postCommandBuffers[i]);
                    postCommandBuffers[i] = null;
                }
                if (transferCommandBuffers[i] != null) {
                    vkContext.freeTransferCommandBuffer(transferCommandBuffers[i]);
                    transferCommandBuffers[i] = null;
                }
                if (cullComputeCommandBuffers[i] != null) {
                    vkContext.freeComputeCommandBuffer(cullComputeCommandBuffers[i]);
                    cullComputeCommandBuffers[i] = null;
                }
                if (hizComputeCommandBuffers[i] != null) {
                    vkContext.freeComputeCommandBuffer(hizComputeCommandBuffers[i]);
                    hizComputeCommandBuffers[i] = null;
                }
                fenceWaited[i] = false;
            }
            if (hizReadySemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, hizReadySemaphore, null);
                hizReadySemaphore = VK_NULL_HANDLE;
            }
            hizSemaphorePending = false;
            for (int w = 0; w < RECORD_WORKER_COUNT; w++) {
                for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
                    if (workerCommandPools[w][i] != VK_NULL_HANDLE) {
                        vkDestroyCommandPool(device, workerCommandPools[w][i], null);
                        workerCommandPools[w][i] = VK_NULL_HANDLE;
                    }
                    for (int k = 0; k < MAX_PARALLEL_CMDS_PER_WORKER; k++) {
                        workerCommandBuffers[w][i][k] = null;
                    }
                }
            }
        }

        if (recordExecutor != null) {
            recordExecutor.shutdownNow();
            recordExecutor = null;
        }

        if (greedyMesher != null) {
            greedyMesher.shutdown();
        }

        if (atlasUVTable != null) {
            atlasUVTable.destroy();
        }

        if (vkShaderPackPipeline != null) {
            vkShaderPackPipeline.close();
            vkShaderPackPipeline = null;
        }
        gpuProfiler.close(vkContext != null && vkContext.isInitialized() ? vkContext.getVkDevice() : null);
        explicitPackLoaded = false;
        pendingShaderPackReload = false;
        lastLoggedRenderPipeline = "";

        if (vkPipelineManager != null) {
            vkPipelineManager.close();
            vkPipelineManager = null;
        }

        if (vkMegaBuffer != null) {
            vkMegaBuffer.close();
            vkMegaBuffer = null;
        }

        if (vkInteropManager != null) {
            vkInteropManager.close();
            vkInteropManager = null;
        }

        if (vkContext != null) {
            vkContext.close();
            vkContext = null;
        }

        initialized = false;
    }
}
