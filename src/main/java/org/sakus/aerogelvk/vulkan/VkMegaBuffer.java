package org.sakus.aerogelvk.vulkan;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.util.vma.VmaAllocationInfo;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;

/**
 * Manages GPU-driven geometry buffers, SSBOs, and indirect draw buffers using Vulkan Memory Allocator (VMA):
 * - megaVertexBuffer: 256 MB Device-Local buffer for 64-bit packed vertices with sub-allocation Free-List.
 * - chunkMetaBuffer: std430 SSBO for ChunkDrawMeta (32 bytes per section, up to 16,384 sections).
 * - indirectCmdBuffer: VkDrawIndexedIndirectCommand (20 bytes per command, up to 16,384 commands).
 * - drawCountBuffer: 4 bytes uint32_t for vkCmdDrawIndexedIndirectCount.
 * - visibleChunkIdBuffer: uint32_t array mapping gl_InstanceIndex -> chunkSlotIndex.
 * - globalQuadIndexBuffer: Pre-filled index buffer for up to 1,048,576 quads using pattern (0,1,2, 2,1,3).
 * - Staging Ring Buffer: 32 MB Persistently Mapped Host-Visible buffer for streaming mesh updates.
 */
public class VkMegaBuffer implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(VkMegaBuffer.class);
    private static VkMegaBuffer instance;

    public static final long DEFAULT_VERTEX_BUFFER_BYTES = 256L * 1024 * 1024; // 256 MB
    public static final int MAX_CHUNKS = 16384;
    public static final int CHUNK_META_SIZE_BYTES = 32;
    public static final int INDIRECT_CMD_SIZE_BYTES = 20; // VkDrawIndexedIndirectCommand
    public static final int MAX_QUADS = 1048576; // 1M quads
    public static final long STAGING_BUFFER_BYTES = 32L * 1024 * 1024; // 32 MB
    public static final int ATLAS_UV_ENTRY_BYTES = 32; // vec2 uvMin, vec2 uvMax, vec4 tint (std430)
    public static final int MAX_ATLAS_SPRITES = 65536;
    private static final int COPY_BATCH_LIMIT = 512; // Keeps VkBufferCopy array well within LWJGL 64KB MemoryStack

    public static class AllocatedBuffer {
        public final long buffer;
        public final long allocation;
        public final long size;

        public AllocatedBuffer(long buffer, long allocation, long size) {
            this.buffer = buffer;
            this.allocation = allocation;
            this.size = size;
        }
    }

    private static class FreeBlock {
        int offset; // In units of 64-bit vertices
        int length; // In units of 64-bit vertices

        FreeBlock(int offset, int length) {
            this.offset = offset;
            this.length = length;
        }
    }

    private record SectionAlloc(int vertexOffset, int vertexCount) {}
    private record RetiredAlloc(int vertexOffset, int vertexCount, long retiredFrame) {}
    private record PendingTransfer(long srcOffset, long dstOffset, long size) {}
    private record DeferredUpload(int sectionX, int sectionY, int sectionZ, int subKind,
                                  long[] packedVertices, int quadCount, int aabbPacked, int flags) {}

    // Allocated VMA buffers
    private AllocatedBuffer megaVertexBuffer;
    private AllocatedBuffer chunkMetaBuffer;
    private AllocatedBuffer indirectCmdBuffer;
    private AllocatedBuffer drawCountBuffer;
    private AllocatedBuffer visibleChunkIdBuffer;
    private AllocatedBuffer globalQuadIndexBuffer;
    private final AllocatedBuffer[] stagingBuffers = new AllocatedBuffer[VkContext.FRAMES_IN_FLIGHT];
    private AllocatedBuffer atlasUvBuffer;

    // Persistent mapped pointers
    private ByteBuffer chunkMetaMapped;
    private final ByteBuffer[] stagingMapped = new ByteBuffer[VkContext.FRAMES_IN_FLIGHT];
    private ByteBuffer atlasUvMapped;
    private final long[] stagingHeads = new long[VkContext.FRAMES_IN_FLIGHT];
    private int activeWriteSlot = 0;

    // Free-list vertex sub-allocator
    private int totalVertexCapacity;
    private final List<FreeBlock> freeBlocks = new ArrayList<>();

    // Section slot tracking
    private final Map<Long, Integer> sectionKeyToSlot = new HashMap<>();
    private final Map<Integer, SectionAlloc> slotAllocations = new HashMap<>();
    private final BitSet freeSlots = new BitSet(MAX_CHUNKS);
    private final BitSet translucentSlotBits = new BitSet(MAX_CHUNKS);
    private int highestUsedSlot = -1;

    // Pending GPU transfers & frame-delayed vertex retirement
    @SuppressWarnings("unchecked")
    private final List<PendingTransfer>[] pendingTransfersBySlot = new List[VkContext.FRAMES_IN_FLIGHT];
    private final Deque<DeferredUpload> deferredUploads = new ArrayDeque<>();
    private final List<RetiredAlloc> retiredAllocations = new ArrayList<>();
    private long currentUploadFrame = 0;
    private boolean initialized = false;

    public static synchronized VkMegaBuffer getInstance() {
        if (instance == null) {
            instance = new VkMegaBuffer();
        }
        return instance;
    }

    public static VkMegaBuffer get() {
        return getInstance();
    }

    /**
     * Advances the upload frame counter and returns retired vertex allocations
     * whose submitting frames have completed ({@code >= framesInFlight} ago).
     */
    public synchronized void advanceFrameAndReclaim(int framesInFlight) {
        currentUploadFrame++;
        if (retiredAllocations.isEmpty()) {
            return;
        }
        int retainDelay = Math.max(1, framesInFlight);
        var it = retiredAllocations.iterator();
        while (it.hasNext()) {
            RetiredAlloc retired = it.next();
            if (currentUploadFrame - retired.retiredFrame() >= retainDelay) {
                freeVertices(retired.vertexOffset(), retired.vertexCount());
                it.remove();
            }
        }
    }

    /**
     * Begins a frame slot: selects the slot-local staging ring, advances retired vertex reclamation,
     * and drains any uploads deferred from a previous frame's staging ring overflow.
     */
    public synchronized void beginFrameSlot(int frameSlot, int framesInFlight) {
        this.activeWriteSlot = Math.floorMod(frameSlot, VkContext.FRAMES_IN_FLIGHT);
        this.stagingHeads[this.activeWriteSlot] = 0;
        advanceFrameAndReclaim(framesInFlight);
        drainDeferredUploads();
    }

    private void drainDeferredUploads() {
        while (!deferredUploads.isEmpty()) {
            DeferredUpload du = deferredUploads.peekFirst();
            long byteSize = du.packedVertices() != null ? (long) du.packedVertices().length * 8L : 0L;
            if (byteSize > 0 && stagingHeads[activeWriteSlot] + byteSize > STAGING_BUFFER_BYTES) {
                break; // Still full for this slot; leave remaining uploads for next frame
            }
            deferredUploads.pollFirst();
            uploadSectionSubMeshInternal(
                    du.sectionX(), du.sectionY(), du.sectionZ(), du.subKind(),
                    du.packedVertices(), du.quadCount(), du.aabbPacked(), du.flags(), false);
        }
    }

    public VkMegaBuffer() {
        this.totalVertexCapacity = (int) (DEFAULT_VERTEX_BUFFER_BYTES / 8L);
        for (int i = 0; i < VkContext.FRAMES_IN_FLIGHT; i++) {
            this.pendingTransfersBySlot[i] = new ArrayList<>();
        }
    }

    public synchronized void init() {
        if (initialized) {
            LOGGER.warn("[Valkyrie] VkMegaBuffer already initialized.");
            return;
        }

        VkContext ctx = VkContext.getInstance();
        long maxRange = ctx.getMaxStorageBufferRange();
        long vertexBufferBytes = (maxRange > 0 ? Math.min(DEFAULT_VERTEX_BUFFER_BYTES, maxRange) : DEFAULT_VERTEX_BUFFER_BYTES) & ~7L;
        this.totalVertexCapacity = (int) (vertexBufferBytes / 8L);

        LOGGER.info("[Valkyrie] Initializing VkMegaBuffer (MegaVertexBuffer: {} MB, MaxChunks: {})...",
                vertexBufferBytes / (1024 * 1024), MAX_CHUNKS);

        long vma = ctx.getVmaAllocator();

        // 1. Mega Vertex Buffer (Device Local)
        megaVertexBuffer = createVmaBuffer(
                vma,
                vertexBufferBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                Vma.VMA_MEMORY_USAGE_GPU_ONLY,
                0
        );

        // Initialize free-list with the entire vertex buffer
        freeBlocks.clear();
        freeBlocks.add(new FreeBlock(0, totalVertexCapacity));

        // Initialize slot tracking
        freeSlots.clear();
        freeSlots.set(0, MAX_CHUNKS);
        sectionKeyToSlot.clear();
        slotAllocations.clear();

        // 2. Chunk Metadata Buffer (32 bytes * 16384 sections = 512 KB, Host Visible Coherent Mapped)
        long chunkMetaBytes = (long) MAX_CHUNKS * CHUNK_META_SIZE_BYTES;
        chunkMetaBuffer = createVmaBuffer(
                vma,
                chunkMetaBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                Vma.VMA_MEMORY_USAGE_CPU_TO_GPU,
                Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | Vma.VMA_ALLOCATION_CREATE_MAPPED_BIT
        );
        chunkMetaMapped = mapBuffer(vma, chunkMetaBuffer);

        // 3. Indirect Command Buffer (20 bytes * 16384 = 320 KB)
        long indirectBytes = (long) MAX_CHUNKS * INDIRECT_CMD_SIZE_BYTES;
        indirectCmdBuffer = createVmaBuffer(
                vma,
                indirectBytes,
                VK10.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                Vma.VMA_MEMORY_USAGE_GPU_ONLY,
                0
        );

        // 4. Draw Count Buffer (4 bytes uint32_t)
        drawCountBuffer = createVmaBuffer(
                vma,
                4L,
                VK10.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                        | VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                Vma.VMA_MEMORY_USAGE_GPU_ONLY,
                0
        );

        // 5. Visible Chunk ID Buffer (4 bytes * 16384 = 64 KB)
        long visibleChunkBytes = (long) MAX_CHUNKS * 4L;
        visibleChunkIdBuffer = createVmaBuffer(
                vma,
                visibleChunkBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                Vma.VMA_MEMORY_USAGE_GPU_ONLY,
                0
        );

        // 6. Global Quad Index Buffer (1,048,576 quads * 6 indices * 4 bytes = ~24 MB, Device-Local)
        initGlobalQuadIndexBuffer(ctx, vma);

        // 7. Per-slot Staging Ring Buffers (32 MB each, Host Visible Coherent Mapped)
        for (int i = 0; i < VkContext.FRAMES_IN_FLIGHT; i++) {
            stagingBuffers[i] = createVmaBuffer(
                    vma,
                    STAGING_BUFFER_BYTES,
                    VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    Vma.VMA_MEMORY_USAGE_CPU_TO_GPU,
                    Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | Vma.VMA_ALLOCATION_CREATE_MAPPED_BIT
            );
            stagingMapped[i] = mapBuffer(vma, stagingBuffers[i]);
            stagingHeads[i] = 0;
        }
        activeWriteSlot = 0;

        // 8. Atlas UV Table SSBO (2 MB, Host Visible Coherent Mapped)
        long atlasUvBytes = (long) MAX_ATLAS_SPRITES * ATLAS_UV_ENTRY_BYTES;
        atlasUvBuffer = createVmaBuffer(
                vma,
                atlasUvBytes,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                Vma.VMA_MEMORY_USAGE_CPU_TO_GPU,
                Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | Vma.VMA_ALLOCATION_CREATE_MAPPED_BIT
        );
        atlasUvMapped = mapBuffer(vma, atlasUvBuffer);

        initialized = true;
        LOGGER.info("[Valkyrie] VkMegaBuffer successfully initialized.");
    }

    private void initGlobalQuadIndexBuffer(VkContext ctx, long vma) {
        long indexByteSize = (long) MAX_QUADS * 6L * 4L; // ~24 MB
        globalQuadIndexBuffer = createVmaBuffer(
                vma,
                indexByteSize,
                VK10.VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                Vma.VMA_MEMORY_USAGE_GPU_ONLY,
                0
        );

        // Create temporary staging buffer to upload pre-filled indices
        AllocatedBuffer indexStaging = createVmaBuffer(
                vma,
                indexByteSize,
                VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                Vma.VMA_MEMORY_USAGE_CPU_TO_GPU,
                Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | Vma.VMA_ALLOCATION_CREATE_MAPPED_BIT
        );

        ByteBuffer stagingMap = mapBuffer(vma, indexStaging);
        IntBuffer intBuf = stagingMap.asIntBuffer();

        // Populate indices with consistent CCW pattern (0, 1, 2, 0, 2, 3) per quad
        // Quad q uses vertices 4*q + (0, 1, 2, 3)
        for (int q = 0; q < MAX_QUADS; q++) {
            int base = q * 4;
            intBuf.put(base + 0);
            intBuf.put(base + 1);
            intBuf.put(base + 2);
            intBuf.put(base + 0);
            intBuf.put(base + 2);
            intBuf.put(base + 3);
        }

        // Copy from staging to device-local index buffer
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkBufferCopy.Buffer copyRegion = VkBufferCopy.calloc(1, stack)
                    .srcOffset(0)
                    .dstOffset(0)
                    .size(indexByteSize);
            VK10.vkCmdCopyBuffer(cmd, indexStaging.buffer, globalQuadIndexBuffer.buffer, copyRegion);
        }
        ctx.endSingleTimeCommands(cmd);

        // Cleanup temporary staging buffer
        Vma.vmaUnmapMemory(vma, indexStaging.allocation);
        Vma.vmaDestroyBuffer(vma, indexStaging.buffer, indexStaging.allocation);

        LOGGER.info("[Valkyrie] Global Quad Index Buffer pre-filled and uploaded ({} quads, {} MB).",
                MAX_QUADS, indexByteSize / (1024 * 1024));
    }

    /**
     * Uploads or updates a chunk section mesh.
     * Allocates vertices in megaVertexBuffer, streams data via Staging Ring Buffer,
     * and updates ChunkDrawMeta SSBO.
     */
    public synchronized int uploadSectionMesh(int sectionX, int sectionY, int sectionZ,
                                              long[] packedVertices, int quadCount,
                                              int aabbPacked, int flags) {
        int subKind = ((flags & 4) != 0 ? 2 : 0) | ((flags & 1) != 0 ? 1 : 0);
        return uploadSectionSubMesh(sectionX, sectionY, sectionZ, subKind, packedVertices, quadCount, aabbPacked, flags);
    }

    /**
     * Uploads or updates a chunk section sub-mesh (0=opaque full, 1=translucent, 2=opaque half-scale, 3=translucent half-scale).
     */
    public synchronized int uploadSectionSubMesh(int sectionX, int sectionY, int sectionZ, int subKind,
                                                 long[] packedVertices, int quadCount,
                                                 int aabbPacked, int flags) {
        return uploadSectionSubMeshInternal(sectionX, sectionY, sectionZ, subKind, packedVertices, quadCount, aabbPacked, flags, true);
    }

    private int uploadSectionSubMeshInternal(int sectionX, int sectionY, int sectionZ, int subKind,
                                             long[] packedVertices, int quadCount,
                                             int aabbPacked, int flags, boolean allowDefer) {
        if (!initialized) {
            throw new IllegalStateException("VkMegaBuffer not initialized!");
        }

        boolean hasGeometry = quadCount > 0 && packedVertices != null && packedVertices.length > 0;
        if (hasGeometry) {
            long byteSize = (long) packedVertices.length * 8L;
            if (byteSize > STAGING_BUFFER_BYTES) {
                LOGGER.error("[Valkyrie] Section ({}, {}, {}) subKind={} exceeds staging ring capacity ({} > {} bytes); skipping.",
                        sectionX, sectionY, sectionZ, subKind, byteSize, STAGING_BUFFER_BYTES);
                return -1;
            }
            // C8 fix: if the current slot's staging ring is full, defer to next frame WITHOUT vkQueueWaitIdle.
            // Un-uploaded sections are never published to ChunkDrawMeta until their transfer is actually queued.
            if (stagingHeads[activeWriteSlot] + byteSize > STAGING_BUFFER_BYTES) {
                if (allowDefer) {
                    deferredUploads.addLast(new DeferredUpload(
                            sectionX, sectionY, sectionZ, subKind, packedVertices, quadCount, aabbPacked, flags));
                }
                return -1;
            }
        }

        long sectionKey = packSectionKey(sectionX, sectionY, sectionZ, subKind);
        Integer existingSlot = sectionKeyToSlot.get(sectionKey);

        // Do not allocate a new slot for an empty companion sub-mesh (translucent/half-scale)
        if (existingSlot == null && subKind != 0 && !hasGeometry) {
            return -1;
        }

        int slot;
        if (existingSlot != null) {
            slot = existingSlot;
            // Retire previous vertex allocation for frame-delayed free (avoids overwriting in-flight GPU reads)
            SectionAlloc prevAlloc = slotAllocations.remove(slot);
            if (prevAlloc != null && prevAlloc.vertexCount > 0) {
                retiredAllocations.add(new RetiredAlloc(prevAlloc.vertexOffset, prevAlloc.vertexCount, currentUploadFrame));
            }
        } else {
            slot = freeSlots.nextSetBit(0);
            if (slot < 0 || slot >= MAX_CHUNKS) {
                LOGGER.error("[Valkyrie] Out of chunk section slots! Maximum is {}", MAX_CHUNKS);
                return -1;
            }
            freeSlots.clear(slot);
            sectionKeyToSlot.put(sectionKey, slot);
            if (slot > highestUsedSlot) {
                highestUsedSlot = slot;
            }
        }

        // Empty section (no geometry)
        if (!hasGeometry) {
            translucentSlotBits.clear(slot);
            writeChunkMeta(slot, sectionX, sectionY, sectionZ, 0, 0, aabbPacked, flags);
            slotAllocations.put(slot, new SectionAlloc(0, 0));
            return slot;
        }

        int vertexCount = packedVertices.length;
        int vertexOffset = allocateVertices(vertexCount);
        if (vertexOffset < 0) {
            translucentSlotBits.clear(slot);
            LOGGER.error("[Valkyrie] Out of Mega Vertex Buffer memory for section ({}, {}, {})! Needed {} vertices.",
                    sectionX, sectionY, sectionZ, vertexCount);
            writeChunkMeta(slot, sectionX, sectionY, sectionZ, 0, 0, aabbPacked, flags);
            return slot;
        }

        long byteSize = (long) vertexCount * 8L;
        long srcOffset = stagingHeads[activeWriteSlot];
        long dstOffset = (long) vertexOffset * 8L;

        // Copy vertices to persistently mapped slot staging buffer
        ByteBuffer mapped = stagingMapped[activeWriteSlot];
        mapped.position((int) srcOffset);
        mapped.asLongBuffer().put(packedVertices);

        // Advance staging ring head (aligned to 64 bytes)
        stagingHeads[activeWriteSlot] = (srcOffset + byteSize + 63L) & ~63L;

        // Record pending GPU copy for this frame slot
        pendingTransfersBySlot[activeWriteSlot].add(new PendingTransfer(srcOffset, dstOffset, byteSize));

        // Update ChunkDrawMeta SSBO only after staging copy is queued
        translucentSlotBits.set(slot, (flags & 1) != 0);
        writeChunkMeta(slot, sectionX, sectionY, sectionZ, vertexOffset, quadCount, aabbPacked, flags);
        slotAllocations.put(slot, new SectionAlloc(vertexOffset, vertexCount));

        return slot;
    }

    public synchronized boolean hasTranslucentSections() {
        return !translucentSlotBits.isEmpty();
    }

    /**
     * Frees a chunk section's allocation and slots (including translucent/half-scale sub-meshes) when unloaded from world.
     */
    public synchronized void unloadSection(int sectionX, int sectionY, int sectionZ) {
        for (int subKind = 0; subKind < 4; subKind++) {
            long sectionKey = packSectionKey(sectionX, sectionY, sectionZ, subKind);
            Integer slot = sectionKeyToSlot.remove(sectionKey);
            if (slot != null) {
                SectionAlloc alloc = slotAllocations.remove(slot);
                if (alloc != null && alloc.vertexCount > 0) {
                    retiredAllocations.add(new RetiredAlloc(alloc.vertexOffset, alloc.vertexCount, currentUploadFrame));
                }
                translucentSlotBits.clear(slot);
                writeChunkMeta(slot, 0, 0, 0, 0, 0, 0, 0);
                freeSlots.set(slot);
                if (slot == highestUsedSlot) {
                    while (highestUsedSlot >= 0 && freeSlots.get(highestUsedSlot)) {
                        highestUsedSlot--;
                    }
                }
            }
        }
    }

    /**
     * Uploads the Atlas UV + tint table into the mapped AtlasUVSSBO.
     */
    public synchronized void uploadAtlasUvTable(ByteBuffer uvTableData) {
        if (!initialized || atlasUvMapped == null || uvTableData == null) {
            return;
        }
        ByteBuffer src = uvTableData.duplicate();
        src.position(0);
        int copyBytes = Math.min(src.remaining(), MAX_ATLAS_SPRITES * ATLAS_UV_ENTRY_BYTES);
        src.limit(copyBytes);
        atlasUvMapped.position(0);
        atlasUvMapped.put(src);
        atlasUvMapped.position(0);
    }

    public synchronized boolean hasPendingTransfers(int frameSlot) {
        int slot = Math.floorMod(frameSlot, VkContext.FRAMES_IN_FLIGHT);
        return initialized && !pendingTransfersBySlot[slot].isEmpty();
    }

    public synchronized boolean hasDeferredUploads() {
        return !deferredUploads.isEmpty();
    }

    /**
     * Records all pending staging-to-device transfers for the current active slot
     * (single-command-buffer / headless convenience wrapper).
     */
    public synchronized void recordPendingTransfers(VkCommandBuffer cmd) {
        recordPendingTransfersForSlot(activeWriteSlot, cmd, false);
    }

    /**
     * Records all pending staging-to-device transfers for {@code frameSlot}.
     * When {@code releaseToGraphics} is true (dedicated transfer queue in a different family),
     * emits a Queue Family Ownership Transfer release barrier instead of a shader-read barrier;
     * the graphics command buffer must pair this with {@link #recordTransferAcquireBarrier(VkCommandBuffer)}.
     */
    public synchronized boolean recordPendingTransfersForSlot(int frameSlot, VkCommandBuffer cmd, boolean releaseToGraphics) {
        int slot = Math.floorMod(frameSlot, VkContext.FRAMES_IN_FLIGHT);
        List<PendingTransfer> list = pendingTransfersBySlot[slot];
        if (!initialized || list.isEmpty()) {
            stagingHeads[slot] = 0;
            return false;
        }

        AllocatedBuffer srcStaging = stagingBuffers[slot];
        int total = list.size();
        for (int start = 0; start < total; start += COPY_BATCH_LIMIT) {
            int batchCount = Math.min(COPY_BATCH_LIMIT, total - start);
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkBufferCopy.Buffer copyRegions = VkBufferCopy.calloc(batchCount, stack);
                for (int i = 0; i < batchCount; i++) {
                    PendingTransfer pt = list.get(start + i);
                    copyRegions.get(i)
                            .srcOffset(pt.srcOffset)
                            .dstOffset(pt.dstOffset)
                            .size(pt.size);
                }
                VK10.vkCmdCopyBuffer(cmd, srcStaging.buffer, megaVertexBuffer.buffer, copyRegions);
            }
        }

        VkContext ctx = VkContext.getInstance();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            if (releaseToGraphics && ctx.hasSeparateTransferQueueFamily()) {
                VkBufferMemoryBarrier.Buffer releaseBarrier = VkBufferMemoryBarrier.calloc(1, stack)
                        .sType(VK10.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                        .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                        .dstAccessMask(0)
                        .srcQueueFamilyIndex(ctx.getTransferQueueFamilyIndex())
                        .dstQueueFamilyIndex(ctx.getQueueFamilyIndex())
                        .buffer(megaVertexBuffer.buffer)
                        .offset(0)
                        .size(VK10.VK_WHOLE_SIZE);

                VK10.vkCmdPipelineBarrier(
                        cmd,
                        VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK10.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        0,
                        null,
                        releaseBarrier,
                        null
                );
            } else {
                // Same queue family: standard transfer write -> shader read barrier
                VkBufferMemoryBarrier.Buffer barrier = VkBufferMemoryBarrier.calloc(1, stack)
                        .sType(VK10.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                        .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                        .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT)
                        .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                        .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                        .buffer(megaVertexBuffer.buffer)
                        .offset(0)
                        .size(VK10.VK_WHOLE_SIZE);

                VK10.vkCmdPipelineBarrier(
                        cmd,
                        VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK10.VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0,
                        null,
                        barrier,
                        null
                );
            }
        }

        list.clear();
        stagingHeads[slot] = 0;
        return true;
    }

    /**
     * Records the graphics-side Queue Family Ownership Transfer acquire barrier matching
     * {@link #recordPendingTransfersForSlot(int, VkCommandBuffer, boolean)} with {@code releaseToGraphics=true}.
     */
    public synchronized void recordTransferAcquireBarrier(VkCommandBuffer graphicsCmd) {
        if (!initialized || megaVertexBuffer == null) {
            return;
        }
        VkContext ctx = VkContext.getInstance();
        if (!ctx.hasSeparateTransferQueueFamily()) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkBufferMemoryBarrier.Buffer acquireBarrier = VkBufferMemoryBarrier.calloc(1, stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                    .srcAccessMask(0)
                    .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT)
                    .srcQueueFamilyIndex(ctx.getTransferQueueFamilyIndex())
                    .dstQueueFamilyIndex(ctx.getQueueFamilyIndex())
                    .buffer(megaVertexBuffer.buffer)
                    .offset(0)
                    .size(VK10.VK_WHOLE_SIZE);

            VK10.vkCmdPipelineBarrier(
                    graphicsCmd,
                    VK10.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK10.VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    null,
                    acquireBarrier,
                    null
            );
        }
    }

    private void writeChunkMeta(int slot, int x, int y, int z, int vertexOffset, int quadCount, int aabbPacked, int flags) {
        int base = slot * CHUNK_META_SIZE_BYTES;
        chunkMetaMapped.putInt(base + 0, x);
        chunkMetaMapped.putInt(base + 4, y);
        chunkMetaMapped.putInt(base + 8, z);
        chunkMetaMapped.putInt(base + 12, vertexOffset);
        chunkMetaMapped.putInt(base + 16, quadCount);
        chunkMetaMapped.putInt(base + 20, aabbPacked);
        chunkMetaMapped.putInt(base + 24, flags);
        chunkMetaMapped.putInt(base + 28, 0); // padding
    }

    private int allocateVertices(int count) {
        for (int i = 0; i < freeBlocks.size(); i++) {
            FreeBlock block = freeBlocks.get(i);
            if (block.length >= count) {
                int allocatedOffset = block.offset;
                if (block.length == count) {
                    freeBlocks.remove(i);
                } else {
                    block.offset += count;
                    block.length -= count;
                }
                return allocatedOffset;
            }
        }
        return -1;
    }

    private void freeVertices(int offset, int count) {
        if (count <= 0) return;

        // Insert new free block in offset-sorted order
        int insertIndex = 0;
        while (insertIndex < freeBlocks.size() && freeBlocks.get(insertIndex).offset < offset) {
            insertIndex++;
        }
        freeBlocks.add(insertIndex, new FreeBlock(offset, count));

        // Coalesce adjacent free blocks
        for (int i = 0; i < freeBlocks.size() - 1; ) {
            FreeBlock curr = freeBlocks.get(i);
            FreeBlock next = freeBlocks.get(i + 1);

            if (curr.offset + curr.length == next.offset) {
                curr.length += next.length;
                freeBlocks.remove(i + 1);
            } else {
                i++;
            }
        }
    }

    private static long packSectionKey(int x, int y, int z, int subKind) {
        return (((long) x & 0x3FFFFFL) << 42)
                | (((long) subKind & 0xFL) << 38)
                | (((long) y & 0xFFFFL) << 22)
                | ((long) z & 0x3FFFFFL);
    }

    private AllocatedBuffer createVmaBuffer(long vma, long size, int usage, int memoryUsage, int allocationFlags) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(size)
                    .usage(usage)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE);

            VmaAllocationCreateInfo allocInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(memoryUsage)
                    .flags(allocationFlags);

            LongBuffer pBuffer = stack.mallocLong(1);
            PointerBuffer pAllocation = stack.mallocPointer(1);

            int res = Vma.vmaCreateBuffer(vma, bufferInfo, allocInfo, pBuffer, pAllocation, null);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create VMA buffer of size " + size + ", error: " + res);
            }

            return new AllocatedBuffer(pBuffer.get(0), pAllocation.get(0), size);
        }
    }

    private ByteBuffer mapBuffer(long vma, AllocatedBuffer buffer) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer pData = stack.mallocPointer(1);
            int res = Vma.vmaMapMemory(vma, buffer.allocation, pData);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to map VMA memory: " + res);
            }
            return MemoryUtil.memByteBuffer(pData.get(0), (int) buffer.size);
        }
    }

    public AllocatedBuffer getMegaVertexBuffer() {
        return megaVertexBuffer;
    }

    public AllocatedBuffer getChunkMetaBuffer() {
        return chunkMetaBuffer;
    }

    public AllocatedBuffer getIndirectCmdBuffer() {
        return indirectCmdBuffer;
    }

    public AllocatedBuffer getDrawCountBuffer() {
        return drawCountBuffer;
    }

    public AllocatedBuffer getVisibleChunkIdBuffer() {
        return visibleChunkIdBuffer;
    }

    public AllocatedBuffer getGlobalQuadIndexBuffer() {
        return globalQuadIndexBuffer;
    }

    public AllocatedBuffer getStagingBuffer() {
        return stagingBuffers[activeWriteSlot];
    }

    public AllocatedBuffer getAtlasUvBuffer() {
        return atlasUvBuffer;
    }

    public synchronized int getActiveSectionCount() {
        return sectionKeyToSlot.size();
    }

    public synchronized int getDeferredUploadCount() {
        return deferredUploads.size();
    }

    public synchronized long getStagingUsedBytes(int slot) {
        return stagingHeads[Math.floorMod(slot, VkContext.FRAMES_IN_FLIGHT)];
    }

    /**
     * Returns the upper bound (exclusive) of allocated ChunkDrawMeta slots to dispatch
     * in chunk_cull.comp, accounting for sparse freed slots.
     */
    public synchronized int getSlotDispatchLimit() {
        return highestUsedSlot + 1;
    }

    public synchronized void clearAllSections() {
        for (Integer slot : new ArrayList<>(sectionKeyToSlot.values())) {
            SectionAlloc alloc = slotAllocations.remove(slot);
            if (alloc != null && alloc.vertexCount > 0) {
                freeVertices(alloc.vertexOffset, alloc.vertexCount);
            }
            writeChunkMeta(slot, 0, 0, 0, 0, 0, 0, 0);
            freeSlots.set(slot);
        }
        sectionKeyToSlot.clear();
        for (int i = 0; i < VkContext.FRAMES_IN_FLIGHT; i++) {
            pendingTransfersBySlot[i].clear();
            stagingHeads[i] = 0;
        }
        deferredUploads.clear();
        highestUsedSlot = -1;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public synchronized void cleanup() {
        close();
    }

    @Override
    public synchronized void close() {
        if (!initialized) {
            return;
        }

        LOGGER.info("[Valkyrie] Cleaning up VkMegaBuffer resources...");
        VkContext ctx = VkContext.getInstance();
        long vma = ctx.getVmaAllocator();

        if (vma != 0) {
            if (atlasUvBuffer != null) {
                Vma.vmaUnmapMemory(vma, atlasUvBuffer.allocation);
                Vma.vmaDestroyBuffer(vma, atlasUvBuffer.buffer, atlasUvBuffer.allocation);
                atlasUvBuffer = null;
                atlasUvMapped = null;
            }

            for (int i = 0; i < VkContext.FRAMES_IN_FLIGHT; i++) {
                if (stagingBuffers[i] != null) {
                    Vma.vmaUnmapMemory(vma, stagingBuffers[i].allocation);
                    Vma.vmaDestroyBuffer(vma, stagingBuffers[i].buffer, stagingBuffers[i].allocation);
                    stagingBuffers[i] = null;
                    stagingMapped[i] = null;
                }
                stagingHeads[i] = 0;
            }

            if (chunkMetaBuffer != null) {
                Vma.vmaUnmapMemory(vma, chunkMetaBuffer.allocation);
                Vma.vmaDestroyBuffer(vma, chunkMetaBuffer.buffer, chunkMetaBuffer.allocation);
                chunkMetaBuffer = null;
                chunkMetaMapped = null;
            }

            if (globalQuadIndexBuffer != null) {
                Vma.vmaDestroyBuffer(vma, globalQuadIndexBuffer.buffer, globalQuadIndexBuffer.allocation);
                globalQuadIndexBuffer = null;
            }

            if (visibleChunkIdBuffer != null) {
                Vma.vmaDestroyBuffer(vma, visibleChunkIdBuffer.buffer, visibleChunkIdBuffer.allocation);
                visibleChunkIdBuffer = null;
            }

            if (drawCountBuffer != null) {
                Vma.vmaDestroyBuffer(vma, drawCountBuffer.buffer, drawCountBuffer.allocation);
                drawCountBuffer = null;
            }

            if (indirectCmdBuffer != null) {
                Vma.vmaDestroyBuffer(vma, indirectCmdBuffer.buffer, indirectCmdBuffer.allocation);
                indirectCmdBuffer = null;
            }

            if (megaVertexBuffer != null) {
                Vma.vmaDestroyBuffer(vma, megaVertexBuffer.buffer, megaVertexBuffer.allocation);
                megaVertexBuffer = null;
            }
        }

        freeBlocks.clear();
        sectionKeyToSlot.clear();
        slotAllocations.clear();
        freeSlots.clear();
        for (int i = 0; i < VkContext.FRAMES_IN_FLIGHT; i++) {
            pendingTransfersBySlot[i].clear();
        }
        deferredUploads.clear();

        initialized = false;
        LOGGER.info("[Valkyrie] VkMegaBuffer destroyed cleanly.");
    }
}
