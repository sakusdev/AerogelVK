package com.example.valkyrie;

import com.example.valkyrie.compat.CompatManager;
import com.example.valkyrie.config.ValkyrieConfig;
import com.example.valkyrie.meshing.AtlasUVTable;
import com.example.valkyrie.meshing.GreedyMesher;
import com.example.valkyrie.shaderpack.ShaderUniformLayout;
import com.example.valkyrie.shaderpack.VkShaderPackPipeline;
import com.example.valkyrie.vulkan.VkContext;
import com.example.valkyrie.vulkan.VkInteropManager;
import com.example.valkyrie.vulkan.VkMegaBuffer;
import com.example.valkyrie.vulkan.VkPipelineManager;
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
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

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
    private static final int MAX_SECTION_CAPTURES_PER_FRAME = 256;

    private final Set<Long> dirtySections = ConcurrentHashMap.newKeySet();

    private final VkCommandBuffer[] commandBuffers = new VkCommandBuffer[FRAMES_IN_FLIGHT];
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

    private final Matrix4f prevViewProj = new Matrix4f();
    private final Matrix4f[] csmMatrices = new Matrix4f[VkPipelineManager.CSM_CASCADE_COUNT];

    private ValkyrieRenderer() {
        this.atlasUVTable = AtlasUVTable.getInstance();
        this.greedyMesher = GreedyMesher.getInstance();
        for (int i = 0; i < csmMatrices.length; i++) {
            csmMatrices[i] = new Matrix4f();
        }
    }

    public synchronized void ensureInitialized(int width, int height) {
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

        // 1. Vulkan 1.3 Context
        this.vkContext = VkContext.getInstance();
        if (!vkContext.isInitialized()) {
            vkContext.init();
        }

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

        // 8. Allocate Double-Buffered Command Buffers & Frame Synchronization Fences
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
                    .flags(VK_FENCE_CREATE_SIGNALED_BIT);

            LongBuffer pFence = stack.mallocLong(1);
            for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
                this.commandBuffers[i] = vkContext.allocateCommandBuffer();
                int res = vkCreateFence(vkContext.getVkDevice(), fenceInfo, null, pFence);
                if (res != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create frame fence[" + i + "]: " + res);
                }
                this.frameFences[i] = pFence.get(0);
                this.fenceWaited[i] = true;
            }
        }

        this.initialized = true;
        LOGGER.info("[Valkyrie] ValkyrieRenderer successfully initialized.");
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
        RenderTarget mainTarget = Minecraft.getInstance().getMainRenderTarget();
        int width = (mainTarget != null && mainTarget.width > 0) ? mainTarget.width : currentWidth;
        int height = (mainTarget != null && mainTarget.height > 0) ? mainTarget.height : currentHeight;

        ensureInitialized(width, height);

        if (pendingShaderPackReload) {
            synchronized (this) {
                if (pendingShaderPackReload) {
                    pendingShaderPackReload = false;
                    reloadShaderPackLocked();
                }
            }
        }

        // 1. Wait on this frame slot's fence ONLY if compositeToOpenGL didn't already wait for it
        int slot = currentFrameSlot;
        VkDevice device = vkContext.getVkDevice();
        if (!fenceWaited[slot]) {
            vkWaitForFences(device, frameFences[slot], true, 1_000_000_000L);
            fenceWaited[slot] = true;
        }
        vkResetFences(device, frameFences[slot]);

        // 1b. Drain deduplicated dirty section coordinates and capture non-empty snapshots
        drainDirtySections(Minecraft.getInstance().level);

        // 2. Poll completed meshing results and stream into MegaBuffer
        GreedyMesher.MeshedSectionResult result;
        while ((result = greedyMesher.pollCompleted()) != null) {
            vkMegaBuffer.uploadSectionMesh(
                    result.sectionX(),
                    result.sectionY(),
                    result.sectionZ(),
                    result.packedVertices(),
                    result.quadCount(),
                    result.aabbPacked(),
                    result.flags()
            );
            dirtyShadow = true;
        }

        // 3. Refresh Block Atlas UVs & Texture if dirty
        if (atlasUVTable.isDirty()) {
            syncAtlasResources();
            updateDescriptorSet0Buffers();
        }

        // 4. Compute Matrices and Sun Direction
        ClientLevel level = Minecraft.getInstance().level;
        float partialTicks = deltaTracker != null ? deltaTracker.getGameTimeDeltaPartialTick(true) : 0f;
        float timeOfDay = (level != null) ? level.getTimeOfDay(partialTicks) : 0.25f;

        float sunAngle = (timeOfDay - 0.25f) * (float) (Math.PI * 2.0);
        Vector3f sunDir = new Vector3f((float) Math.cos(sunAngle), (float) Math.sin(sunAngle), 0.25f).normalize();

        Vector3f cameraPos = new Vector3f(
                (float) camera.getPosition().x,
                (float) camera.getPosition().y,
                (float) camera.getPosition().z
        );

        // frustumMatrix from GameRenderer already holds camera rotation inverse (world -> view rotation)
        Matrix4f viewMatrix = new Matrix4f(frustumMatrix).translate(-cameraPos.x, -cameraPos.y, -cameraPos.z);
        Matrix4f projMatrix = toVulkanReversedZ(projectionMatrix);
        Matrix4f viewProjMatrix = new Matrix4f(projMatrix).mul(viewMatrix);
        Matrix4f invViewProjMatrix = new Matrix4f(viewProjMatrix).invert();

        // Compute 3 CSM Cascade Matrices
        computeCsmMatrices(cameraPos, sunDir, viewMatrix);

        // Update Frame Uniform Buffer (UBO)
        vkPipelineManager.updateFrameUbo(
                viewMatrix,
                projMatrix,
                viewProjMatrix,
                invViewProjMatrix,
                prevViewProj,
                csmMatrices,
                sunDir,
                cameraPos,
                width,
                height,
                frameIndex,
                (float) (frameIndex * 0.016666)
        );

        prevViewProj.set(viewProjMatrix);

        ShaderUniformLayout.UniformData shaderPackUniforms = null;
        if (isShaderPackActive()) {
            shaderPackUniforms = buildShaderPackUniformData(
                    level, camera, cameraPos, frustumMatrix, projectionMatrix, partialTicks
            );
        }

        // 5. Record Command Buffer
        VkCommandBuffer activeCmd = commandBuffers[slot];
        recordFrameCommands(activeCmd, width, height, shaderPackUniforms);

        // 6. Submit Command Buffer (using external semaphores when supported, else fence sync in composite)
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer pCmd = stack.pointers(activeCmd);

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(pCmd);

            if (vkInteropManager.usesExternalSemaphores()) {
                LongBuffer pSignalSemaphores = stack.longs(vkInteropManager.getVkRenderCompleteSemaphore());
                submitInfo.pSignalSemaphores(pSignalSemaphores);

                if (vkInteropManager.consumeGlSemaphoreSignaled()) {
                    LongBuffer pWaitSemaphores = stack.longs(vkInteropManager.getVkGlReadySemaphore());
                    IntBuffer pWaitDstStageMask = stack.ints(
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                    | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                    | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                    );
                    submitInfo.waitSemaphoreCount(1)
                            .pWaitSemaphores(pWaitSemaphores)
                            .pWaitDstStageMask(pWaitDstStageMask);
                }
            }

            int res = vkQueueSubmit(vkContext.getGraphicsQueue(), submitInfo, frameFences[slot]);
            if (res != VK_SUCCESS) {
                LOGGER.error("[Valkyrie] Failed to submit render command buffer: {}", res);
            } else {
                fenceWaited[slot] = false;
            }
        }

        lastSubmittedSlot = slot;
        currentFrameSlot = (slot + 1) % FRAMES_IN_FLIGHT;
        dirtyShadow = false;
        frameIndex++;
    }

    private void recordFrameCommands(
            VkCommandBuffer commandBuffer,
            int width,
            int height,
            ShaderUniformLayout.UniformData shaderPackUniforms
    ) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkResetCommandBuffer(commandBuffer, 0);

            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

            vkBeginCommandBuffer(commandBuffer, beginInfo);

            // Staging ring -> Device Local SSBO copy
            vkMegaBuffer.recordPendingTransfers(commandBuffer);

            int activeSections = vkMegaBuffer.getActiveSectionCount();
            int slotDispatchLimit = vkMegaBuffer.getSlotDispatchLimit();

            if (activeSections > 0 && slotDispatchLimit > 0) {
                if (shaderPackUniforms != null && isShaderPackActive()) {
                    try {
                        logActivePipelineIfChanged("shaderpack:" + vkShaderPackPipeline.getActivePackName());
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
                        LOGGER.error("[Valkyrie] Native shaderpack frame failed; falling back to built-in deferred PBR pipeline.", t);
                        explicitPackLoaded = false;
                        vkShaderPackPipeline.unloadPack();
                        logActivePipelineIfChanged("builtin-pbr");
                        recordGpuDrivenTerrainPasses(
                                commandBuffer, width, height, slotDispatchLimit,
                                vkInteropManager.getSharedColorImage(),
                                vkInteropManager.getSharedDepthImage(),
                                vkInteropManager.getSharedDepthImageView(),
                                stack
                        );
                    }
                } else {
                    logActivePipelineIfChanged("builtin-pbr");
                    recordGpuDrivenTerrainPasses(
                            commandBuffer, width, height, slotDispatchLimit,
                            vkInteropManager.getSharedColorImage(),
                            vkInteropManager.getSharedDepthImage(),
                            vkInteropManager.getSharedDepthImageView(),
                            stack
                    );
                }
            } else {
                // Phase 2 fallback: animated sky/horizon procedural gradient
                hasPreviousFrameHiZ = false;
                logActivePipelineIfChanged("phase2-fallback");
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
        vkPipelineManager.updateFrameUbo(
                viewMatrix, projMatrix, viewProjMatrix, invViewProjMatrix, prevViewProj,
                csmMatrices, sunDir, cameraPos, width, height, frameIndex, (float) (frameIndex * 0.016666)
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
                                sharedColorImage, sharedDepthImage, sharedDepthImageView, stack
                        );
                    }
                } else {
                    logActivePipelineIfChanged("builtin-pbr");
                    recordGpuDrivenTerrainPasses(
                            cmd, width, height, slotDispatchLimit,
                            sharedColorImage, sharedDepthImage, sharedDepthImageView, stack
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
            MemoryStack stack
    ) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

        // --------------------------------------------------------------------
        // Pass 1: Clear DrawCountSSBO & Compute Frustum + Hi-Z Culling (chunk_cull.comp)
        // --------------------------------------------------------------------
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

        // Push constants: totalChunkCount = slotDispatchLimit, hiZEnabled
        ValkyrieConfig config = ValkyrieConfig.get();
        boolean useHiZ = hasPreviousFrameHiZ && config.isHiZOcclusionEnabled();
        ByteBuffer pcCull = stack.malloc(8);
        pcCull.putInt(0, slotDispatchLimit);
        pcCull.putInt(4, useHiZ ? 1 : 0);
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, pcCull);

        int groupCountX = (slotDispatchLimit + 63) / 64;
        vkCmdDispatch(cmd, groupCountX, 1, 1);

        // Barrier: Compute write -> Indirect Command & Draw Count read
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

        // --------------------------------------------------------------------
        // Pass 2: Cascaded Shadow Map (CSM) Dynamic Rendering Pass
        // --------------------------------------------------------------------
        if (config.isCsmEnabled()) {
            int activeCascades = Math.min(config.getCsmCascades(), VkPipelineManager.CSM_CASCADE_COUNT);
            for (int c = 0; c < activeCascades; c++) {
                if (vkPipelineManager.shouldUpdateCascade(c, frameIndex, dirtyShadow)) {
                    renderCsmCascade(cmd, c, slotDispatchLimit, stack);
                }
            }
        }

        // --------------------------------------------------------------------
        // Pass 3: G-Buffer Dynamic Rendering Pass (gbuffer.vert + gbuffer.frag)
        // --------------------------------------------------------------------
        renderGBuffer(cmd, width, height, slotDispatchLimit, stack);

        // --------------------------------------------------------------------
        // Pass 4: Deferred Lighting & Water SSR Compute Pass (deferred_lighting.comp)
        // --------------------------------------------------------------------
        renderDeferredLighting(cmd, width, height, stack);

        // --------------------------------------------------------------------
        // Pass 4b: Hi-Z Occlusion Pyramid Downsample for Next Frame (hiz_downsample.comp)
        // --------------------------------------------------------------------
        if (config.isHiZOcclusionEnabled()) {
            renderHiZPyramid(cmd, width, height, stack);
            hasPreviousFrameHiZ = true;
        } else {
            hasPreviousFrameHiZ = false;
        }

        // --------------------------------------------------------------------
        // Pass 5: Post-Process ACES Filmic Tonemapping (post_composite.comp)
        // --------------------------------------------------------------------
        renderPostComposite(cmd, width, height, sharedColorImage, stack);

        // --------------------------------------------------------------------
        // Pass 6: Convert Reversed-Z G-Buffer Depth -> Standard-Z Shared Depth
        // --------------------------------------------------------------------
        renderDepthConvert(cmd, width, height, sharedDepthImage, sharedDepthImageView, stack);
    }

    private void renderCsmCascade(VkCommandBuffer cmd, int cascade, int maxDrawCount, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());
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

    private void renderGBuffer(VkCommandBuffer cmd, int width, int height, int maxDrawCount, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

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

        VkRenderingAttachmentInfo.Buffer colorAttaches = VkRenderingAttachmentInfo.calloc(2, stack);
        colorAttaches.get(0).sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(vkPipelineManager.getRt0View())
                .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        colorAttaches.get(0).clearValue().color().float32(0, 0f).float32(1, 0f).float32(2, 0f).float32(3, 0f);

        colorAttaches.get(1).sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(vkPipelineManager.getRt1View())
                .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        colorAttaches.get(1).clearValue().color().uint32(0, 0).uint32(1, 0).uint32(2, 0).uint32(3, 0);

        VkRenderingAttachmentInfo.Buffer depthAttach = VkRenderingAttachmentInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(vkPipelineManager.getGbufferDepthView())
                .imageLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        depthAttach.clearValue().depthStencil().depth(0.0f); // Reverse-Z: 0.0 is Far, 1.0 is Near

        VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_INFO)
                .renderArea(ra -> ra.extent().set(width, height))
                .layerCount(1)
                .pColorAttachments(colorAttaches)
                .pDepthAttachment(depthAttach.get(0));

        vkCmdBeginRendering(cmd, renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineManager.getGbufferPipeline());
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
            // Transition only destination mip to GENERAL while keeping already-written mips in SHADER_READ_ONLY_OPTIMAL
            transitionImageLayout(cmd, hiZImage,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1,
                    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            pcHiZ.putInt(0, mip);
            pcHiZ.putInt(4, srcW);
            pcHiZ.putInt(8, srcH);
            pcHiZ.putInt(12, dstW);
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, pcHiZ);

            vkCmdDispatch(cmd, (dstW + 7) / 8, (dstH + 7) / 8, 1);

            // Transition this written mip to SHADER_READ_ONLY_OPTIMAL before next mip reads it
            transitionImageLayout(cmd, hiZImage,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            srcW = dstW;
            srcH = dstH;
            dstW = Math.max(1, dstW / 2);
            dstH = Math.max(1, dstH / 2);
        }
    }

    private void renderDeferredLighting(VkCommandBuffer cmd, int width, int height, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

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

    private void renderPostComposite(VkCommandBuffer cmd, int width, int height, long sharedColorImage, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

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

    private void renderDepthConvert(VkCommandBuffer cmd, int width, int height, long sharedDepthImage, long sharedDepthImageView, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

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
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);

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
            Matrix4f lightProj = new Matrix4f().ortho(-dist, dist, -dist, dist, -dist * 2f, dist * 2f, true);
            csmMatrices[i].set(lightProj).mul(lightView);
        }
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
        return initialized;
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
        ensureShaderPackPipelineInitialized();
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

    private void drainDirtySections(ClientLevel level) {
        if (level == null || greedyMesher == null || dirtySections.isEmpty()) {
            return;
        }
        if (atlasUVTable != null && atlasUVTable.getSpriteCount() <= 1) {
            atlasUVTable.rebuildFromBlockAtlas();
        }
        int dispatched = 0;
        Iterator<Long> it = dirtySections.iterator();
        while (it.hasNext() && dispatched < MAX_SECTION_CAPTURES_PER_FRAME) {
            long key = it.next();
            it.remove();
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

    /**
     * Queues a chunk section rebuild (deduplicated; snapshots are captured per frame to prevent light-update bursts).
     */
    public void scheduleSectionRebuild(int sectionX, int sectionY, int sectionZ) {
        if (Minecraft.getInstance().level == null || greedyMesher == null) {
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
        if (vkMegaBuffer == null || !vkMegaBuffer.isInitialized()) {
            return;
        }
        for (int sectionY = minSection; sectionY < maxSection; sectionY++) {
            vkMegaBuffer.unloadSection(chunkX, sectionY, chunkZ);
        }
        dirtyShadow = true;
    }

    public synchronized void onWorldReload() {
        LOGGER.info("[Valkyrie] World reload event triggered.");
        dirtySections.clear();
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
        if (!initialized) return;

        LOGGER.info("[Valkyrie] Shutting down ValkyrieRenderer...");

        if (vkContext != null && vkContext.isInitialized()) {
            VkDevice device = vkContext.getVkDevice();
            vkDeviceWaitIdle(device);

            for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
                if (frameFences[i] != VK_NULL_HANDLE) {
                    vkDestroyFence(device, frameFences[i], null);
                    frameFences[i] = VK_NULL_HANDLE;
                }
                if (commandBuffers[i] != null) {
                    vkContext.freeCommandBuffer(commandBuffers[i]);
                    commandBuffers[i] = null;
                }
                fenceWaited[i] = false;
            }
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
        explicitPackLoaded = false;
        pendingShaderPackReload = false;
        lastLoggedRenderPipeline = "";

        if (vkPipelineManager != null) {
            vkPipelineManager.close();
        }

        if (vkMegaBuffer != null) {
            vkMegaBuffer.close();
        }

        if (vkInteropManager != null) {
            vkInteropManager.close();
        }

        if (vkContext != null) {
            vkContext.close();
        }

        initialized = false;
        LOGGER.info("[Valkyrie] ValkyrieRenderer shut down cleanly.");
    }
}
