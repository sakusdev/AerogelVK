package com.example.valkyrie;

import com.example.valkyrie.meshing.AtlasUVTable;
import com.example.valkyrie.meshing.GreedyMesher;
import com.example.valkyrie.vulkan.VkContext;
import com.example.valkyrie.vulkan.VkInteropManager;
import com.example.valkyrie.vulkan.VkMegaBuffer;
import com.example.valkyrie.vulkan.VkPipelineManager;
import com.mojang.blaze3d.pipeline.RenderTarget;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.lwjgl.vulkan.KHRDrawIndirectCount.vkCmdDrawIndexedIndirectCountKHR;
import static org.lwjgl.vulkan.VK10.*;
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
    private AtlasUVTable atlasUVTable;
    private GreedyMesher greedyMesher;

    private VkCommandBuffer commandBuffer;
    private long frameFence;

    private int currentWidth = 0;
    private int currentHeight = 0;
    private long frameIndex = 0;
    private boolean dirtyShadow = true;
    private boolean initialized = false;

    private final Matrix4f prevViewProj = new Matrix4f();
    private final Matrix4f[] csmMatrices = new Matrix4f[VkPipelineManager.CSM_CASCADE_COUNT];

    private ValkyrieRenderer() {
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

        // 7. Update Descriptor Set 0 with MegaBuffer SSBOs & Atlas
        updateDescriptorSet0Buffers();

        // 8. Allocate Primary Command Buffer & Frame Synchronization Fence
        try (MemoryStack stack = MemoryStack.stackPush()) {
            this.commandBuffer = vkContext.allocateCommandBuffer();

            VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
                    .flags(VK_FENCE_CREATE_SIGNALED_BIT);

            LongBuffer pFence = stack.mallocLong(1);
            int res = vkCreateFence(vkContext.getVkDevice(), fenceInfo, null, pFence);
            if (res != VK_SUCCESS) {
                throw new RuntimeException("Failed to create frame fence: " + res);
            }
            this.frameFence = pFence.get(0);
        }

        this.initialized = true;
        LOGGER.info("[Valkyrie] ValkyrieRenderer successfully initialized.");
    }

    private void updateDescriptorSet0Buffers() {
        vkPipelineManager.updateDescriptorSet0(
                vkMegaBuffer.getMegaVertexBuffer().buffer,
                vkMegaBuffer.getChunkMetaBuffer().buffer,
                vkMegaBuffer.getIndirectCmdBuffer().buffer,
                vkMegaBuffer.getDrawCountBuffer().buffer,
                vkMegaBuffer.getVisibleChunkIdBuffer().buffer,
                vkMegaBuffer.getChunkMetaBuffer().buffer, // fallback or staging buffer for atlas
                VK_NULL_HANDLE
        );
    }

    /**
     * Primary rendering hook called from MixinLevelRenderer.
     */
    public void renderWorld(Camera camera, Matrix4f frustumMatrix, Matrix4f projectionMatrix, DeltaTracker deltaTracker) {
        RenderTarget mainTarget = Minecraft.getInstance().getMainRenderTarget();
        int width = (mainTarget != null && mainTarget.width > 0) ? mainTarget.width : currentWidth;
        int height = (mainTarget != null && mainTarget.height > 0) ? mainTarget.height : currentHeight;

        ensureInitialized(width, height);

        // 1. Poll completed meshing results and stream into MegaBuffer
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

        // 2. Refresh Block Atlas UVs if dirty
        if (atlasUVTable.isDirty()) {
            atlasUVTable.clearDirty();
            updateDescriptorSet0Buffers();
        }

        // 3. Compute Matrices and Sun Direction
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

        Matrix4f viewMatrix = new Matrix4f();
        camera.rotation().get(viewMatrix);
        viewMatrix.scale(1.0f, -1.0f, -1.0f); // OpenGL to Vulkan coordinate convention

        Matrix4f projMatrix = new Matrix4f(projectionMatrix);
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

        // 4. Wait on previous frame fence
        VkDevice device = vkContext.getVkDevice();
        vkWaitForFences(device, frameFence, true, 1_000_000_000L);
        vkResetFences(device, frameFence);

        // 5. Record Command Buffer
        recordFrameCommands(width, height);

        // 6. Submit Command Buffer
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer pCmd = stack.pointers(commandBuffer);

            LongBuffer pSignalSemaphores = stack.longs(vkInteropManager.getVkRenderCompleteSemaphore());
            IntBuffer pWaitDstStageMask = stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(pCmd)
                    .pSignalSemaphores(pSignalSemaphores);

            int res = vkQueueSubmit(vkContext.getGraphicsQueue(), submitInfo, frameFence);
            if (res != VK_SUCCESS) {
                LOGGER.error("[Valkyrie] Failed to submit render command buffer: {}", res);
            }
        }

        dirtyShadow = false;
        frameIndex++;
    }

    private void recordFrameCommands(int width, int height) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkResetCommandBuffer(commandBuffer, 0);

            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

            vkBeginCommandBuffer(commandBuffer, beginInfo);

            // Staging ring -> Device Local SSBO copy
            vkMegaBuffer.recordPendingTransfers(commandBuffer);

            int activeSections = vkMegaBuffer.getActiveSectionCount();

            if (activeSections > 0) {
                recordGpuDrivenTerrainPasses(commandBuffer, width, height, activeSections, stack);
            } else {
                // Phase 2 fallback: animated sky/horizon procedural gradient
                recordPhase2FallbackPass(commandBuffer, width, height, stack);
            }

            vkEndCommandBuffer(commandBuffer);
        }
    }

    private void recordGpuDrivenTerrainPasses(VkCommandBuffer cmd, int width, int height, int activeSections, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

        // --------------------------------------------------------------------
        // Pass 1: Compute Frustum + Hi-Z Culling (chunk_cull.comp)
        // --------------------------------------------------------------------
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipelineManager.getChunkCullPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, pSets, null);

        // Push constants: totalChunkCount = activeSections, hiZEnabled = 0
        ByteBuffer pcCull = stack.malloc(8);
        pcCull.putInt(0, activeSections);
        pcCull.putInt(4, 0);
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcCull);

        int groupCountX = (activeSections + 63) / 64;
        vkCmdDispatch(cmd, groupCountX, 1, 1);

        // Barrier: Compute write -> Indirect Command & Draw Count read
        VkBufferMemoryBarrier.Buffer indirectBarriers = VkBufferMemoryBarrier.calloc(2, stack);
        indirectBarriers.get(0).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
                .buffer(vkMegaBuffer.getIndirectCmdBuffer().buffer)
                .offset(0).size(VK_WHOLE_SIZE);

        indirectBarriers.get(1).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
                .buffer(vkMegaBuffer.getDrawCountBuffer().buffer)
                .offset(0).size(VK_WHOLE_SIZE);

        vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                0, null, indirectBarriers, null);

        // --------------------------------------------------------------------
        // Pass 2: Cascaded Shadow Map (CSM) Dynamic Rendering Pass
        // --------------------------------------------------------------------
        for (int c = 0; c < VkPipelineManager.CSM_CASCADE_COUNT; c++) {
            if (vkPipelineManager.shouldUpdateCascade(c, frameIndex, dirtyShadow)) {
                renderCsmCascade(cmd, c, activeSections, stack);
            }
        }

        // --------------------------------------------------------------------
        // Pass 3: G-Buffer Dynamic Rendering Pass (gbuffer.vert + gbuffer.frag)
        // --------------------------------------------------------------------
        renderGBuffer(cmd, width, height, activeSections, stack);

        // --------------------------------------------------------------------
        // Pass 4: Deferred Lighting & Water SSR Compute Pass (deferred_lighting.comp)
        // --------------------------------------------------------------------
        renderDeferredLighting(cmd, width, height, stack);

        // --------------------------------------------------------------------
        // Pass 5: Post-Process ACES Filmic Tonemapping (post_composite.comp)
        // --------------------------------------------------------------------
        renderPostComposite(cmd, width, height, stack);
    }

    private void renderCsmCascade(VkCommandBuffer cmd, int cascade, int activeSections, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

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
                .renderArea(ra -> ra.extent().set(VkPipelineManager.CSM_RESOLUTION, VkPipelineManager.CSM_RESOLUTION))
                .layerCount(1)
                .pDepthAttachment(depthAttach.get(0));

        vkCmdBeginRendering(cmd, renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineManager.getCsmShadowPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, pSets, null);

        // Push cascade index
        ByteBuffer pcCsm = stack.malloc(4);
        pcCsm.putInt(0, cascade);
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, pcCsm);

        VkViewport.Buffer vp = VkViewport.calloc(1, stack)
                .x(0).y(0).width(VkPipelineManager.CSM_RESOLUTION).height(VkPipelineManager.CSM_RESOLUTION).minDepth(0f).maxDepth(1f);
        vkCmdSetViewport(cmd, 0, vp);

        VkRect2D.Buffer sc = VkRect2D.calloc(1, stack);
        sc.extent().set(VkPipelineManager.CSM_RESOLUTION, VkPipelineManager.CSM_RESOLUTION);
        vkCmdSetScissor(cmd, 0, sc);

        vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 1.75f);

        vkCmdBindIndexBuffer(cmd, vkMegaBuffer.getGlobalQuadIndexBuffer().buffer, 0, VK_INDEX_TYPE_UINT32);

        // Draw indexed indirect count
        vkCmdDrawIndexedIndirectCountKHR(
                cmd,
                vkMegaBuffer.getIndirectCmdBuffer().buffer,
                0,
                vkMegaBuffer.getDrawCountBuffer().buffer,
                0,
                activeSections,
                20
        );

        vkCmdEndRendering(cmd);

        transitionImageLayout(cmd, vkPipelineManager.getCsmArrayImage(),
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, cascade, 1,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    private void renderGBuffer(VkCommandBuffer cmd, int width, int height, int activeSections, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

        // Prepare RT0, RT1, SharedDepth
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

        transitionImageLayout(cmd, vkInteropManager.getSharedDepthImage(),
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
                .imageView(vkInteropManager.getSharedDepthImageView())
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

        vkCmdDrawIndexedIndirectCountKHR(
                cmd,
                vkMegaBuffer.getIndirectCmdBuffer().buffer,
                0,
                vkMegaBuffer.getDrawCountBuffer().buffer,
                0,
                activeSections,
                20
        );

        vkCmdEndRendering(cmd);

        // Transition G-Buffer outputs to SHADER_READ_ONLY_OPTIMAL for lighting compute pass
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

        transitionImageLayout(cmd, vkInteropManager.getSharedDepthImage(),
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
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

    private void renderPostComposite(VkCommandBuffer cmd, int width, int height, MemoryStack stack) {
        long pipeLayout = vkPipelineManager.getPipelineLayout();
        LongBuffer pSets = stack.longs(vkPipelineManager.getDescriptorSet0(), vkPipelineManager.getDescriptorSet1());

        transitionImageLayout(cmd, vkInteropManager.getSharedColorImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipelineManager.getPostCompositePipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, pSets, null);

        ByteBuffer pcPost = stack.malloc(8);
        pcPost.putFloat(0, 1.0f); // exposure
        pcPost.putInt(4, 1);      // FXAA enabled
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcPost);

        int groupsX = (width + 7) / 8;
        int groupsY = (height + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Transition Shared Color & Depth to TRANSFER_SRC_OPTIMAL for OpenGL zero-copy Blit
        transitionImageLayout(cmd, vkInteropManager.getSharedColorImage(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        transitionImageLayout(cmd, vkInteropManager.getSharedDepthImage(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
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

        // Also transition shared depth buffer to valid transfer state
        transitionImageLayout(cmd, vkInteropManager.getSharedDepthImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    private void computeCsmMatrices(Vector3f cameraPos, Vector3f sunDir, Matrix4f viewMatrix) {
        float[] splits = { 32.0f, 96.0f, 192.0f };
        for (int i = 0; i < VkPipelineManager.CSM_CASCADE_COUNT; i++) {
            float dist = splits[i];
            Vector3f lightPos = new Vector3f(cameraPos).add(new Vector3f(sunDir).mul(dist * 0.5f));
            Matrix4f lightView = new Matrix4f().lookAt(
                    lightPos.x, lightPos.y, lightPos.z,
                    cameraPos.x, cameraPos.y, cameraPos.z,
                    0f, 1f, 0f
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
            vkInteropManager.compositeToMainTarget(mainTarget);
        }
    }

    /**
     * Schedules an asynchronous Greedy Meshing task for the given chunk section.
     */
    public void scheduleSectionRebuild(int sectionX, int sectionY, int sectionZ) {
        ClientLevel level = Minecraft.getInstance().level;
        if (level == null || greedyMesher == null) {
            return;
        }

        GreedyMesher.SectionSnapshot snapshot = GreedyMesher.SectionSnapshot.capture(level, sectionX, sectionY, sectionZ);
        greedyMesher.enqueue(snapshot);
    }

    public synchronized void onWorldReload() {
        LOGGER.info("[Valkyrie] World reload event triggered.");
        if (vkMegaBuffer != null) {
            vkMegaBuffer.clearAllSections();
        }
        if (greedyMesher != null) {
            greedyMesher.clear();
        }
        if (atlasUVTable != null) {
            atlasUVTable.rebuildFromBlockAtlas();
        }
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
    }

    public synchronized void shutdown() {
        if (!initialized) return;

        LOGGER.info("[Valkyrie] Shutting down ValkyrieRenderer...");

        if (vkContext != null && vkContext.isInitialized()) {
            VkDevice device = vkContext.getVkDevice();
            vkDeviceWaitIdle(device);

            if (frameFence != VK_NULL_HANDLE) {
                vkDestroyFence(device, frameFence, null);
                frameFence = VK_NULL_HANDLE;
            }

            if (commandBuffer != null) {
                vkContext.freeCommandBuffer(commandBuffer);
                commandBuffer = null;
            }
        }

        if (greedyMesher != null) {
            greedyMesher.shutdown();
        }

        if (atlasUVTable != null) {
            atlasUVTable.destroy();
        }

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
