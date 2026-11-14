package com.example.valkyrie.vulkan;

import com.example.valkyrie.config.ValkyrieConfig;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK12.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
import static org.lwjgl.vulkan.VK13.*;

/**
 * Manages Vulkan 1.3 Descriptor Sets, Graphics/Compute Pipeline Layouts, Pipelines,
 * Frame Uniform Buffer (UBO), internal G-Buffer attachments (RT0, RT1, Hi-Z, CSM, HDR),
 * and Shadow Map 2-Tier Caching logic (SPEC.md Section 5.3).
 */
public final class VkPipelineManager implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(VkPipelineManager.class);
    private static VkPipelineManager instance;

    public static synchronized VkPipelineManager getInstance() {
        if (instance == null) {
            instance = new VkPipelineManager();
        }
        return instance;
    }

    // Vulkan Context reference
    private VkDevice device;
    private long vmaAllocator;

    // Descriptor Set Layouts
    private long descriptorSetLayout0; // Set 0: Scene/Geometry SSBOs, FrameUBO, Block Atlas
    private long descriptorSetLayout1; // Set 1: G-Buffer, Depth, Hi-Z, CSM, HDR, SharedColor images
    private long pipelineLayout;

    // Descriptor Pool & Sets
    private long descriptorPool;
    private long descriptorSet0;
    private long descriptorSet1;

    // Pipelines
    private long testGradientPipeline;
    private long chunkCullPipeline;
    private long hizDownsamplePipeline;
    private long gbufferPipeline;
    private long csmShadowPipeline;
    private long depthConvertPipeline;
    private long deferredLightingPipeline;
    private long postCompositePipeline;

    // Frame Uniform Buffer (std140, 1024 bytes Host-Visible Coherent)
    private long frameUboBuffer;
    private long frameUboAllocation;
    private long frameUboAddress;
    private final ByteBuffer frameUboHostBuffer = MemoryUtil.memAlloc(1024);

    // G-Buffer Internal Attachments
    private int width = 1;
    private int height = 1;

    // RT0: VK_FORMAT_R8G8B8A8_SRGB
    private long rt0Image;
    private long rt0Allocation;
    private long rt0View;

    // RT1: VK_FORMAT_R16G16_UINT
    private long rt1Image;
    private long rt1Allocation;
    private long rt1View;

    // Internal Reversed-Z G-Buffer Depth: VK_FORMAT_D32_SFLOAT
    private long gbufferDepthImage;
    private long gbufferDepthAllocation;
    private long gbufferDepthView;

    // Hi-Z: VK_FORMAT_R32_SFLOAT
    public static final int MAX_HIZ_MIPS = 16;
    private long hiZImage;
    private long hiZAllocation;
    private long hiZView;
    private final long[] hiZMipViews = new long[MAX_HIZ_MIPS];
    private int hiZMipLevels = 1;

    // CSM Array: VK_FORMAT_D16_UNORM (configurable resolution x 3 layers)
    public static final int CSM_RESOLUTION = 2048;
    public static final int CSM_CASCADE_COUNT = 3;
    private int csmResolution = 1024;
    private long csmArrayImage;
    private long csmArrayAllocation;
    private long csmArrayView;
    private final long[] csmLayerViews = new long[CSM_CASCADE_COUNT];

    // HDR Color Image: VK_FORMAT_R16G16B16A16_SFLOAT
    private long hdrColorImage;
    private long hdrColorAllocation;
    private long hdrColorView;

    // Samplers
    private long nearestSampler;
    private long linearSampler;
    private long shadowSampler;
    private long atlasSampler;

    // Block atlas (dummy fallback + uploaded MC block atlas)
    private long dummyAtlasImage;
    private long dummyAtlasAllocation;
    private long dummyAtlasView;

    private long blockAtlasImage;
    private long blockAtlasAllocation;
    private long blockAtlasView;

    private boolean initialized = false;

    private VkPipelineManager() {
    }

    public synchronized void init(int initialWidth, int initialHeight, long sharedColorView, long sharedDepthView) {
        if (initialized) return;

        ValkyrieConfig config = ValkyrieConfig.get();
        config.loadOrCreate();

        VkContext ctx = VkContext.getInstance();
        this.device = ctx.getVkDevice();
        this.vmaAllocator = ctx.getVmaAllocator();
        this.width = Math.max(1, initialWidth);
        this.height = Math.max(1, initialHeight);
        this.csmResolution = config.getCsmResolution();

        LOGGER.info("[Valkyrie] Initializing VkPipelineManager (width={}, height={}, csmRes={})...", width, height, csmResolution);

        createSamplers();
        createDummyAtlas();
        createFrameUbo();
        createDescriptorSetLayouts();
        createPipelineLayout();
        createDescriptorPoolAndSets();
        createComputePipelines();
        createGraphicsPipelines();
        createCsmResources();
        recreateSizeDependentResources(width, height, sharedColorView, sharedDepthView);

        initialized = true;
        LOGGER.info("[Valkyrie] VkPipelineManager successfully initialized.");
    }

    // ------------------------------------------------------------------------
    // Shadow Map 2-Tier Caching (SPEC.md Section 5.3)
    // ------------------------------------------------------------------------

    /**
     * Cascade 0 (0..32 blocks): Updated every frame when CSM is enabled.
     * Cascade 1 & 2: Staggered across frames based on csmUpdateInterval, avoiding
     * rendering all outer cascades simultaneously during chunk streaming.
     */
    public boolean shouldUpdateCascade(int cascadeIndex, long frameIndex, boolean dirtyShadow) {
        ValkyrieConfig config = ValkyrieConfig.get();
        if (!config.isCsmEnabled() || cascadeIndex >= config.getCsmCascades()) {
            return false;
        }
        if (cascadeIndex == 0) {
            return true;
        }
        int interval = Math.max(2, config.getCsmUpdateInterval());
        int phase = (cascadeIndex == 1) ? (interval / 3) : ((interval * 2) / 3);
        if (frameIndex % interval == phase) {
            return true;
        }
        return dirtyShadow && (frameIndex % 3 == cascadeIndex);
    }

    public int getCsmResolution() {
        return csmResolution;
    }

    // ------------------------------------------------------------------------
    // Frame UBO Update
    // ------------------------------------------------------------------------

    public void updateFrameUbo(
            Matrix4f view,
            Matrix4f proj,
            Matrix4f viewProj,
            Matrix4f invViewProj,
            Matrix4f prevViewProj,
            Matrix4f[] csmMatrices,
            Vector3f sunDir,
            Vector3f cameraPos,
            int viewportWidth,
            int viewportHeight,
            long frameIndex,
            float time
    ) {
        if (frameUboAddress == MemoryUtil.NULL) return;

        frameUboHostBuffer.clear();

        // mat4 view (0..63)
        view.get(0, frameUboHostBuffer);
        // mat4 proj (64..127)
        proj.get(64, frameUboHostBuffer);
        // mat4 viewProj (128..191)
        viewProj.get(128, frameUboHostBuffer);
        // mat4 invViewProj (192..255)
        invViewProj.get(192, frameUboHostBuffer);
        // mat4 prevViewProj (256..319)
        prevViewProj.get(256, frameUboHostBuffer);

        // mat4 csmMatrices[3] (320..511)
        for (int i = 0; i < 3; i++) {
            if (csmMatrices != null && i < csmMatrices.length && csmMatrices[i] != null) {
                csmMatrices[i].get(320 + i * 64, frameUboHostBuffer);
            } else {
                new Matrix4f().identity().get(320 + i * 64, frameUboHostBuffer);
            }
        }

        // vec4 sunDir (512..527)
        frameUboHostBuffer.putFloat(512, sunDir != null ? sunDir.x : 0f);
        frameUboHostBuffer.putFloat(516, sunDir != null ? sunDir.y : 1f);
        frameUboHostBuffer.putFloat(520, sunDir != null ? sunDir.z : 0f);
        frameUboHostBuffer.putFloat(524, 0f);

        // vec4 cameraPos (528..543)
        frameUboHostBuffer.putFloat(528, cameraPos != null ? cameraPos.x : 0f);
        frameUboHostBuffer.putFloat(532, cameraPos != null ? cameraPos.y : 0f);
        frameUboHostBuffer.putFloat(536, cameraPos != null ? cameraPos.z : 0f);
        frameUboHostBuffer.putFloat(540, 1f);

        // vec2 viewportSize (544..551)
        frameUboHostBuffer.putFloat(544, (float) viewportWidth);
        frameUboHostBuffer.putFloat(548, (float) viewportHeight);

        // uint frameIndex (552..555)
        frameUboHostBuffer.putInt(552, (int) frameIndex);

        // float time (556..559)
        frameUboHostBuffer.putFloat(556, time);

        // Runtime lighting/culling flags & shadow range (560..575)
        ValkyrieConfig config = ValkyrieConfig.get();
        int renderFlags = 0;
        if (config.isCsmEnabled()) renderFlags |= 1;
        if (config.isFastShadowPcf()) renderFlags |= 2;
        if (config.isSsrEnabled()) renderFlags |= 4;
        frameUboHostBuffer.putInt(560, renderFlags);
        frameUboHostBuffer.putInt(564, config.getCsmCascades());
        frameUboHostBuffer.putFloat(568, config.getMaxShadowDistance());
        frameUboHostBuffer.putFloat(572, 0f);

        // Copy host bytes directly into mapped Vulkan buffer
        MemoryUtil.memCopy(MemoryUtil.memAddress(frameUboHostBuffer), frameUboAddress, 576);
    }

    // ------------------------------------------------------------------------
    // Descriptor Set Layouts & Pipeline Layout
    // ------------------------------------------------------------------------

    private void createDescriptorSetLayouts() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Set 0 Bindings
            VkDescriptorSetLayoutBinding.Buffer bindings0 = VkDescriptorSetLayoutBinding.calloc(8, stack);

            // Binding 0: FrameUBO (UNIFORM_BUFFER, stage vertex|fragment|compute)
            bindings0.get(0)
                    .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            // Binding 1: MegaVertexSSBO (STORAGE_BUFFER, stage vertex|compute)
            bindings0.get(1)
                    .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            // Binding 2: ChunkMetaSSBO (STORAGE_BUFFER, stage vertex|compute)
            bindings0.get(2)
                    .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            // Binding 3: IndirectCmdSSBO (STORAGE_BUFFER, stage compute)
            bindings0.get(3)
                    .binding(3)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);

            // Binding 4: DrawCountSSBO (STORAGE_BUFFER, stage compute)
            bindings0.get(4)
                    .binding(4)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);

            // Binding 5: VisibleChunkIdSSBO (STORAGE_BUFFER, stage vertex|compute)
            bindings0.get(5)
                    .binding(5)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            // Binding 6: AtlasUVSSBO (STORAGE_BUFFER, stage fragment|compute)
            bindings0.get(6)
                    .binding(6)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            // Binding 7: uBlockAtlas (COMBINED_IMAGE_SAMPLER, stage fragment|compute)
            bindings0.get(7)
                    .binding(7)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo layoutInfo0 = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                    .pBindings(bindings0);

            LongBuffer pLayout0 = stack.mallocLong(1);
            int res = vkCreateDescriptorSetLayout(device, layoutInfo0, null, pLayout0);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create descriptor set layout 0: " + res);
            descriptorSetLayout0 = pLayout0.get(0);

            // Set 1 Bindings
            VkDescriptorSetLayoutBinding.Buffer bindings1 = VkDescriptorSetLayoutBinding.calloc(8, stack);

            // Binding 0: uGBufferRT0 (COMBINED_IMAGE_SAMPLER, compute)
            bindings1.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            // Binding 1: uGBufferRT1 (COMBINED_IMAGE_SAMPLER, compute)
            bindings1.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            // Binding 2: uDepthBuffer (COMBINED_IMAGE_SAMPLER, compute | fragment for depth conversion)
            bindings1.get(2).binding(2).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            // Binding 3: uHiZPyramid (COMBINED_IMAGE_SAMPLER, compute)
            bindings1.get(3).binding(3).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            // Binding 4: uShadowMapArray (COMBINED_IMAGE_SAMPLER, compute)
            bindings1.get(4).binding(4).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            // Binding 5: uHdrColorImg (STORAGE_IMAGE, compute)
            bindings1.get(5).binding(5).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            // Binding 6: uSharedColorImg (STORAGE_IMAGE, compute)
            bindings1.get(6).binding(6).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            // Binding 7: uHiZMips[16] (STORAGE_IMAGE array, compute)
            bindings1.get(7).binding(7).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(MAX_HIZ_MIPS).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo layoutInfo1 = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                    .pBindings(bindings1);

            LongBuffer pLayout1 = stack.mallocLong(1);
            res = vkCreateDescriptorSetLayout(device, layoutInfo1, null, pLayout1);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create descriptor set layout 1: " + res);
            descriptorSetLayout1 = pLayout1.get(0);
        }
    }

    private void createPipelineLayout() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer pSetLayouts = stack.longs(descriptorSetLayout0, descriptorSetLayout1);

            // Push Constant Range: 16 bytes for vertex and compute stages
            VkPushConstantRange.Buffer pushConstants = VkPushConstantRange.calloc(1, stack)
                    .stageFlags(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0)
                    .size(16);

            VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                    .pSetLayouts(pSetLayouts)
                    .pPushConstantRanges(pushConstants);

            LongBuffer pPipelineLayout = stack.mallocLong(1);
            int res = vkCreatePipelineLayout(device, layoutInfo, null, pPipelineLayout);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create pipeline layout: " + res);
            pipelineLayout = pPipelineLayout.get(0);
        }
    }

    private void createDescriptorPoolAndSets() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(4, stack);
            poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).descriptorCount(4);
            poolSizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(16);
            poolSizes.get(2).type(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(16);
            poolSizes.get(3).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(32);

            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                    .flags(VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT)
                    .maxSets(16)
                    .pPoolSizes(poolSizes);

            LongBuffer pPool = stack.mallocLong(1);
            int res = vkCreateDescriptorPool(device, poolInfo, null, pPool);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create descriptor pool: " + res);
            descriptorPool = pPool.get(0);

            // Allocate Set 0 and Set 1
            LongBuffer pLayouts = stack.longs(descriptorSetLayout0, descriptorSetLayout1);
            VkDescriptorSetAllocateInfo allocInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                    .descriptorPool(descriptorPool)
                    .pSetLayouts(pLayouts);

            LongBuffer pSets = stack.mallocLong(2);
            res = vkAllocateDescriptorSets(device, allocInfo, pSets);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to allocate descriptor sets: " + res);
            descriptorSet0 = pSets.get(0);
            descriptorSet1 = pSets.get(1);
        }
    }

    // ------------------------------------------------------------------------
    // Pipelines Creation
    // ------------------------------------------------------------------------

    private void createComputePipelines() {
        testGradientPipeline = createComputePipeline("/assets/valkyrie/shaders/test_gradient.comp");
        chunkCullPipeline = createComputePipeline("/assets/valkyrie/shaders/chunk_cull.comp");
        hizDownsamplePipeline = createComputePipeline("/assets/valkyrie/shaders/hiz_downsample.comp");
        deferredLightingPipeline = createComputePipeline("/assets/valkyrie/shaders/deferred_lighting.comp");
        postCompositePipeline = createComputePipeline("/assets/valkyrie/shaders/post_composite.comp");
    }

    private long createComputePipeline(String shaderResource) {
        long shaderModule = VkShaderCompiler.compileResourceToModule(device, shaderResource, VkShaderCompiler.STAGE_COMPUTE);
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPipelineShaderStageCreateInfo stageInfo = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                    .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(shaderModule)
                    .pName(stack.UTF8("main"));

            VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO)
                    .layout(pipelineLayout)
                    .stage(stageInfo);

            LongBuffer pPipeline = stack.mallocLong(1);
            int res = vkCreateComputePipelines(device, VK_NULL_HANDLE, pipelineInfo, null, pPipeline);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create compute pipeline (" + shaderResource + "): " + res);
            return pPipeline.get(0);
        } finally {
            vkDestroyShaderModule(device, shaderModule, null);
        }
    }

    private void createGraphicsPipelines() {
        createGBufferPipeline();
        createCsmShadowPipeline();
        createDepthConvertPipeline();
    }

    private void createGBufferPipeline() {
        long vertModule = VkShaderCompiler.compileResourceToModule(device, "/assets/valkyrie/shaders/gbuffer.vert", VkShaderCompiler.STAGE_VERTEX);
        long fragModule = VkShaderCompiler.compileResourceToModule(device, "/assets/valkyrie/shaders/gbuffer.frag", VkShaderCompiler.STAGE_FRAGMENT);

        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(2, stack);
            stages.get(0).sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO).stage(VK_SHADER_STAGE_VERTEX_BIT).module(vertModule).pName(stack.UTF8("main"));
            stages.get(1).sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO).stage(VK_SHADER_STAGE_FRAGMENT_BIT).module(fragModule).pName(stack.UTF8("main"));

            // Vertex Input: empty (pulling from SSBO via gl_VertexIndex)
            VkPipelineVertexInputStateCreateInfo vertexInput = VkPipelineVertexInputStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);

            VkPipelineInputAssemblyStateCreateInfo inputAssembly = VkPipelineInputAssemblyStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
                    .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                    .primitiveRestartEnable(false);

            VkPipelineViewportStateCreateInfo viewportState = VkPipelineViewportStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
                    .viewportCount(1)
                    .scissorCount(1);

            VkPipelineRasterizationStateCreateInfo rasterizer = VkPipelineRasterizationStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
                    .polygonMode(VK_POLYGON_MODE_FILL)
                    .cullMode(VK_CULL_MODE_BACK_BIT)
                    // Y-up projection (m11 > 0) + positive viewport flips Vulkan framebuffer signed area parity
                    .frontFace(VK_FRONT_FACE_CLOCKWISE)
                    .lineWidth(1.0f);

            VkPipelineMultisampleStateCreateInfo multisampling = VkPipelineMultisampleStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
                    .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT);

            VkPipelineDepthStencilStateCreateInfo depthStencil = VkPipelineDepthStencilStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO)
                    .depthTestEnable(true)
                    .depthWriteEnable(true)
                    .depthCompareOp(VK_COMPARE_OP_GREATER_OR_EQUAL) // Reverse-Z
                    .depthBoundsTestEnable(false)
                    .stencilTestEnable(false);

            // 2 Color Attachments: RT0 (R8G8B8A8_SRGB), RT1 (R16G16_UINT)
            VkPipelineColorBlendAttachmentState.Buffer blendAttachments = VkPipelineColorBlendAttachmentState.calloc(2, stack);
            for (int i = 0; i < 2; i++) {
                blendAttachments.get(i)
                        .blendEnable(false)
                        .colorWriteMask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
            }

            VkPipelineColorBlendStateCreateInfo colorBlend = VkPipelineColorBlendStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO)
                    .pAttachments(blendAttachments);

            IntBuffer dynamicStates = stack.ints(VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR);
            VkPipelineDynamicStateCreateInfo dynamicState = VkPipelineDynamicStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO)
                    .pDynamicStates(dynamicStates);

            // Vulkan 1.3 Dynamic Rendering: VkPipelineRenderingCreateInfo
            IntBuffer colorFormats = stack.ints(VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R16G16_UINT);
            VkPipelineRenderingCreateInfo renderingCreateInfo = VkPipelineRenderingCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
                    .pColorAttachmentFormats(colorFormats)
                    .depthAttachmentFormat(VK_FORMAT_D32_SFLOAT);

            VkGraphicsPipelineCreateInfo.Buffer pipelineInfo = VkGraphicsPipelineCreateInfo.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
                    .pNext(renderingCreateInfo.address())
                    .pStages(stages)
                    .pVertexInputState(vertexInput)
                    .pInputAssemblyState(inputAssembly)
                    .pViewportState(viewportState)
                    .pRasterizationState(rasterizer)
                    .pMultisampleState(multisampling)
                    .pDepthStencilState(depthStencil)
                    .pColorBlendState(colorBlend)
                    .pDynamicState(dynamicState)
                    .layout(pipelineLayout);

            LongBuffer pPipeline = stack.mallocLong(1);
            int res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, pipelineInfo, null, pPipeline);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create G-Buffer graphics pipeline: " + res);
            gbufferPipeline = pPipeline.get(0);
        } finally {
            vkDestroyShaderModule(device, vertModule, null);
            vkDestroyShaderModule(device, fragModule, null);
        }
    }

    private void createCsmShadowPipeline() {
        long vertModule = VkShaderCompiler.compileResourceToModule(device, "/assets/valkyrie/shaders/csm_shadow.vert", VkShaderCompiler.STAGE_VERTEX);

        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(1, stack);
            stages.get(0).sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO).stage(VK_SHADER_STAGE_VERTEX_BIT).module(vertModule).pName(stack.UTF8("main"));

            VkPipelineVertexInputStateCreateInfo vertexInput = VkPipelineVertexInputStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);

            VkPipelineInputAssemblyStateCreateInfo inputAssembly = VkPipelineInputAssemblyStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
                    .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                    .primitiveRestartEnable(false);

            VkPipelineViewportStateCreateInfo viewportState = VkPipelineViewportStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
                    .viewportCount(1)
                    .scissorCount(1);

            VkPipelineRasterizationStateCreateInfo rasterizer = VkPipelineRasterizationStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
                    .polygonMode(VK_POLYGON_MODE_FILL)
                    .cullMode(VK_CULL_MODE_NONE)
                    .depthClampEnable(true)
                    .depthBiasEnable(true)
                    .lineWidth(1.0f);

            VkPipelineMultisampleStateCreateInfo multisampling = VkPipelineMultisampleStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
                    .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT);

            VkPipelineDepthStencilStateCreateInfo depthStencil = VkPipelineDepthStencilStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO)
                    .depthTestEnable(true)
                    .depthWriteEnable(true)
                    .depthCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL)
                    .depthBoundsTestEnable(false)
                    .stencilTestEnable(false);

            IntBuffer dynamicStates = stack.ints(VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS);
            VkPipelineDynamicStateCreateInfo dynamicState = VkPipelineDynamicStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO)
                    .pDynamicStates(dynamicStates);

            // Depth-only Dynamic Rendering: VK_FORMAT_D16_UNORM
            VkPipelineRenderingCreateInfo renderingCreateInfo = VkPipelineRenderingCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
                    .depthAttachmentFormat(VK_FORMAT_D16_UNORM);

            VkGraphicsPipelineCreateInfo.Buffer pipelineInfo = VkGraphicsPipelineCreateInfo.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
                    .pNext(renderingCreateInfo.address())
                    .pStages(stages)
                    .pVertexInputState(vertexInput)
                    .pInputAssemblyState(inputAssembly)
                    .pViewportState(viewportState)
                    .pRasterizationState(rasterizer)
                    .pMultisampleState(multisampling)
                    .pDepthStencilState(depthStencil)
                    .pDynamicState(dynamicState)
                    .layout(pipelineLayout);

            LongBuffer pPipeline = stack.mallocLong(1);
            int res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, pipelineInfo, null, pPipeline);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create CSM graphics pipeline: " + res);
            csmShadowPipeline = pPipeline.get(0);
        } finally {
            vkDestroyShaderModule(device, vertModule, null);
        }
    }

    private void createDepthConvertPipeline() {
        long vertModule = VkShaderCompiler.compileResourceToModule(device, "/assets/valkyrie/shaders/depth_convert.vert", VkShaderCompiler.STAGE_VERTEX);
        long fragModule = VkShaderCompiler.compileResourceToModule(device, "/assets/valkyrie/shaders/depth_convert.frag", VkShaderCompiler.STAGE_FRAGMENT);

        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(2, stack);
            stages.get(0).sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO).stage(VK_SHADER_STAGE_VERTEX_BIT).module(vertModule).pName(stack.UTF8("main"));
            stages.get(1).sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO).stage(VK_SHADER_STAGE_FRAGMENT_BIT).module(fragModule).pName(stack.UTF8("main"));

            VkPipelineVertexInputStateCreateInfo vertexInput = VkPipelineVertexInputStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);

            VkPipelineInputAssemblyStateCreateInfo inputAssembly = VkPipelineInputAssemblyStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
                    .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                    .primitiveRestartEnable(false);

            VkPipelineViewportStateCreateInfo viewportState = VkPipelineViewportStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
                    .viewportCount(1)
                    .scissorCount(1);

            VkPipelineRasterizationStateCreateInfo rasterizer = VkPipelineRasterizationStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
                    .polygonMode(VK_POLYGON_MODE_FILL)
                    .cullMode(VK_CULL_MODE_NONE)
                    .lineWidth(1.0f);

            VkPipelineMultisampleStateCreateInfo multisampling = VkPipelineMultisampleStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
                    .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT);

            VkPipelineDepthStencilStateCreateInfo depthStencil = VkPipelineDepthStencilStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO)
                    .depthTestEnable(true)
                    .depthWriteEnable(true)
                    .depthCompareOp(VK_COMPARE_OP_ALWAYS)
                    .depthBoundsTestEnable(false)
                    .stencilTestEnable(false);

            IntBuffer dynamicStates = stack.ints(VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR);
            VkPipelineDynamicStateCreateInfo dynamicState = VkPipelineDynamicStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO)
                    .pDynamicStates(dynamicStates);

            VkPipelineRenderingCreateInfo renderingCreateInfo = VkPipelineRenderingCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
                    .depthAttachmentFormat(VK_FORMAT_D32_SFLOAT);

            VkGraphicsPipelineCreateInfo.Buffer pipelineInfo = VkGraphicsPipelineCreateInfo.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
                    .pNext(renderingCreateInfo.address())
                    .pStages(stages)
                    .pVertexInputState(vertexInput)
                    .pInputAssemblyState(inputAssembly)
                    .pViewportState(viewportState)
                    .pRasterizationState(rasterizer)
                    .pMultisampleState(multisampling)
                    .pDepthStencilState(depthStencil)
                    .pDynamicState(dynamicState)
                    .layout(pipelineLayout);

            LongBuffer pPipeline = stack.mallocLong(1);
            int res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, pipelineInfo, null, pPipeline);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create depth convert graphics pipeline: " + res);
            depthConvertPipeline = pPipeline.get(0);
        } finally {
            vkDestroyShaderModule(device, vertModule, null);
            vkDestroyShaderModule(device, fragModule, null);
        }
    }

    // ------------------------------------------------------------------------
    // Resource Allocation & Resizing
    // ------------------------------------------------------------------------

    private void createSamplers() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Nearest Clamp
            VkSamplerCreateInfo nearestInfo = VkSamplerCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                    .magFilter(VK_FILTER_NEAREST)
                    .minFilter(VK_FILTER_NEAREST)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

            LongBuffer pSampler = stack.mallocLong(1);
            vkCreateSampler(device, nearestInfo, null, pSampler);
            nearestSampler = pSampler.get(0);

            // Linear Clamp
            VkSamplerCreateInfo linearInfo = VkSamplerCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                    .magFilter(VK_FILTER_LINEAR)
                    .minFilter(VK_FILTER_LINEAR)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

            vkCreateSampler(device, linearInfo, null, pSampler);
            linearSampler = pSampler.get(0);

            // Shadow comparison sampler
            VkSamplerCreateInfo shadowInfo = VkSamplerCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                    .magFilter(VK_FILTER_LINEAR)
                    .minFilter(VK_FILTER_LINEAR)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .compareEnable(true)
                    .compareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

            vkCreateSampler(device, shadowInfo, null, pSampler);
            shadowSampler = pSampler.get(0);

            // Block Atlas Sampler (nearest texel + linear mipmap LOD for crisp pixel textures without shimmer)
            VkSamplerCreateInfo atlasInfo = VkSamplerCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                    .magFilter(VK_FILTER_NEAREST)
                    .minFilter(VK_FILTER_NEAREST)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_LINEAR)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(16.0f);

            vkCreateSampler(device, atlasInfo, null, pSampler);
            atlasSampler = pSampler.get(0);
        }
    }

    private void createFrameUbo() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(1024)
                    .usage(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

            VmaAllocationCreateInfo allocInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_AUTO)
                    .flags(Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | Vma.VMA_ALLOCATION_CREATE_MAPPED_BIT);

            LongBuffer pBuffer = stack.mallocLong(1);
            PointerBuffer pAlloc = stack.mallocPointer(1);
            int res = Vma.vmaCreateBuffer(vmaAllocator, bufferInfo, allocInfo, pBuffer, pAlloc, null);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to allocate FrameUBO: " + res);
            frameUboBuffer = pBuffer.get(0);
            frameUboAllocation = pAlloc.get(0);

            PointerBuffer pData = stack.mallocPointer(1);
            res = Vma.vmaMapMemory(vmaAllocator, frameUboAllocation, pData);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to map FrameUBO: " + res);
            frameUboAddress = pData.get(0);
        }
    }

    private void createCsmResources() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer pAlloc = stack.mallocLong(1);
            csmArrayImage = createImage(
                    csmResolution, csmResolution, VK_FORMAT_D16_UNORM,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    1, CSM_CASCADE_COUNT, pAlloc
            );
            csmArrayAllocation = pAlloc.get(0);

            csmArrayView = createImageView(
                    csmArrayImage, VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT,
                    VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, 1, 0, CSM_CASCADE_COUNT
            );

            for (int i = 0; i < CSM_CASCADE_COUNT; i++) {
                csmLayerViews[i] = createImageView(
                        csmArrayImage, VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT,
                        VK_IMAGE_VIEW_TYPE_2D, 0, 1, i, 1
                );
            }

            // Transition CSM array to SHADER_READ_ONLY_OPTIMAL initially so DescriptorSet1 is always valid
            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            recordImageBarrier(cmd, csmArrayImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, CSM_CASCADE_COUNT,
                    0, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stack);
            ctx.endSingleTimeCommands(cmd);
        }
    }

    private void createDummyAtlas() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer pAlloc = stack.mallocLong(1);
            dummyAtlasImage = createImage(
                    1, 1, VK_FORMAT_R8G8B8A8_SRGB,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    1, 1, pAlloc
            );
            dummyAtlasAllocation = pAlloc.get(0);

            dummyAtlasView = createImageView(
                    dummyAtlasImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1
            );

            // Clear 1x1 fallback atlas to opaque white and transition to SHADER_READ_ONLY_OPTIMAL
            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            recordImageBarrier(cmd, dummyAtlasImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

            VkClearColorValue clearWhite = VkClearColorValue.calloc(stack);
            clearWhite.float32(0, 1f).float32(1, 1f).float32(2, 1f).float32(3, 1f);
            VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack)
                    .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            vkCmdClearColorImage(cmd, dummyAtlasImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clearWhite, range);

            recordImageBarrier(cmd, dummyAtlasImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, stack);
            ctx.endSingleTimeCommands(cmd);
        }
    }

    /**
     * Uploads RGBA8 pixels of Minecraft's Block Atlas into a mipmapped Vulkan VkImage.
     */
    public synchronized void uploadBlockAtlas(int atlasWidth, int atlasHeight, ByteBuffer rgbaPixels) {
        if (!initialized || atlasWidth <= 0 || atlasHeight <= 0 || rgbaPixels == null) {
            return;
        }

        vkDeviceWaitIdle(device);

        if (blockAtlasView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, blockAtlasView, null);
            blockAtlasView = VK_NULL_HANDLE;
        }
        if (blockAtlasImage != VK_NULL_HANDLE) {
            Vma.vmaDestroyImage(vmaAllocator, blockAtlasImage, blockAtlasAllocation);
            blockAtlasImage = VK_NULL_HANDLE;
            blockAtlasAllocation = VK_NULL_HANDLE;
        }

        int mipLevels = Math.max(1, 1 + (int) Math.floor(Math.log(Math.max(atlasWidth, atlasHeight)) / Math.log(2)));
        // Limit mip chain to 5 levels (down to 16x16 per tile) to avoid excessive cross-tile bleeding
        mipLevels = Math.min(mipLevels, 5);

        long pixelBytes = (long) atlasWidth * atlasHeight * 4L;

        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Create temporary staging buffer
            VkBufferCreateInfo bufInfo = VkBufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(pixelBytes)
                    .usage(VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            VmaAllocationCreateInfo bufAllocInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_CPU_TO_GPU)
                    .flags(Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | Vma.VMA_ALLOCATION_CREATE_MAPPED_BIT);

            LongBuffer pBuf = stack.mallocLong(1);
            PointerBuffer pBufAlloc = stack.mallocPointer(1);
            int res = Vma.vmaCreateBuffer(vmaAllocator, bufInfo, bufAllocInfo, pBuf, pBufAlloc, null);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create atlas staging buffer: " + res);
            long stagingBuf = pBuf.get(0);
            long stagingAlloc = pBufAlloc.get(0);

            PointerBuffer pMapped = stack.mallocPointer(1);
            Vma.vmaMapMemory(vmaAllocator, stagingAlloc, pMapped);
            ByteBuffer src = rgbaPixels.duplicate();
            src.position(0);
            src.limit((int) pixelBytes);
            MemoryUtil.memByteBuffer(pMapped.get(0), (int) pixelBytes).put(src);
            Vma.vmaUnmapMemory(vmaAllocator, stagingAlloc);

            LongBuffer pImgAlloc = stack.mallocLong(1);
            blockAtlasImage = createImage(
                    atlasWidth, atlasHeight, VK_FORMAT_R8G8B8A8_SRGB,
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    mipLevels, 1, pImgAlloc
            );
            blockAtlasAllocation = pImgAlloc.get(0);
            blockAtlasView = createImageView(
                    blockAtlasImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_VIEW_TYPE_2D, 0, mipLevels, 0, 1
            );

            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

            recordImageBarrier(cmd, blockAtlasImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

            VkBufferImageCopy.Buffer copyRegion = VkBufferImageCopy.calloc(1, stack)
                    .bufferOffset(0)
                    .bufferRowLength(0)
                    .bufferImageHeight(0);
            copyRegion.imageSubresource()
                    .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                    .mipLevel(0)
                    .baseArrayLayer(0)
                    .layerCount(1);
            copyRegion.imageExtent().set(atlasWidth, atlasHeight, 1);
            vkCmdCopyBufferToImage(cmd, stagingBuf, blockAtlasImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copyRegion);

            // Generate mipmaps via vkCmdBlitImage
            int mipW = atlasWidth;
            int mipH = atlasHeight;
            for (int i = 1; i < mipLevels; i++) {
                recordImageBarrier(cmd, blockAtlasImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

                int nextW = Math.max(1, mipW / 2);
                int nextH = Math.max(1, mipH / 2);

                VkImageBlit.Buffer blit = VkImageBlit.calloc(1, stack);
                blit.srcOffsets(0).set(0, 0, 0);
                blit.srcOffsets(1).set(mipW, mipH, 1);
                blit.srcSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(i - 1).baseArrayLayer(0).layerCount(1);

                blit.dstOffsets(0).set(0, 0, 0);
                blit.dstOffsets(1).set(nextW, nextH, 1);
                blit.dstSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(i).baseArrayLayer(0).layerCount(1);

                vkCmdBlitImage(cmd, blockAtlasImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        blockAtlasImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blit, VK_FILTER_LINEAR);

                recordImageBarrier(cmd, blockAtlasImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1,
                        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, stack);

                mipW = nextW;
                mipH = nextH;
            }

            // Transition final mip to SHADER_READ_ONLY_OPTIMAL
            recordImageBarrier(cmd, blockAtlasImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, mipLevels - 1, 1, 0, 1,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, stack);

            ctx.endSingleTimeCommands(cmd);
            Vma.vmaDestroyBuffer(vmaAllocator, stagingBuf, stagingAlloc);
            LOGGER.info("[Valkyrie] Uploaded Minecraft Block Atlas ({}x{}, {} mip levels) to Vulkan.", atlasWidth, atlasHeight, mipLevels);
        }
    }

    public synchronized void recreateSizeDependentResources(int newWidth, int newHeight, long sharedColorView, long sharedDepthView) {
        this.width = Math.max(1, newWidth);
        this.height = Math.max(1, newHeight);

        destroySizeDependentResources();

        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer pAlloc = stack.mallocLong(1);

            // RT0: R8G8B8A8_SRGB
            rt0Image = createImage(width, height, VK_FORMAT_R8G8B8A8_SRGB,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 1, pAlloc);
            rt0Allocation = pAlloc.get(0);
            rt0View = createImageView(rt0Image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1);

            // RT1: R16G16_UINT
            rt1Image = createImage(width, height, VK_FORMAT_R16G16_UINT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 1, pAlloc);
            rt1Allocation = pAlloc.get(0);
            rt1View = createImageView(rt1Image, VK_FORMAT_R16G16_UINT, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1);

            // Internal Reversed-Z G-Buffer Depth: D32_SFLOAT
            gbufferDepthImage = createImage(width, height, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 1, pAlloc);
            gbufferDepthAllocation = pAlloc.get(0);
            gbufferDepthView = createImageView(gbufferDepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1);

            // Hi-Z: R32_SFLOAT
            hiZMipLevels = Math.min(MAX_HIZ_MIPS, Math.max(1, 1 + (int) Math.floor(Math.log(Math.max(width, height)) / Math.log(2))));
            hiZImage = createImage(width, height, VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    hiZMipLevels, 1, pAlloc);
            hiZAllocation = pAlloc.get(0);
            hiZView = createImageView(hiZImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 0, hiZMipLevels, 0, 1);
            for (int m = 0; m < hiZMipLevels; m++) {
                hiZMipViews[m] = createImageView(hiZImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, m, 1, 0, 1);
            }

            // Transition Hi-Z to SHADER_READ_ONLY_OPTIMAL and clear to 0.0 (Reversed-Z far) initially
            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            recordImageBarrier(cmd, hiZImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, hiZMipLevels, 0, 1,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);
            VkClearColorValue clearZero = VkClearColorValue.calloc(stack);
            VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack)
                    .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(hiZMipLevels).baseArrayLayer(0).layerCount(1);
            vkCmdClearColorImage(cmd, hiZImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clearZero, range);
            recordImageBarrier(cmd, hiZImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, hiZMipLevels, 0, 1,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stack);
            ctx.endSingleTimeCommands(cmd);

            // HDR Color: R16G16B16A16_SFLOAT
            hdrColorImage = createImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 1, pAlloc);
            hdrColorAllocation = pAlloc.get(0);
            hdrColorView = createImageView(hdrColorImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1);

            // Update Descriptor Set 1 (binding uDepthBuffer to gbufferDepthView!)
            updateDescriptorSet1(sharedColorView, gbufferDepthView);
        }
    }

    private void destroySizeDependentResources() {
        if (rt0View != VK_NULL_HANDLE) { vkDestroyImageView(device, rt0View, null); rt0View = VK_NULL_HANDLE; }
        if (rt0Image != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, rt0Image, rt0Allocation); rt0Image = VK_NULL_HANDLE; }

        if (rt1View != VK_NULL_HANDLE) { vkDestroyImageView(device, rt1View, null); rt1View = VK_NULL_HANDLE; }
        if (rt1Image != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, rt1Image, rt1Allocation); rt1Image = VK_NULL_HANDLE; }

        if (gbufferDepthView != VK_NULL_HANDLE) { vkDestroyImageView(device, gbufferDepthView, null); gbufferDepthView = VK_NULL_HANDLE; }
        if (gbufferDepthImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, gbufferDepthImage, gbufferDepthAllocation); gbufferDepthImage = VK_NULL_HANDLE; }

        for (int m = 0; m < MAX_HIZ_MIPS; m++) {
            if (hiZMipViews[m] != VK_NULL_HANDLE) {
                vkDestroyImageView(device, hiZMipViews[m], null);
                hiZMipViews[m] = VK_NULL_HANDLE;
            }
        }
        if (hiZView != VK_NULL_HANDLE) { vkDestroyImageView(device, hiZView, null); hiZView = VK_NULL_HANDLE; }
        if (hiZImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, hiZImage, hiZAllocation); hiZImage = VK_NULL_HANDLE; }

        if (hdrColorView != VK_NULL_HANDLE) { vkDestroyImageView(device, hdrColorView, null); hdrColorView = VK_NULL_HANDLE; }
        if (hdrColorImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, hdrColorImage, hdrColorAllocation); hdrColorImage = VK_NULL_HANDLE; }
    }

    // ------------------------------------------------------------------------
    // Descriptor Set Updates
    // ------------------------------------------------------------------------

    public void updateDescriptorSet0(
            long megaVertexBuffer,
            long chunkMetaBuffer,
            long indirectCmdBuffer,
            long drawCountBuffer,
            long visibleChunkIdBuffer,
            long atlasUvBuffer,
            long blockAtlasView
    ) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(8, stack);

            // 0: FrameUBO
            VkDescriptorBufferInfo.Buffer uboInfo = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(frameUboBuffer).offset(0).range(VK_WHOLE_SIZE);
            writes.get(0).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet0).dstBinding(0)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).pBufferInfo(uboInfo);

            // 1: MegaVertexSSBO
            VkDescriptorBufferInfo.Buffer vtxInfo = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(megaVertexBuffer).offset(0).range(VK_WHOLE_SIZE);
            writes.get(1).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet0).dstBinding(1)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(vtxInfo);

            // 2: ChunkMetaSSBO
            VkDescriptorBufferInfo.Buffer metaInfo = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(chunkMetaBuffer).offset(0).range(VK_WHOLE_SIZE);
            writes.get(2).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet0).dstBinding(2)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(metaInfo);

            // 3: IndirectCmdSSBO
            VkDescriptorBufferInfo.Buffer cmdInfo = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(indirectCmdBuffer).offset(0).range(VK_WHOLE_SIZE);
            writes.get(3).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet0).dstBinding(3)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(cmdInfo);

            // 4: DrawCountSSBO
            VkDescriptorBufferInfo.Buffer countInfo = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(drawCountBuffer).offset(0).range(VK_WHOLE_SIZE);
            writes.get(4).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet0).dstBinding(4)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(countInfo);

            // 5: VisibleChunkIdSSBO
            VkDescriptorBufferInfo.Buffer idInfo = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(visibleChunkIdBuffer).offset(0).range(VK_WHOLE_SIZE);
            writes.get(5).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet0).dstBinding(5)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(idInfo);

            // 6: AtlasUVSSBO
            VkDescriptorBufferInfo.Buffer uvInfo = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(atlasUvBuffer).offset(0).range(VK_WHOLE_SIZE);
            writes.get(6).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet0).dstBinding(6)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(uvInfo);

            // 7: uBlockAtlas
            long atlasView = (blockAtlasView != VK_NULL_HANDLE) ? blockAtlasView : dummyAtlasView;
            VkDescriptorImageInfo.Buffer atlasInfo = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    .imageView(atlasView)
                    .sampler(atlasSampler);
            writes.get(7).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet0).dstBinding(7)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(atlasInfo);

            vkUpdateDescriptorSets(device, writes, null);
        }
    }

    public void updateDescriptorSet1(long sharedColorView, long gbufferDepthView) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(8, stack);

            // 0: uGBufferRT0
            VkDescriptorImageInfo.Buffer rt0Info = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL).imageView(rt0View).sampler(nearestSampler);
            writes.get(0).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet1).dstBinding(0)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(rt0Info);

            // 1: uGBufferRT1
            VkDescriptorImageInfo.Buffer rt1Info = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL).imageView(rt1View).sampler(nearestSampler);
            writes.get(1).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet1).dstBinding(1)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(rt1Info);

            // 2: uDepthBuffer (internal Reversed-Z depth)
            VkDescriptorImageInfo.Buffer depthInfo = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL).imageView(gbufferDepthView).sampler(nearestSampler);
            writes.get(2).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet1).dstBinding(2)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(depthInfo);

            // 3: uHiZPyramid
            VkDescriptorImageInfo.Buffer hiZInfo = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL).imageView(hiZView).sampler(nearestSampler);
            writes.get(3).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet1).dstBinding(3)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(hiZInfo);

            // 4: uShadowMapArray
            VkDescriptorImageInfo.Buffer csmInfo = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL).imageView(csmArrayView).sampler(shadowSampler);
            writes.get(4).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet1).dstBinding(4)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(csmInfo);

            // 5: uHdrColorImg (STORAGE_IMAGE)
            VkDescriptorImageInfo.Buffer hdrInfo = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_GENERAL).imageView(hdrColorView);
            writes.get(5).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet1).dstBinding(5)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(hdrInfo);

            // 6: uSharedColorImg (STORAGE_IMAGE)
            VkDescriptorImageInfo.Buffer sharedInfo = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_GENERAL).imageView(sharedColorView);
            writes.get(6).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet1).dstBinding(6)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(sharedInfo);

            // 7: uHiZMips[16] (STORAGE_IMAGE array)
            VkDescriptorImageInfo.Buffer hiZMipInfos = VkDescriptorImageInfo.calloc(MAX_HIZ_MIPS, stack);
            for (int m = 0; m < MAX_HIZ_MIPS; m++) {
                long view = (m < hiZMipLevels) ? hiZMipViews[m] : hiZMipViews[0];
                hiZMipInfos.get(m).imageLayout(VK_IMAGE_LAYOUT_GENERAL).imageView(view);
            }
            writes.get(7).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet1).dstBinding(7)
                    .descriptorCount(MAX_HIZ_MIPS)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(hiZMipInfos);

            vkUpdateDescriptorSets(device, writes, null);
        }
    }

    private void recordImageBarrier(
            VkCommandBuffer cmd, long image, int oldLayout, int newLayout, int aspectMask,
            int baseMip, int mipCount, int baseLayer, int layerCount,
            int srcAccess, int dstAccess, int srcStage, int dstStage, MemoryStack stack
    ) {
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

    // ------------------------------------------------------------------------
    // Helper Image/View Factories
    // ------------------------------------------------------------------------

    private long createImage(int w, int h, int format, int usage, int mipLevels, int arrayLayers, LongBuffer outAllocation) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                    .imageType(VK_IMAGE_TYPE_2D)
                    .extent(it -> it.width(w).height(h).depth(1))
                    .mipLevels(mipLevels)
                    .arrayLayers(arrayLayers)
                    .format(format)
                    .tiling(VK_IMAGE_TILING_OPTIMAL)
                    .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                    .usage(usage)
                    .samples(VK_SAMPLE_COUNT_1_BIT)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);

            VmaAllocationCreateInfo allocInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_AUTO);

            LongBuffer pImage = stack.mallocLong(1);
            PointerBuffer pAlloc = stack.mallocPointer(1);
            int res = Vma.vmaCreateImage(vmaAllocator, imageInfo, allocInfo, pImage, pAlloc, null);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create image: " + res);

            outAllocation.put(0, pAlloc.get(0));
            return pImage.get(0);
        }
    }

    private long createImageView(long image, int format, int aspectFlags, int viewType, int baseMip, int mipCount, int baseLayer, int layerCount) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                    .image(image)
                    .viewType(viewType)
                    .format(format)
                    .subresourceRange(r -> r
                            .aspectMask(aspectFlags)
                            .baseMipLevel(baseMip)
                            .levelCount(mipCount)
                            .baseArrayLayer(baseLayer)
                            .layerCount(layerCount));

            LongBuffer pView = stack.mallocLong(1);
            int res = vkCreateImageView(device, viewInfo, null, pView);
            if (res != VK_SUCCESS) throw new RuntimeException("Failed to create image view: " + res);
            return pView.get(0);
        }
    }

    // ------------------------------------------------------------------------
    // Getters
    // ------------------------------------------------------------------------

    public long getDescriptorSetLayout0() { return descriptorSetLayout0; }
    public long getDescriptorSetLayout1() { return descriptorSetLayout1; }
    public long getPipelineLayout() { return pipelineLayout; }

    public long getDescriptorSet0() { return descriptorSet0; }
    public long getDescriptorSet1() { return descriptorSet1; }

    public long getTestGradientPipeline() { return testGradientPipeline; }
    public long getChunkCullPipeline() { return chunkCullPipeline; }
    public long getHizDownsamplePipeline() { return hizDownsamplePipeline; }
    public long getGbufferPipeline() { return gbufferPipeline; }
    public long getCsmShadowPipeline() { return csmShadowPipeline; }
    public long getDepthConvertPipeline() { return depthConvertPipeline; }
    public long getDeferredLightingPipeline() { return deferredLightingPipeline; }
    public long getPostCompositePipeline() { return postCompositePipeline; }

    public long getFrameUboBuffer() { return frameUboBuffer; }
    public long getRt0Image() { return rt0Image; }
    public long getRt0View() { return rt0View; }
    public long getRt1Image() { return rt1Image; }
    public long getRt1View() { return rt1View; }
    public long getGbufferDepthImage() { return gbufferDepthImage; }
    public long getGbufferDepthView() { return gbufferDepthView; }
    public long getHiZImage() { return hiZImage; }
    public long getHiZView() { return hiZView; }
    public int getHiZMipLevels() { return hiZMipLevels; }
    public long getBlockAtlasView() { return blockAtlasView != VK_NULL_HANDLE ? blockAtlasView : dummyAtlasView; }
    public long getCsmArrayImage() { return csmArrayImage; }
    public long getCsmArrayView() { return csmArrayView; }
    public long getCsmLayerView(int cascade) { return csmLayerViews[cascade]; }
    public long getHdrColorImage() { return hdrColorImage; }
    public long getHdrColorView() { return hdrColorView; }

    public long getNearestSampler() { return nearestSampler; }
    public long getLinearSampler() { return linearSampler; }
    public long getShadowSampler() { return shadowSampler; }

    public boolean isInitialized() { return initialized; }

    @Override
    public synchronized void close() {
        if (!initialized) return;

        LOGGER.info("[Valkyrie] Cleaning up VkPipelineManager...");
        vkDeviceWaitIdle(device);

        destroySizeDependentResources();

        for (int i = 0; i < CSM_CASCADE_COUNT; i++) {
            if (csmLayerViews[i] != VK_NULL_HANDLE) vkDestroyImageView(device, csmLayerViews[i], null);
        }
        if (csmArrayView != VK_NULL_HANDLE) vkDestroyImageView(device, csmArrayView, null);
        if (csmArrayImage != VK_NULL_HANDLE) Vma.vmaDestroyImage(vmaAllocator, csmArrayImage, csmArrayAllocation);

        if (blockAtlasView != VK_NULL_HANDLE) { vkDestroyImageView(device, blockAtlasView, null); blockAtlasView = VK_NULL_HANDLE; }
        if (blockAtlasImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, blockAtlasImage, blockAtlasAllocation); blockAtlasImage = VK_NULL_HANDLE; }

        if (dummyAtlasView != VK_NULL_HANDLE) vkDestroyImageView(device, dummyAtlasView, null);
        if (dummyAtlasImage != VK_NULL_HANDLE) Vma.vmaDestroyImage(vmaAllocator, dummyAtlasImage, dummyAtlasAllocation);

        if (nearestSampler != VK_NULL_HANDLE) vkDestroySampler(device, nearestSampler, null);
        if (linearSampler != VK_NULL_HANDLE) vkDestroySampler(device, linearSampler, null);
        if (shadowSampler != VK_NULL_HANDLE) vkDestroySampler(device, shadowSampler, null);
        if (atlasSampler != VK_NULL_HANDLE) vkDestroySampler(device, atlasSampler, null);

        if (frameUboAllocation != VK_NULL_HANDLE) {
            Vma.vmaUnmapMemory(vmaAllocator, frameUboAllocation);
            Vma.vmaDestroyBuffer(vmaAllocator, frameUboBuffer, frameUboAllocation);
            frameUboAddress = MemoryUtil.NULL;
        }
        MemoryUtil.memFree(frameUboHostBuffer);

        if (testGradientPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, testGradientPipeline, null);
        if (chunkCullPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, chunkCullPipeline, null);
        if (hizDownsamplePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, hizDownsamplePipeline, null);
        if (gbufferPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, gbufferPipeline, null);
        if (csmShadowPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, csmShadowPipeline, null);
        if (depthConvertPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, depthConvertPipeline, null);
        if (deferredLightingPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, deferredLightingPipeline, null);
        if (postCompositePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, postCompositePipeline, null);

        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, null);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, null);
        if (descriptorSetLayout0 != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout0, null);
        if (descriptorSetLayout1 != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout1, null);

        initialized = false;
        LOGGER.info("[Valkyrie] VkPipelineManager closed cleanly.");
    }
}
