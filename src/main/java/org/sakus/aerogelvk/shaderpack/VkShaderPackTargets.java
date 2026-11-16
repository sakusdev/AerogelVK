package org.sakus.aerogelvk.shaderpack;

import org.sakus.aerogelvk.vulkan.VkContext;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import static org.lwjgl.vulkan.VK10.*;

/**
 * Manages offscreen Vulkan 1.3 images, views, and samplers for OptiFine/Iris shaderpacks:
 * - colortex0..19 (ping-pong main & alt images with configurable formats and optional mipmaps)
 * - depthtex0..2 (D32_SFLOAT depth attachments / sampled depth textures)
 * - shadowtex0..1 (D32_SFLOAT shadow depth maps) & shadowcolor0..1 (RGBA8 shadow color maps)
 * - noisetex (256x256 procedural RGBA8 noise with repeat linear sampling)
 * - default 16x16 procedural lightmap texture + 1x1 white, black, and flat-normal fallback textures.
 */
public final class VkShaderPackTargets implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(VkShaderPackTargets.class);

    public static final int MAX_COLORTEX = ShaderDescriptorBindings.COLORTEX_COUNT;
    public static final int MAX_DEPTHTEX = 3;
    public static final int MAX_SHADOWTEX = 2;
    public static final int MAX_SHADOWCOLOR = 2;
    public static final int DEFAULT_SHADOW_RES = 1024;
    public static final int NOISE_RES = 256;
    public static final int LIGHTMAP_RES = 16;

    private VkDevice device;
    private long vmaAllocator;

    private int width = 1;
    private int height = 1;
    private int shadowResolution = DEFAULT_SHADOW_RES;

    // Color targets (0..19) - Ping-pong Main & Alt
    private final long[] colorMainImages = new long[MAX_COLORTEX];
    private final long[] colorMainAllocations = new long[MAX_COLORTEX];
    private final long[] colorMainSampleViews = new long[MAX_COLORTEX];
    private final long[] colorMainAttachViews = new long[MAX_COLORTEX];

    private final long[] colorAltImages = new long[MAX_COLORTEX];
    private final long[] colorAltAllocations = new long[MAX_COLORTEX];
    private final long[] colorAltSampleViews = new long[MAX_COLORTEX];
    private final long[] colorAltAttachViews = new long[MAX_COLORTEX];

    private final int[] colorFormats = new int[MAX_COLORTEX];
    private final int[] colorMipLevels = new int[MAX_COLORTEX];
    private final boolean[] colorMipmapRequested = new boolean[MAX_COLORTEX];
    private final boolean[] colorClearEnabled = new boolean[MAX_COLORTEX];
    private final float[][] colorClearColors = new float[MAX_COLORTEX][];
    private final boolean[] flipped = new boolean[MAX_COLORTEX];

    // Depth targets (0..2)
    private final long[] depthImages = new long[MAX_DEPTHTEX];
    private final long[] depthAllocations = new long[MAX_DEPTHTEX];
    private final long[] depthViews = new long[MAX_DEPTHTEX];

    // Shadow depth targets (0..1)
    private final long[] shadowDepthImages = new long[MAX_SHADOWTEX];
    private final long[] shadowDepthAllocations = new long[MAX_SHADOWTEX];
    private final long[] shadowDepthViews = new long[MAX_SHADOWTEX];

    // Shadow color targets (0..1)
    private final long[] shadowColorImages = new long[MAX_SHADOWCOLOR];
    private final long[] shadowColorAllocations = new long[MAX_SHADOWCOLOR];
    private final long[] shadowColorViews = new long[MAX_SHADOWCOLOR];

    // Static procedural & fallback textures
    private long noiseImage;
    private long noiseAllocation;
    private long noiseView;

    private long lightmapImage;
    private long lightmapAllocation;
    private long lightmapView;

    private long whiteImage;
    private long whiteAllocation;
    private long whiteView;

    private long blackImage;
    private long blackAllocation;
    private long blackView;

    private long flatNormalImage;
    private long flatNormalAllocation;
    private long flatNormalView;

    // Samplers
    private long linearSampler;
    private long nearestSampler;
    private long repeatLinearSampler;
    private long shadowSampler;

    private boolean initialized = false;

    public VkShaderPackTargets() {
        Arrays.fill(colorFormats, VK_FORMAT_R8G8B8A8_UNORM);
        Arrays.fill(colorMipLevels, 1);
        Arrays.fill(colorClearEnabled, true);
    }

    /**
     * Initializes offscreen shaderpack targets with default formats.
     */
    public synchronized void init(int width, int height) {
        init(width, height, DEFAULT_SHADOW_RES, null, null);
    }

    /**
     * Initializes offscreen shaderpack targets from a parsed {@link ShaderPackProperties}.
     */
    public synchronized void init(int width, int height, ShaderPackProperties properties) {
        if (properties == null) {
            init(width, height);
            return;
        }
        String[] fmtNames = new String[MAX_COLORTEX];
        boolean[] mipmaps = new boolean[MAX_COLORTEX];
        for (int i = 0; i < MAX_COLORTEX; i++) {
            fmtNames[i] = properties.getColorTexFormat(i);
            mipmaps[i] = properties.isColorTexMipmapEnabled(i);
            colorClearEnabled[i] = properties.isColorTexClearEnabled(i);
            colorClearColors[i] = properties.getColorTexClearColor(i);
        }
        init(width, height, resolveShadowResolution(properties.getShadowMapResolution()), fmtNames, mipmaps);
    }

    /**
     * Reconfigures offscreen shaderpack targets from a parsed {@link ShaderPackProperties}.
     */
    public synchronized void reconfigure(int newWidth, int newHeight, ShaderPackProperties properties) {
        if (properties == null) {
            reconfigure(newWidth, newHeight, resolveShadowResolution(DEFAULT_SHADOW_RES), null, null);
            return;
        }
        String[] fmtNames = new String[MAX_COLORTEX];
        boolean[] mipmaps = new boolean[MAX_COLORTEX];
        for (int i = 0; i < MAX_COLORTEX; i++) {
            fmtNames[i] = properties.getColorTexFormat(i);
            mipmaps[i] = properties.isColorTexMipmapEnabled(i);
            colorClearEnabled[i] = properties.isColorTexClearEnabled(i);
            colorClearColors[i] = properties.getColorTexClearColor(i);
        }
        reconfigure(newWidth, newHeight, resolveShadowResolution(properties.getShadowMapResolution()), fmtNames, mipmaps);
    }

    private static int resolveShadowResolution(int packShadowRes) {
        int cap = org.sakus.aerogelvk.config.ValkyrieConfig.get().getShaderPackShadowResolutionCap();
        int res = Math.max(1, packShadowRes);
        return (cap > 0) ? Math.min(res, cap) : res;
    }

    /**
     * Initializes offscreen shaderpack targets with parsed shaderpack formats and shadow resolution.
     */
    public synchronized void init(
            int width,
            int height,
            int shadowResolution,
            String[] colortexFormatNames,
            boolean[] colortexMipmaps
    ) {
        if (initialized) {
            reconfigure(width, height, shadowResolution, colortexFormatNames, colortexMipmaps);
            return;
        }

        VkContext ctx = VkContext.getInstance();
        if (!ctx.isInitialized()) {
            ctx.init();
        }
        this.device = ctx.getVkDevice();
        this.vmaAllocator = ctx.getVmaAllocator();
        this.width = Math.max(1, width);
        this.height = Math.max(1, height);
        this.shadowResolution = clampShadowRes(shadowResolution);

        applyFormatConfig(colortexFormatNames, colortexMipmaps);

        createSamplers();
        createStaticTextures();
        createShadowTargets();
        createSizeDependentTargets();

        initialized = true;
        LOGGER.info("[Valkyrie] VkShaderPackTargets initialized ({}x{}, shadowRes={}).",
                this.width, this.height, this.shadowResolution);
    }

    /**
     * Reconfigures formats, shadow resolution, or viewport size when switching shaderpacks.
     */
    public synchronized void reconfigure(
            int newWidth,
            int newHeight,
            int newShadowRes,
            String[] colortexFormatNames,
            boolean[] colortexMipmaps
    ) {
        if (!initialized) {
            init(newWidth, newHeight, newShadowRes, colortexFormatNames, colortexMipmaps);
            return;
        }

        vkDeviceWaitIdle(device);

        this.width = Math.max(1, newWidth);
        this.height = Math.max(1, newHeight);
        int clampedShadow = clampShadowRes(newShadowRes);
        boolean shadowChanged = (clampedShadow != this.shadowResolution);
        this.shadowResolution = clampedShadow;

        applyFormatConfig(colortexFormatNames, colortexMipmaps);

        destroySizeDependentTargets();
        if (shadowChanged) {
            destroyShadowTargets();
            createShadowTargets();
        }
        createSizeDependentTargets();
        resetPingPong();
    }

    /**
     * Resizes viewport-dependent colortex0..19 and depthtex0..2 targets.
     */
    public synchronized void resize(int newWidth, int newHeight) {
        newWidth = Math.max(1, newWidth);
        newHeight = Math.max(1, newHeight);
        if (!initialized) {
            init(newWidth, newHeight);
            return;
        }
        if (this.width == newWidth && this.height == newHeight) {
            return;
        }

        vkDeviceWaitIdle(device);
        this.width = newWidth;
        this.height = newHeight;

        destroySizeDependentTargets();
        createSizeDependentTargets();
        resetPingPong();
        LOGGER.info("[Valkyrie] Resized VkShaderPackTargets to {}x{}.", this.width, this.height);
    }

    private void applyFormatConfig(String[] formatNames, boolean[] mipmaps) {
        for (int i = 0; i < MAX_COLORTEX; i++) {
            String fmtName = (formatNames != null && i < formatNames.length) ? formatNames[i] : null;
            int requestedFormat = parseVkFormat(fmtName);
            boolean mipmapEnabled = (mipmaps != null && i < mipmaps.length && mipmaps[i]);
            colorFormats[i] = ensureSupportedColorFormat(requestedFormat, mipmapEnabled);
            colorMipmapRequested[i] = mipmapEnabled;
        }
    }

    private int ensureSupportedColorFormat(int vkFormat, boolean mipmapEnabled) {
        if (device == null) return vkFormat;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkFormatProperties props = VkFormatProperties.calloc(stack);
            vkGetPhysicalDeviceFormatProperties(VkContext.getInstance().getPhysicalDevice(), vkFormat, props);
            int required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if (mipmapEnabled) {
                required |= VK_FORMAT_FEATURE_BLIT_SRC_BIT
                        | VK_FORMAT_FEATURE_BLIT_DST_BIT
                        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
            }
            if ((props.optimalTilingFeatures() & required) == required) {
                return vkFormat;
            }
            LOGGER.warn("[Valkyrie] Vulkan format {} does not support required features (mipmap={}) on this GPU; falling back to RGBA16F.",
                    vkFormat, mipmapEnabled);
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        }
    }

    /**
     * Maps OptiFine/Iris colortex format strings to Vulkan VkFormat constants.
     * Note: Integer formats (e.g. RGBA16UI) are mapped to float equivalents (RGBA16F/RGBA32F)
     * when transpiled shaders declare floating-point vec4 fragment outputs, preventing pipeline mismatches.
     */
    public static int parseVkFormat(String formatName) {
        if (formatName == null || formatName.isBlank()) {
            return VK_FORMAT_R8G8B8A8_UNORM;
        }
        String norm = formatName.trim().toUpperCase(Locale.ROOT);
        return switch (norm) {
            case "R8" -> VK_FORMAT_R8_UNORM;
            case "RG8" -> VK_FORMAT_R8G8_UNORM;
            case "RGB8", "RGBA8", "RGBA" -> VK_FORMAT_R8G8B8A8_UNORM;
            case "R8_SNORM" -> VK_FORMAT_R8_SNORM;
            case "RG8_SNORM" -> VK_FORMAT_R8G8_SNORM;
            case "RGB8_SNORM", "RGBA8_SNORM" -> VK_FORMAT_R8G8B8A8_SNORM;
            case "R16" -> VK_FORMAT_R16_UNORM;
            case "RG16" -> VK_FORMAT_R16G16_UNORM;
            case "RGB16", "RGBA16" -> VK_FORMAT_R16G16B16A16_UNORM;
            case "R16_SNORM" -> VK_FORMAT_R16_SNORM;
            case "RG16_SNORM" -> VK_FORMAT_R16G16_SNORM;
            case "RGB16_SNORM", "RGBA16_SNORM" -> VK_FORMAT_R16G16B16A16_SNORM;
            case "R16F" -> VK_FORMAT_R16_SFLOAT;
            case "RG16F" -> VK_FORMAT_R16G16_SFLOAT;
            case "RGB16F", "RGBA16F" -> VK_FORMAT_R16G16B16A16_SFLOAT;
            case "R32F" -> VK_FORMAT_R32_SFLOAT;
            case "RG32F" -> VK_FORMAT_R32G32_SFLOAT;
            case "RGB32F", "RGBA32F" -> VK_FORMAT_R32G32B32A32_SFLOAT;
            case "R11F_G11F_B10F" -> VK_FORMAT_B10G11R11_UFLOAT_PACK32;
            case "RGB10_A2", "R10_G10_B10_A2" -> VK_FORMAT_A2B10G10R10_UNORM_PACK32;
            case "R8I", "R8UI", "R16I", "R16UI" -> VK_FORMAT_R16_SFLOAT;
            case "RG8I", "RG8UI", "RG16I", "RG16UI" -> VK_FORMAT_R16G16_SFLOAT;
            case "RGB8I", "RGBA8I", "RGB8UI", "RGBA8UI",
                 "RGB16I", "RGBA16I", "RGB16UI", "RGBA16UI" -> VK_FORMAT_R16G16B16A16_SFLOAT;
            case "R32I", "R32UI" -> VK_FORMAT_R32_SFLOAT;
            case "RG32I", "RG32UI" -> VK_FORMAT_R32G32_SFLOAT;
            case "RGB32I", "RGBA32I", "RGB32UI", "RGBA32UI" -> VK_FORMAT_R32G32B32A32_SFLOAT;
            default -> VK_FORMAT_R8G8B8A8_UNORM;
        };
    }

    private static int clampShadowRes(int res) {
        if (res <= 0) return DEFAULT_SHADOW_RES;
        return Math.max(128, Math.min(4096, res));
    }

    private void createSamplers() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer pSampler = stack.mallocLong(1);

            VkSamplerCreateInfo linearInfo = VkSamplerCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                    .magFilter(VK_FILTER_LINEAR)
                    .minFilter(VK_FILTER_LINEAR)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_LINEAR)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(16.0f);
            if (vkCreateSampler(device, linearInfo, null, pSampler) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create linear sampler");
            }
            linearSampler = pSampler.get(0);

            VkSamplerCreateInfo nearestInfo = VkSamplerCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                    .magFilter(VK_FILTER_NEAREST)
                    .minFilter(VK_FILTER_NEAREST)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(16.0f);
            if (vkCreateSampler(device, nearestInfo, null, pSampler) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create nearest sampler");
            }
            nearestSampler = pSampler.get(0);

            VkSamplerCreateInfo repeatInfo = VkSamplerCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                    .magFilter(VK_FILTER_LINEAR)
                    .minFilter(VK_FILTER_LINEAR)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_LINEAR)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .minLod(0.0f)
                    .maxLod(1.0f);
            if (vkCreateSampler(device, repeatInfo, null, pSampler) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create repeat linear sampler");
            }
            repeatLinearSampler = pSampler.get(0);

            // Note: Many OptiFine shaderpacks declare shadowtex0/1 as sampler2D and do manual depth comparison,
            // while others use shadow2D. A linear clamp sampler without hardware compare works with sampler2D.
            VkSamplerCreateInfo shadowInfo = VkSamplerCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                    .magFilter(VK_FILTER_LINEAR)
                    .minFilter(VK_FILTER_LINEAR)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .compareEnable(false)
                    .minLod(0.0f)
                    .maxLod(1.0f);
            if (vkCreateSampler(device, shadowInfo, null, pSampler) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create shadow sampler");
            }
            shadowSampler = pSampler.get(0);
        }
    }

    private void createStaticTextures() {
        // 1. 256x256 RGBA8 procedural noise texture
        ByteBuffer noiseBytes = MemoryUtil.memAlloc(NOISE_RES * NOISE_RES * 4);
        try {
            Random rng = new Random(0x5EED_CAFE_1234L);
            byte[] row = new byte[NOISE_RES * 4];
            for (int y = 0; y < NOISE_RES; y++) {
                rng.nextBytes(row);
                noiseBytes.put(row);
            }
            noiseBytes.flip();

            long[] out = createUploadedRgbaTexture(NOISE_RES, NOISE_RES, noiseBytes);
            noiseImage = out[0];
            noiseAllocation = out[1];
            noiseView = out[2];
        } finally {
            MemoryUtil.memFree(noiseBytes);
        }

        // 2. 16x16 RGBA8 procedural Minecraft lightmap (x = blockLight 0..15, y = skyLight 0..15)
        ByteBuffer lmBytes = MemoryUtil.memAlloc(LIGHTMAP_RES * LIGHTMAP_RES * 4);
        try {
            for (int y = 0; y < LIGHTMAP_RES; y++) {
                float sky = y / 15.0f;
                float skyIntensity = (float) Math.pow(sky, 1.35) * 0.92f + 0.08f;
                for (int x = 0; x < LIGHTMAP_RES; x++) {
                    float block = x / 15.0f;
                    float blockIntensity = (float) Math.pow(block, 1.25) * 0.95f;

                    float r = Math.min(1.0f, skyIntensity * 0.96f + blockIntensity * 1.0f);
                    float g = Math.min(1.0f, skyIntensity * 0.98f + blockIntensity * 0.85f);
                    float b = Math.min(1.0f, skyIntensity * 1.00f + blockIntensity * 0.65f);

                    lmBytes.put((byte) Math.round(r * 255.0f));
                    lmBytes.put((byte) Math.round(g * 255.0f));
                    lmBytes.put((byte) Math.round(b * 255.0f));
                    lmBytes.put((byte) 255);
                }
            }
            lmBytes.flip();

            long[] out = createUploadedRgbaTexture(LIGHTMAP_RES, LIGHTMAP_RES, lmBytes);
            lightmapImage = out[0];
            lightmapAllocation = out[1];
            lightmapView = out[2];
        } finally {
            MemoryUtil.memFree(lmBytes);
        }

        // 3. 1x1 fallback textures: white, black, flat normal
        whiteImage = create1x1ColorTexture(1.0f, 1.0f, 1.0f, 1.0f, out -> {
            whiteAllocation = out[0];
            whiteView = out[1];
        });
        blackImage = create1x1ColorTexture(0.0f, 0.0f, 0.0f, 0.0f, out -> {
            blackAllocation = out[0];
            blackView = out[1];
        });
        flatNormalImage = create1x1ColorTexture(0.5f, 0.5f, 1.0f, 1.0f, out -> {
            flatNormalAllocation = out[0];
            flatNormalView = out[1];
        });
    }

    private interface AllocViewConsumer {
        void accept(long[] allocAndView);
    }

    private long create1x1ColorTexture(float r, float g, float b, float a, AllocViewConsumer consumer) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer pAlloc = stack.mallocLong(1);
            long img = createImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, pAlloc);
            long alloc = pAlloc.get(0);
            long view = createImageView(img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            recordImageBarrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

            VkClearColorValue clearVal = VkClearColorValue.calloc(stack);
            clearVal.float32(0, r).float32(1, g).float32(2, b).float32(3, a);
            VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack)
                    .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clearVal, range);

            recordImageBarrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);
            ctx.endSingleTimeCommands(cmd);

            consumer.accept(new long[]{alloc, view});
            return img;
        }
    }

    private long[] createUploadedRgbaTexture(int texW, int texH, ByteBuffer rgbaPixels) {
        long byteSize = (long) texW * texH * 4L;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkBufferCreateInfo bufInfo = VkBufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(byteSize)
                    .usage(VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            VmaAllocationCreateInfo bufAllocInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_CPU_TO_GPU)
                    .flags(Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | Vma.VMA_ALLOCATION_CREATE_MAPPED_BIT);

            LongBuffer pBuf = stack.mallocLong(1);
            PointerBuffer pBufAlloc = stack.mallocPointer(1);
            int res = Vma.vmaCreateBuffer(vmaAllocator, bufInfo, bufAllocInfo, pBuf, pBufAlloc, null);
            if (res != VK_SUCCESS) {
                throw new RuntimeException("Failed to create texture staging buffer: " + res);
            }
            long stagingBuf = pBuf.get(0);
            long stagingAlloc = pBufAlloc.get(0);

            PointerBuffer pMapped = stack.mallocPointer(1);
            Vma.vmaMapMemory(vmaAllocator, stagingAlloc, pMapped);
            ByteBuffer src = rgbaPixels.duplicate();
            src.position(0).limit((int) byteSize);
            MemoryUtil.memByteBuffer(pMapped.get(0), (int) byteSize).put(src);
            Vma.vmaUnmapMemory(vmaAllocator, stagingAlloc);

            LongBuffer pImgAlloc = stack.mallocLong(1);
            long img = createImage(texW, texH, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, pImgAlloc);
            long imgAlloc = pImgAlloc.get(0);
            long view = createImageView(img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            recordImageBarrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

            VkBufferImageCopy.Buffer copyRegion = VkBufferImageCopy.calloc(1, stack)
                    .bufferOffset(0).bufferRowLength(0).bufferImageHeight(0);
            copyRegion.imageSubresource()
                    .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            copyRegion.imageExtent().set(texW, texH, 1);
            vkCmdCopyBufferToImage(cmd, stagingBuf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copyRegion);

            recordImageBarrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);
            ctx.endSingleTimeCommands(cmd);

            Vma.vmaDestroyBuffer(vmaAllocator, stagingBuf, stagingAlloc);
            return new long[]{img, imgAlloc, view};
        }
    }

    private void createShadowTargets() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer pAlloc = stack.mallocLong(1);
            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

            for (int i = 0; i < MAX_SHADOWTEX; i++) {
                shadowDepthImages[i] = createImage(
                        shadowResolution, shadowResolution, VK_FORMAT_D32_SFLOAT,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        1, pAlloc
                );
                shadowDepthAllocations[i] = pAlloc.get(0);
                shadowDepthViews[i] = createImageView(
                        shadowDepthImages[i], VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1
                );

                recordClearDepthImage(cmd, shadowDepthImages[i], 1.0f, stack);
            }

            for (int i = 0; i < MAX_SHADOWCOLOR; i++) {
                shadowColorImages[i] = createImage(
                        shadowResolution, shadowResolution, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        1, pAlloc
                );
                shadowColorAllocations[i] = pAlloc.get(0);
                shadowColorViews[i] = createImageView(
                        shadowColorImages[i], VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1
                );

                recordClearColorImage(cmd, shadowColorImages[i], 1, 0f, 0f, 0f, 0f, stack);
            }

            ctx.endSingleTimeCommands(cmd);
        }
    }

    private void createSizeDependentTargets() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer pAlloc = stack.mallocLong(1);
            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

            int maxDim = Math.max(width, height);
            int fullMips = Math.max(1, 1 + (int) Math.floor(Math.log(maxDim) / Math.log(2)));

            for (int i = 0; i < MAX_COLORTEX; i++) {
                int fmt = colorFormats[i];
                int mips = colorMipmapRequested[i] ? Math.min(fullMips, 8) : 1;
                colorMipLevels[i] = mips;

                int usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT
                        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                        | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

                // Main image
                colorMainImages[i] = createImage(width, height, fmt, usage, mips, pAlloc);
                colorMainAllocations[i] = pAlloc.get(0);
                colorMainSampleViews[i] = createImageView(colorMainImages[i], fmt, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips);
                colorMainAttachViews[i] = (mips == 1)
                        ? colorMainSampleViews[i]
                        : createImageView(colorMainImages[i], fmt, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

                // Alt image
                colorAltImages[i] = createImage(width, height, fmt, usage, mips, pAlloc);
                colorAltAllocations[i] = pAlloc.get(0);
                colorAltSampleViews[i] = createImageView(colorAltImages[i], fmt, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips);
                colorAltAttachViews[i] = (mips == 1)
                        ? colorAltSampleViews[i]
                        : createImageView(colorAltImages[i], fmt, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

                recordClearColorImage(cmd, colorMainImages[i], mips, 0f, 0f, 0f, 0f, stack);
                recordClearColorImage(cmd, colorAltImages[i], mips, 0f, 0f, 0f, 0f, stack);
            }

            for (int i = 0; i < MAX_DEPTHTEX; i++) {
                depthImages[i] = createImage(
                        width, height, VK_FORMAT_D32_SFLOAT,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        1, pAlloc
                );
                depthAllocations[i] = pAlloc.get(0);
                depthViews[i] = createImageView(depthImages[i], VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1);

                recordClearDepthImage(cmd, depthImages[i], 1.0f, stack);
            }

            ctx.endSingleTimeCommands(cmd);
        }
    }

    /**
     * Clears enabled colortex0..19 (main & alt), depthtex0..2, shadowtex0..1, and shadowcolor0..1 at the beginning
     * of a frame and transitions all of them to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
     */
    public void recordClearAllTargets(VkCommandBuffer cmd, float clearR, float clearG, float clearB, MemoryStack stack) {
        recordClearAllTargets(cmd, clearR, clearG, clearB, true, stack);
    }

    public void recordClearAllTargets(
            VkCommandBuffer cmd,
            float clearR,
            float clearG,
            float clearB,
            boolean clearShadows,
            MemoryStack stack) {
        for (int i = 0; i < MAX_COLORTEX; i++) {
            if (!colorClearEnabled[i]) {
                continue;
            }
            flipped[i] = false;
            int mips = colorMipLevels[i];
            float[] custom = colorClearColors[i];
            float r = (custom != null && custom.length >= 4) ? custom[0] : ((i == 0) ? clearR : 0f);
            float g = (custom != null && custom.length >= 4) ? custom[1] : ((i == 0) ? clearG : 0f);
            float b = (custom != null && custom.length >= 4) ? custom[2] : ((i == 0) ? clearB : 0f);
            float a = (custom != null && custom.length >= 4) ? custom[3] : 0f;
            recordClearColorImage(cmd, colorMainImages[i], mips, r, g, b, a, stack);
            recordClearColorImage(cmd, colorAltImages[i], mips, r, g, b, a, stack);
        }
        for (int i = 0; i < MAX_DEPTHTEX; i++) {
            recordClearDepthImage(cmd, depthImages[i], 1.0f, stack);
        }
        if (clearShadows) {
            for (int i = 0; i < MAX_SHADOWTEX; i++) {
                recordClearDepthImage(cmd, shadowDepthImages[i], 1.0f, stack);
            }
            for (int i = 0; i < MAX_SHADOWCOLOR; i++) {
                recordClearColorImage(cmd, shadowColorImages[i], 1, 1f, 1f, 1f, 1f, stack);
            }
        }
    }

    /**
     * Copies depthtex[srcIdx] into depthtex[dstIdx] (both expected in SHADER_READ_ONLY_OPTIMAL before and after).
     */
    public void recordCopyDepth(VkCommandBuffer cmd, int srcIdx, int dstIdx, MemoryStack stack) {
        if (srcIdx == dstIdx || srcIdx < 0 || srcIdx >= MAX_DEPTHTEX || dstIdx < 0 || dstIdx >= MAX_DEPTHTEX) {
            return;
        }
        long srcImg = depthImages[srcIdx];
        long dstImg = depthImages[dstIdx];

        recordImageBarrier(cmd, srcImg,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        recordImageBarrier(cmd, dstImg,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
        region.srcSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.dstSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.extent().set(width, height, 1);

        vkCmdCopyImage(cmd,
                srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                dstImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                region);

        recordImageBarrier(cmd, srcImg,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, stack);

        recordImageBarrier(cmd, dstImg,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, stack);
    }

    /**
     * Copies shadowtex[srcIdx] into shadowtex[dstIdx] (both in SHADER_READ_ONLY_OPTIMAL before and after).
     */
    public void recordCopyShadowDepth(VkCommandBuffer cmd, int srcIdx, int dstIdx, MemoryStack stack) {
        if (srcIdx == dstIdx || srcIdx < 0 || srcIdx >= MAX_SHADOWTEX || dstIdx < 0 || dstIdx >= MAX_SHADOWTEX) {
            return;
        }
        long srcImg = shadowDepthImages[srcIdx];
        long dstImg = shadowDepthImages[dstIdx];

        recordImageBarrier(cmd, srcImg,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        recordImageBarrier(cmd, dstImg,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
        region.srcSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.dstSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
        region.extent().set(shadowResolution, shadowResolution, 1);

        vkCmdCopyImage(cmd,
                srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                dstImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                region);

        recordImageBarrier(cmd, srcImg,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, stack);

        recordImageBarrier(cmd, dstImg,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, stack);
    }

    /**
     * Copies a D32_SFLOAT depth image into an external shared depth image (transitions sharedDepthImage to TRANSFER_SRC_OPTIMAL).
     */
    public void recordCopyDepthToExternal(
            VkCommandBuffer cmd,
            int srcDepthIdx,
            long dstSharedDepthImage,
            MemoryStack stack
    ) {
        recordCopyDepthToExternal(cmd, srcDepthIdx, dstSharedDepthImage, width, height, stack);
    }

    public void recordCopyDepthToExternal(
            VkCommandBuffer cmd,
            int srcDepthIdx,
            long dstSharedDepthImage,
            int dstWidth,
            int dstHeight,
            MemoryStack stack
    ) {
        long srcImg = depthImages[srcDepthIdx];

        recordImageBarrier(cmd, srcImg,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        recordImageBarrier(cmd, dstSharedDepthImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        if (dstWidth == width && dstHeight == height) {
            VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
            region.srcSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.dstSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.extent().set(width, height, 1);

            vkCmdCopyImage(cmd,
                    srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    dstSharedDepthImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    region);
        } else {
            VkImageBlit.Buffer blit = VkImageBlit.calloc(1, stack);
            blit.srcSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            blit.srcOffsets(0).set(0, 0, 0);
            blit.srcOffsets(1).set(width, height, 1);
            blit.dstSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            blit.dstOffsets(0).set(0, 0, 0);
            blit.dstOffsets(1).set(Math.max(1, dstWidth), Math.max(1, dstHeight), 1);

            vkCmdBlitImage(cmd,
                    srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    dstSharedDepthImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    blit, VK_FILTER_NEAREST);
        }

        recordImageBarrier(cmd, srcImg,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, stack);

        recordImageBarrier(cmd, dstSharedDepthImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);
    }

    /**
     * Generates mipmaps for a color image whose mip 0 is currently in COLOR_ATTACHMENT_OPTIMAL,
     * transitioning all mip levels to SHADER_READ_ONLY_OPTIMAL when done.
     */
    public void recordTransitionOrMipmapAfterColorWrite(
            VkCommandBuffer cmd,
            long image,
            int mipLevels,
            MemoryStack stack
    ) {
        if (mipLevels <= 1) {
            recordImageBarrier(cmd, image,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);
            return;
        }

        // Transition mip 0 from COLOR_ATTACHMENT_OPTIMAL to TRANSFER_SRC_OPTIMAL
        recordImageBarrier(cmd, image,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        // Transition mips 1..mipLevels-1 to TRANSFER_DST_OPTIMAL
        recordImageBarrier(cmd, image,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 1, mipLevels - 1, 0, 1,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        int mipW = width;
        int mipH = height;
        for (int i = 1; i < mipLevels; i++) {
            int nextW = Math.max(1, mipW / 2);
            int nextH = Math.max(1, mipH / 2);

            VkImageBlit.Buffer blit = VkImageBlit.calloc(1, stack);
            blit.srcOffsets(0).set(0, 0, 0);
            blit.srcOffsets(1).set(mipW, mipH, 1);
            blit.srcSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(i - 1).baseArrayLayer(0).layerCount(1);

            blit.dstOffsets(0).set(0, 0, 0);
            blit.dstOffsets(1).set(nextW, nextH, 1);
            blit.dstSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(i).baseArrayLayer(0).layerCount(1);

            vkCmdBlitImage(cmd,
                    image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    blit, VK_FILTER_LINEAR);

            recordImageBarrier(cmd, image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);

            if (i + 1 < mipLevels) {
                recordImageBarrier(cmd, image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);
            } else {
                recordImageBarrier(cmd, image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);
            }

            mipW = nextW;
            mipH = nextH;
        }
    }

    private void recordClearColorImage(
            VkCommandBuffer cmd,
            long image,
            int mipLevels,
            float r, float g, float b, float a,
            MemoryStack stack
    ) {
        recordImageBarrier(cmd, image,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        VkClearColorValue clearColor = VkClearColorValue.calloc(stack);
        clearColor.float32(0, r).float32(1, g).float32(2, b).float32(3, a);
        VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack)
                .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                .baseMipLevel(0).levelCount(mipLevels)
                .baseArrayLayer(0).layerCount(1);
        vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clearColor, range);

        recordImageBarrier(cmd, image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);
    }

    private void recordClearDepthImage(VkCommandBuffer cmd, long image, float depthVal, MemoryStack stack) {
        recordImageBarrier(cmd, image,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        VkClearDepthStencilValue clearDepth = VkClearDepthStencilValue.calloc(stack).depth(depthVal).stencil(0);
        VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack)
                .aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT)
                .baseMipLevel(0).levelCount(1)
                .baseArrayLayer(0).layerCount(1);
        vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clearDepth, range);

        recordImageBarrier(cmd, image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);
    }

    public static void recordImageBarrier(
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
            int dstStage,
            MemoryStack stack
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

    private long createImage(int w, int h, int format, int usage, int mipLevels, LongBuffer outAllocation) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                    .imageType(VK_IMAGE_TYPE_2D)
                    .extent(e -> e.width(w).height(h).depth(1))
                    .mipLevels(mipLevels)
                    .arrayLayers(1)
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
            if (res != VK_SUCCESS) {
                throw new RuntimeException("Failed to create VkShaderPackTargets image (fmt=" + format + "): " + res);
            }
            outAllocation.put(0, pAlloc.get(0));
            return pImage.get(0);
        }
    }

    private long createImageView(long image, int format, int aspectMask, int baseMip, int levelCount) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                    .image(image)
                    .viewType(VK_IMAGE_VIEW_TYPE_2D)
                    .format(format)
                    .subresourceRange(r -> r
                            .aspectMask(aspectMask)
                            .baseMipLevel(baseMip)
                            .levelCount(levelCount)
                            .baseArrayLayer(0)
                            .layerCount(1));

            LongBuffer pView = stack.mallocLong(1);
            int res = vkCreateImageView(device, viewInfo, null, pView);
            if (res != VK_SUCCESS) {
                throw new RuntimeException("Failed to create VkShaderPackTargets imageView: " + res);
            }
            return pView.get(0);
        }
    }

    // ------------------------------------------------------------------------
    // Ping-pong & Target Accessors
    // ------------------------------------------------------------------------

    public void resetPingPong() {
        Arrays.fill(flipped, false);
    }

    public void flip(int index) {
        if (index >= 0 && index < MAX_COLORTEX) {
            flipped[index] = !flipped[index];
        }
    }

    public boolean isFlipped(int index) {
        return index >= 0 && index < MAX_COLORTEX && flipped[index];
    }

    /**
     * Returns the current READ VkImage for colortex[index] (sampled by shaders in Set 1).
     */
    public long getColorReadImage(int index) {
        return flipped[index] ? colorAltImages[index] : colorMainImages[index];
    }

    /**
     * Returns the current READ VkImageView (all mip levels) for colortex[index] (sampled in Set 1).
     */
    public long getColorReadView(int index) {
        return flipped[index] ? colorAltSampleViews[index] : colorMainSampleViews[index];
    }

    /**
     * Returns the current READ base-level attachment VkImageView for colortex[index] (used by gbuffers passes
     * that accumulate into the main target before composite ping-pong begins).
     */
    public long getColorReadAttachmentView(int index) {
        return flipped[index] ? colorAltAttachViews[index] : colorMainAttachViews[index];
    }

    /**
     * Returns the current WRITE VkImage for colortex[index] (ping-pong partner opposite to READ).
     */
    public long getColorWriteImage(int index) {
        return flipped[index] ? colorMainImages[index] : colorAltImages[index];
    }

    /**
     * Returns the current WRITE VkImageView (all mip levels) for colortex[index].
     */
    public long getColorWriteSampleView(int index) {
        return flipped[index] ? colorMainSampleViews[index] : colorAltSampleViews[index];
    }

    /**
     * Returns the current WRITE attachment VkImageView (mip 0 only) for colortex[index].
     */
    public long getColorWriteAttachmentView(int index) {
        return flipped[index] ? colorMainAttachViews[index] : colorAltAttachViews[index];
    }

    public long getColorMainImage(int index) { return colorMainImages[index]; }
    public long getColorMainView(int index) { return colorMainSampleViews[index]; }
    public long getColorMainAttachmentView(int index) { return colorMainAttachViews[index]; }

    public long getColorAltImage(int index) { return colorAltImages[index]; }
    public long getColorAltView(int index) { return colorAltSampleViews[index]; }
    public long getColorAltAttachmentView(int index) { return colorAltAttachViews[index]; }

    public int getColorFormat(int index) { return colorFormats[index]; }
    public int getColorMipLevels(int index) { return colorMipLevels[index]; }

    public long getDepthImage(int index) { return depthImages[index]; }
    public long getDepthView(int index) { return depthViews[index]; }

    public long getShadowDepthImage(int index) { return shadowDepthImages[index]; }
    public long getShadowDepthView(int index) { return shadowDepthViews[index]; }

    public long getShadowColorImage(int index) { return shadowColorImages[index]; }
    public long getShadowColorView(int index) { return shadowColorViews[index]; }

    public long getNoiseImage() { return noiseImage; }
    public long getNoiseView() { return noiseView; }

    public long getLightmapImage() { return lightmapImage; }
    public long getLightmapView() { return lightmapView; }

    public long getWhiteImage() { return whiteImage; }
    public long getWhiteView() { return whiteView; }

    public long getBlackImage() { return blackImage; }
    public long getBlackView() { return blackView; }

    public long getFlatNormalImage() { return flatNormalImage; }
    public long getFlatNormalView() { return flatNormalView; }

    public long getLinearSampler() { return linearSampler; }
    public long getNearestSampler() { return nearestSampler; }
    public long getRepeatLinearSampler() { return repeatLinearSampler; }
    public long getShadowSampler() { return shadowSampler; }

    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public int getShadowResolution() { return shadowResolution; }
    public boolean isInitialized() { return initialized; }

    // ------------------------------------------------------------------------
    // Cleanup & VMA Destruction
    // ------------------------------------------------------------------------

    private void destroySizeDependentTargets() {
        for (int i = 0; i < MAX_COLORTEX; i++) {
            if (colorMainAttachViews[i] != VK_NULL_HANDLE && colorMainAttachViews[i] != colorMainSampleViews[i]) {
                vkDestroyImageView(device, colorMainAttachViews[i], null);
            }
            colorMainAttachViews[i] = VK_NULL_HANDLE;
            if (colorMainSampleViews[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(device, colorMainSampleViews[i], null);
                colorMainSampleViews[i] = VK_NULL_HANDLE;
            }
            if (colorMainImages[i] != VK_NULL_HANDLE) {
                Vma.vmaDestroyImage(vmaAllocator, colorMainImages[i], colorMainAllocations[i]);
                colorMainImages[i] = VK_NULL_HANDLE;
                colorMainAllocations[i] = VK_NULL_HANDLE;
            }

            if (colorAltAttachViews[i] != VK_NULL_HANDLE && colorAltAttachViews[i] != colorAltSampleViews[i]) {
                vkDestroyImageView(device, colorAltAttachViews[i], null);
            }
            colorAltAttachViews[i] = VK_NULL_HANDLE;
            if (colorAltSampleViews[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(device, colorAltSampleViews[i], null);
                colorAltSampleViews[i] = VK_NULL_HANDLE;
            }
            if (colorAltImages[i] != VK_NULL_HANDLE) {
                Vma.vmaDestroyImage(vmaAllocator, colorAltImages[i], colorAltAllocations[i]);
                colorAltImages[i] = VK_NULL_HANDLE;
                colorAltAllocations[i] = VK_NULL_HANDLE;
            }
        }

        for (int i = 0; i < MAX_DEPTHTEX; i++) {
            if (depthViews[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(device, depthViews[i], null);
                depthViews[i] = VK_NULL_HANDLE;
            }
            if (depthImages[i] != VK_NULL_HANDLE) {
                Vma.vmaDestroyImage(vmaAllocator, depthImages[i], depthAllocations[i]);
                depthImages[i] = VK_NULL_HANDLE;
                depthAllocations[i] = VK_NULL_HANDLE;
            }
        }
    }

    private void destroyShadowTargets() {
        for (int i = 0; i < MAX_SHADOWTEX; i++) {
            if (shadowDepthViews[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(device, shadowDepthViews[i], null);
                shadowDepthViews[i] = VK_NULL_HANDLE;
            }
            if (shadowDepthImages[i] != VK_NULL_HANDLE) {
                Vma.vmaDestroyImage(vmaAllocator, shadowDepthImages[i], shadowDepthAllocations[i]);
                shadowDepthImages[i] = VK_NULL_HANDLE;
                shadowDepthAllocations[i] = VK_NULL_HANDLE;
            }
        }
        for (int i = 0; i < MAX_SHADOWCOLOR; i++) {
            if (shadowColorViews[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(device, shadowColorViews[i], null);
                shadowColorViews[i] = VK_NULL_HANDLE;
            }
            if (shadowColorImages[i] != VK_NULL_HANDLE) {
                Vma.vmaDestroyImage(vmaAllocator, shadowColorImages[i], shadowColorAllocations[i]);
                shadowColorImages[i] = VK_NULL_HANDLE;
                shadowColorAllocations[i] = VK_NULL_HANDLE;
            }
        }
    }

    private void destroyStaticTextures() {
        if (noiseView != VK_NULL_HANDLE) { vkDestroyImageView(device, noiseView, null); noiseView = VK_NULL_HANDLE; }
        if (noiseImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, noiseImage, noiseAllocation); noiseImage = VK_NULL_HANDLE; }

        if (lightmapView != VK_NULL_HANDLE) { vkDestroyImageView(device, lightmapView, null); lightmapView = VK_NULL_HANDLE; }
        if (lightmapImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, lightmapImage, lightmapAllocation); lightmapImage = VK_NULL_HANDLE; }

        if (whiteView != VK_NULL_HANDLE) { vkDestroyImageView(device, whiteView, null); whiteView = VK_NULL_HANDLE; }
        if (whiteImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, whiteImage, whiteAllocation); whiteImage = VK_NULL_HANDLE; }

        if (blackView != VK_NULL_HANDLE) { vkDestroyImageView(device, blackView, null); blackView = VK_NULL_HANDLE; }
        if (blackImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, blackImage, blackAllocation); blackImage = VK_NULL_HANDLE; }

        if (flatNormalView != VK_NULL_HANDLE) { vkDestroyImageView(device, flatNormalView, null); flatNormalView = VK_NULL_HANDLE; }
        if (flatNormalImage != VK_NULL_HANDLE) { Vma.vmaDestroyImage(vmaAllocator, flatNormalImage, flatNormalAllocation); flatNormalImage = VK_NULL_HANDLE; }

        if (linearSampler != VK_NULL_HANDLE) { vkDestroySampler(device, linearSampler, null); linearSampler = VK_NULL_HANDLE; }
        if (nearestSampler != VK_NULL_HANDLE) { vkDestroySampler(device, nearestSampler, null); nearestSampler = VK_NULL_HANDLE; }
        if (repeatLinearSampler != VK_NULL_HANDLE) { vkDestroySampler(device, repeatLinearSampler, null); repeatLinearSampler = VK_NULL_HANDLE; }
        if (shadowSampler != VK_NULL_HANDLE) { vkDestroySampler(device, shadowSampler, null); shadowSampler = VK_NULL_HANDLE; }
    }

    public synchronized void cleanup() {
        close();
    }

    @Override
    public synchronized void close() {
        if (!initialized) return;
        LOGGER.info("[Valkyrie] Cleaning up VkShaderPackTargets...");
        if (device != null) {
            vkDeviceWaitIdle(device);
            destroySizeDependentTargets();
            destroyShadowTargets();
            destroyStaticTextures();
        }
        initialized = false;
        LOGGER.info("[Valkyrie] VkShaderPackTargets closed cleanly.");
    }
}
