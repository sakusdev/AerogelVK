package org.sakus.aerogelvk.vulkan;

import com.mojang.blaze3d.pipeline.RenderTarget;
import org.lwjgl.PointerBuffer;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.opengl.*;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.Platform;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Locale;

/**
 * Manages zero-copy OpenGL 4.6 - Vulkan 1.3 interop using EXT_external_objects:
 * - GL_EXT_memory_object & GL_EXT_memory_object_win32 / GL_EXT_memory_object_fd
 * - GL_EXT_semaphore & GL_EXT_semaphore_win32 / GL_EXT_semaphore_fd
 *
 * Allocates and shares:
 * 1. Shared Color Image (VK_FORMAT_R8G8B8A8_UNORM / GL_RGBA8)
 * 2. Shared Depth Image (VK_FORMAT_D32_SFLOAT / GL_DEPTH_COMPONENT32F)
 * 3. Shared Semaphores (vkRenderCompleteSemaphore, glReadySemaphore)
 * 4. OpenGL Framebuffer (glSharedFbo) for direct blitting into Minecraft's main RenderTarget.
 */
public class VkInteropManager implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(VkInteropManager.class);
    private static VkInteropManager instance;

    public enum InteropMode {
        ZERO_COPY_SEMAPHORE,
        ZERO_COPY_FENCE,
        STAGING_READBACK
    }

    private int width = 0;
    private int height = 0;
    private boolean initialized = false;
    private InteropMode interopMode = InteropMode.STAGING_READBACK;
    private boolean glInteropSupported = false;
    private boolean glSemaphoreSignaled = false;

    // Staging readback buffer (used only in STAGING_READBACK fallback mode)
    private long stagingBuffer = 0;
    private long stagingMemory = 0;
    private long stagingMappedAddr = 0;
    private int stagingColorBytes = 0;
    private int stagingTotalBytes = 0;

    // Vulkan Color resources
    private long colorImage = 0;
    private long colorMemory = 0;
    private long colorImageView = 0;

    // OpenGL Color resources
    private int glColorMemObject = 0;
    private int glSharedColorTex = 0;

    // Vulkan Depth resources
    private long depthImage = 0;
    private long depthMemory = 0;
    private long depthImageView = 0;

    // OpenGL Depth resources
    private int glDepthMemObject = 0;
    private int glSharedDepthTex = 0;

    // OpenGL Framebuffer
    private int glSharedFbo = 0;

    // OpenGL Depth Copy & Distant Horizons Composite helpers (handle mismatched depth formats such as D32F -> D24)
    private int glDepthCopyProgram = 0;
    private int glDepthCopyVao = 0;
    private int glDepthCopyTexLoc = -1;
    private int glDhCompositeProgram = 0;
    private int glDhMcDepthLoc = -1;
    private int glDhColorLoc = -1;
    private int glDhDepthLoc = -1;

    // Shared Semaphores
    private long vkRenderCompleteSemaphore = 0;
    private int glVkRenderCompleteSemaphore = 0;

    private long vkGlReadySemaphore = 0;
    private int glReadySemaphore = 0;

    public static synchronized VkInteropManager getInstance() {
        if (instance == null) {
            instance = new VkInteropManager();
        }
        return instance;
    }

    public static VkInteropManager get() {
        return getInstance();
    }

    /**
     * Recreates shared textures and framebuffers matching the given window dimensions.
     */
    public synchronized void recreate(int newWidth, int newHeight) {
        newWidth = Math.max(1, newWidth);
        newHeight = Math.max(1, newHeight);

        if (initialized && this.width == newWidth && this.height == newHeight) {
            return;
        }

        LOGGER.info("[Valkyrie] Resizing/Creating Vulkan-OpenGL Interop resources: {}x{}", newWidth, newHeight);

        destroySharedTexturesAndFbo();

        this.width = newWidth;
        this.height = newHeight;

        VkContext ctx = VkContext.getInstance();
        VkDevice device = ctx.getVkDevice();
        boolean isWindows = Platform.get() == Platform.WINDOWS;

        GLCapabilities glCaps = GL.getCapabilities();
        boolean hasGlExtMem = glCaps.GL_EXT_memory_object && (isWindows ? glCaps.GL_EXT_memory_object_win32 : glCaps.GL_EXT_memory_object_fd);
        boolean hasGlExtSem = glCaps.GL_EXT_semaphore && (isWindows ? glCaps.GL_EXT_semaphore_win32 : glCaps.GL_EXT_semaphore_fd);

        boolean canTryExtMem = ctx.isExternalMemorySupported() && hasGlExtMem && isGlVkDeviceCompatible(ctx, glCaps);
        boolean canTryExtSem = canTryExtMem && ctx.isExternalSemaphoreSupported() && hasGlExtSem;

        glInteropSupported = false;
        glSemaphoreSignaled = false;

        if (canTryExtMem) {
            try {
                clearGlErrors();
                createSharedColorImage(ctx, device, isWindows);
                checkGlError("createSharedColorImage");
                createSharedDepthImage(ctx, device, isWindows);
                checkGlError("createSharedDepthImage");
                if (canTryExtSem) {
                    initSharedSemaphores(device, isWindows);
                    interopMode = InteropMode.ZERO_COPY_SEMAPHORE;
                } else {
                    interopMode = InteropMode.ZERO_COPY_FENCE;
                }
                createSharedFramebuffer();
                checkGlError("createSharedFramebuffer");
                glInteropSupported = true;
            } catch (Throwable t) {
                LOGGER.warn("[Valkyrie] External memory zero-copy setup failed ({}); falling back to staging readback interop.", t.getMessage());
                destroySharedTexturesAndFbo();
                destroyExternalSemaphores(device);
                clearGlErrors();
                glInteropSupported = false;
                interopMode = InteropMode.STAGING_READBACK;
            }
        } else {
            interopMode = InteropMode.STAGING_READBACK;
        }

        if (interopMode == InteropMode.STAGING_READBACK) {
            clearGlErrors();
            createFallbackImagesAndStaging(ctx, device);
            createSharedFramebuffer();
            LOGGER.info("[Valkyrie] Initialized Staging Readback OpenGL Interop ({}x{}).", width, height);
        } else {
            LOGGER.info("[Valkyrie] Vulkan-OpenGL Interop ready ({}x{}, mode={}).", width, height, interopMode);
        }

        // Transition Vulkan images from UNDEFINED to COLOR_ATTACHMENT / DEPTH_ATTACHMENT initial layouts
        transitionInitialLayouts(ctx);

        initialized = true;
    }

    private static void checkGlError(String stage) {
        int glErr = GL11C.glGetError();
        if (glErr != GL11C.GL_NO_ERROR) {
            throw new IllegalStateException("OpenGL error during " + stage + ": 0x" + Integer.toHexString(glErr));
        }
    }

    private static boolean isGlVkDeviceCompatible(VkContext ctx, GLCapabilities glCaps) {
        String glRenderer = GL11C.glGetString(GL11C.GL_RENDERER);
        String vkName = ctx.getDeviceName();
        boolean glIsSoftware = glRenderer != null && (
                glRenderer.toLowerCase(Locale.ROOT).contains("llvmpipe")
                        || glRenderer.toLowerCase(Locale.ROOT).contains("softpipe")
                        || glRenderer.toLowerCase(Locale.ROOT).contains("swrast")
        );
        boolean vkIsSoftware = vkName != null && vkName.toLowerCase(Locale.ROOT).contains("llvmpipe");
        if (glIsSoftware != vkIsSoftware) {
            LOGGER.info("[Valkyrie] OpenGL renderer '{}' and Vulkan GPU '{}' are different adapters; using staging readback interop.",
                    glRenderer, vkName);
            return false;
        }

        byte[] vkUuid = ctx.getDeviceUuid();
        if (vkUuid != null && vkUuid.length == EXTMemoryObject.GL_UUID_SIZE_EXT && glCaps.GL_EXT_memory_object) {
            clearGlErrors();
            int numUuids = GL11C.glGetInteger(EXTMemoryObject.GL_NUM_DEVICE_UUIDS_EXT);
            int err = GL11C.glGetError();
            if (err == GL11C.GL_NO_ERROR && numUuids > 0) {
                boolean matched = false;
                try (MemoryStack stack = MemoryStack.stackPush()) {
                    ByteBuffer glUuidBuf = stack.malloc(EXTMemoryObject.GL_UUID_SIZE_EXT);
                    byte[] glUuid = new byte[EXTMemoryObject.GL_UUID_SIZE_EXT];
                    for (int i = 0; i < numUuids; i++) {
                        glUuidBuf.clear();
                        EXTMemoryObject.glGetUnsignedBytei_vEXT(EXTMemoryObject.GL_DEVICE_UUID_EXT, i, glUuidBuf);
                        glUuidBuf.get(glUuid);
                        if (Arrays.equals(glUuid, vkUuid)) {
                            matched = true;
                            break;
                        }
                    }
                }
                if (!matched) {
                    LOGGER.info("[Valkyrie] OpenGL renderer '{}' device UUID does not match Vulkan GPU '{}'; using staging readback interop.",
                            glRenderer, vkName);
                    return false;
                }
            }
            clearGlErrors();
        }
        return true;
    }

    private static void clearGlErrors() {
        while (GL11C.glGetError() != GL11C.GL_NO_ERROR) {
            // drain prior GL errors
        }
    }

    private void createSharedColorImage(VkContext ctx, VkDevice device, boolean isWindows) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int handleType = isWindows
                    ? VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                    : VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

            VkExternalMemoryImageCreateInfo extImageInfo = VkExternalMemoryImageCreateInfo.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO)
                    .handleTypes(handleType);

            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                    .pNext(extImageInfo.address())
                    .imageType(VK10.VK_IMAGE_TYPE_2D)
                    .format(VK10.VK_FORMAT_R8G8B8A8_UNORM)
                    .extent(e -> e.width(width).height(height).depth(1))
                    .mipLevels(1)
                    .arrayLayers(1)
                    .samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK10.VK_IMAGE_USAGE_STORAGE_BIT |
                            VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK10.VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK10.VK_IMAGE_USAGE_SAMPLED_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED);

            LongBuffer pImage = stack.mallocLong(1);
            int res = VK10.vkCreateImage(device, imageInfo, null, pImage);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create shared Vulkan color image: " + res);
            }
            colorImage = pImage.get(0);

            VkMemoryRequirements2 reqs2 = VkMemoryRequirements2.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2);
            VkMemoryDedicatedRequirements dedicatedReqs = VkMemoryDedicatedRequirements.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS);
            reqs2.pNext(dedicatedReqs.address());
            VkImageMemoryRequirementsInfo2 reqInfo2 = VkImageMemoryRequirementsInfo2.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2)
                    .image(colorImage);
            VK11.vkGetImageMemoryRequirements2(device, reqInfo2, reqs2);
            VkMemoryRequirements memReqs = reqs2.memoryRequirements();

            VkExportMemoryAllocateInfo exportAllocInfo = VkExportMemoryAllocateInfo.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO)
                    .handleTypes(handleType);

            long allocPNext = exportAllocInfo.address();
            if (dedicatedReqs.prefersDedicatedAllocation() || dedicatedReqs.requiresDedicatedAllocation()) {
                VkMemoryDedicatedAllocateInfo dedicatedAlloc = VkMemoryDedicatedAllocateInfo.calloc(stack)
                        .sType(VK11.VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO)
                        .image(colorImage);
                exportAllocInfo.pNext(dedicatedAlloc.address());
            }

            VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                    .pNext(allocPNext)
                    .allocationSize(memReqs.size())
                    .memoryTypeIndex(ctx.findMemoryType(memReqs.memoryTypeBits(), VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

            LongBuffer pMemory = stack.mallocLong(1);
            res = VK10.vkAllocateMemory(device, allocInfo, null, pMemory);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to allocate shared Vulkan color memory: " + res);
            }
            colorMemory = pMemory.get(0);

            VK10.vkBindImageMemory(device, colorImage, colorMemory, 0);

            // Create Vulkan Color ImageView
            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                    .image(colorImage)
                    .viewType(VK10.VK_IMAGE_VIEW_TYPE_2D)
                    .format(VK10.VK_FORMAT_R8G8B8A8_UNORM)
                    .subresourceRange(r -> r
                            .aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .baseMipLevel(0)
                            .levelCount(1)
                            .baseArrayLayer(0)
                            .layerCount(1));

            LongBuffer pView = stack.mallocLong(1);
            res = VK10.vkCreateImageView(device, viewInfo, null, pView);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Vulkan color image view: " + res);
            }
            colorImageView = pView.get(0);

            // Export to OpenGL
            glColorMemObject = EXTMemoryObject.glCreateMemoryObjectsEXT();
            if (isWindows) {
                VkMemoryGetWin32HandleInfoKHR win32HandleInfo = VkMemoryGetWin32HandleInfoKHR.calloc(stack)
                        .sType(KHRExternalMemoryWin32.VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR)
                        .memory(colorMemory)
                        .handleType(VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);

                PointerBuffer pHandle = stack.mallocPointer(1);
                res = KHRExternalMemoryWin32.vkGetMemoryWin32HandleKHR(device, win32HandleInfo, pHandle);
                if (res != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("Failed to export Vulkan color memory Win32 handle: " + res);
                }
                EXTMemoryObjectWin32.glImportMemoryWin32HandleEXT(
                        glColorMemObject,
                        memReqs.size(),
                        EXTMemoryObjectWin32.GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                        pHandle.get(0)
                );
            } else {
                VkMemoryGetFdInfoKHR fdInfo = VkMemoryGetFdInfoKHR.calloc(stack)
                        .sType(KHRExternalMemoryFd.VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR)
                        .memory(colorMemory)
                        .handleType(VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT);

                IntBuffer pFd = stack.mallocInt(1);
                res = KHRExternalMemoryFd.vkGetMemoryFdKHR(device, fdInfo, pFd);
                if (res != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("Failed to export Vulkan color memory FD: " + res);
                }
                EXTMemoryObjectFD.glImportMemoryFdEXT(
                        glColorMemObject,
                        memReqs.size(),
                        EXTMemoryObjectFD.GL_HANDLE_TYPE_OPAQUE_FD_EXT,
                        pFd.get(0)
                );
            }

            // Create OpenGL shared color texture
            glSharedColorTex = GL11C.glGenTextures();
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glSharedColorTex);
            EXTMemoryObject.glTextureStorageMem2DEXT(
                    glSharedColorTex,
                    1,
                    GL11C.GL_RGBA8,
                    width,
                    height,
                    glColorMemObject,
                    0
            );
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MIN_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MAG_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_WRAP_S, GL12C.GL_CLAMP_TO_EDGE);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_WRAP_T, GL12C.GL_CLAMP_TO_EDGE);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, 0);
        }
    }

    private void createSharedDepthImage(VkContext ctx, VkDevice device, boolean isWindows) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int handleType = isWindows
                    ? VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                    : VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

            VkExternalMemoryImageCreateInfo extImageInfo = VkExternalMemoryImageCreateInfo.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO)
                    .handleTypes(handleType);

            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                    .pNext(extImageInfo.address())
                    .imageType(VK10.VK_IMAGE_TYPE_2D)
                    .format(VK10.VK_FORMAT_D32_SFLOAT)
                    .extent(e -> e.width(width).height(height).depth(1))
                    .mipLevels(1)
                    .arrayLayers(1)
                    .samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                            VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK10.VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK10.VK_IMAGE_USAGE_SAMPLED_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED);

            LongBuffer pImage = stack.mallocLong(1);
            int res = VK10.vkCreateImage(device, imageInfo, null, pImage);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create shared Vulkan depth image: " + res);
            }
            depthImage = pImage.get(0);

            VkMemoryRequirements2 reqs2 = VkMemoryRequirements2.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2);
            VkMemoryDedicatedRequirements dedicatedReqs = VkMemoryDedicatedRequirements.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS);
            reqs2.pNext(dedicatedReqs.address());
            VkImageMemoryRequirementsInfo2 reqInfo2 = VkImageMemoryRequirementsInfo2.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2)
                    .image(depthImage);
            VK11.vkGetImageMemoryRequirements2(device, reqInfo2, reqs2);
            VkMemoryRequirements memReqs = reqs2.memoryRequirements();

            VkExportMemoryAllocateInfo exportAllocInfo = VkExportMemoryAllocateInfo.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO)
                    .handleTypes(handleType);

            long allocPNext = exportAllocInfo.address();
            if (dedicatedReqs.prefersDedicatedAllocation() || dedicatedReqs.requiresDedicatedAllocation()) {
                VkMemoryDedicatedAllocateInfo dedicatedAlloc = VkMemoryDedicatedAllocateInfo.calloc(stack)
                        .sType(VK11.VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO)
                        .image(depthImage);
                exportAllocInfo.pNext(dedicatedAlloc.address());
            }

            VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                    .pNext(allocPNext)
                    .allocationSize(memReqs.size())
                    .memoryTypeIndex(ctx.findMemoryType(memReqs.memoryTypeBits(), VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

            LongBuffer pMemory = stack.mallocLong(1);
            res = VK10.vkAllocateMemory(device, allocInfo, null, pMemory);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to allocate shared Vulkan depth memory: " + res);
            }
            depthMemory = pMemory.get(0);

            VK10.vkBindImageMemory(device, depthImage, depthMemory, 0);

            // Create Vulkan Depth ImageView
            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                    .image(depthImage)
                    .viewType(VK10.VK_IMAGE_VIEW_TYPE_2D)
                    .format(VK10.VK_FORMAT_D32_SFLOAT)
                    .subresourceRange(r -> r
                            .aspectMask(VK10.VK_IMAGE_ASPECT_DEPTH_BIT)
                            .baseMipLevel(0)
                            .levelCount(1)
                            .baseArrayLayer(0)
                            .layerCount(1));

            LongBuffer pView = stack.mallocLong(1);
            res = VK10.vkCreateImageView(device, viewInfo, null, pView);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Vulkan depth image view: " + res);
            }
            depthImageView = pView.get(0);

            // Export to OpenGL
            glDepthMemObject = EXTMemoryObject.glCreateMemoryObjectsEXT();
            if (isWindows) {
                VkMemoryGetWin32HandleInfoKHR win32HandleInfo = VkMemoryGetWin32HandleInfoKHR.calloc(stack)
                        .sType(KHRExternalMemoryWin32.VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR)
                        .memory(depthMemory)
                        .handleType(VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);

                PointerBuffer pHandle = stack.mallocPointer(1);
                res = KHRExternalMemoryWin32.vkGetMemoryWin32HandleKHR(device, win32HandleInfo, pHandle);
                if (res != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("Failed to export Vulkan depth memory Win32 handle: " + res);
                }
                EXTMemoryObjectWin32.glImportMemoryWin32HandleEXT(
                        glDepthMemObject,
                        memReqs.size(),
                        EXTMemoryObjectWin32.GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                        pHandle.get(0)
                );
            } else {
                VkMemoryGetFdInfoKHR fdInfo = VkMemoryGetFdInfoKHR.calloc(stack)
                        .sType(KHRExternalMemoryFd.VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR)
                        .memory(depthMemory)
                        .handleType(VK11.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT);

                IntBuffer pFd = stack.mallocInt(1);
                res = KHRExternalMemoryFd.vkGetMemoryFdKHR(device, fdInfo, pFd);
                if (res != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("Failed to export Vulkan depth memory FD: " + res);
                }
                EXTMemoryObjectFD.glImportMemoryFdEXT(
                        glDepthMemObject,
                        memReqs.size(),
                        EXTMemoryObjectFD.GL_HANDLE_TYPE_OPAQUE_FD_EXT,
                        pFd.get(0)
                );
            }

            // Create OpenGL shared depth texture
            glSharedDepthTex = GL11C.glGenTextures();
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glSharedDepthTex);
            EXTMemoryObject.glTextureStorageMem2DEXT(
                    glSharedDepthTex,
                    1,
                    GL30C.GL_DEPTH_COMPONENT32F,
                    width,
                    height,
                    glDepthMemObject,
                    0
            );
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MIN_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MAG_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_WRAP_S, GL12C.GL_CLAMP_TO_EDGE);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_WRAP_T, GL12C.GL_CLAMP_TO_EDGE);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, 0);
        }
    }

    private void initSharedSemaphores(VkDevice device, boolean isWindows) {
        if (vkRenderCompleteSemaphore != 0 && glReadySemaphore != 0) {
            return; // Semaphores persist across window resizes
        }

        try (MemoryStack stack = MemoryStack.stackPush()) {
            int handleType = isWindows
                    ? VK11.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
                    : VK11.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

            VkExportSemaphoreCreateInfo exportInfo = VkExportSemaphoreCreateInfo.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO)
                    .handleTypes(handleType);

            VkSemaphoreCreateInfo semInfo = VkSemaphoreCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)
                    .pNext(exportInfo.address());

            // 1. Create vkRenderCompleteSemaphore (Vulkan signals, OpenGL waits)
            LongBuffer pSem = stack.mallocLong(1);
            int res = VK10.vkCreateSemaphore(device, semInfo, null, pSem);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Vulkan render complete semaphore: " + res);
            }
            vkRenderCompleteSemaphore = pSem.get(0);

            glVkRenderCompleteSemaphore = EXTSemaphore.glGenSemaphoresEXT();
            if (isWindows) {
                VkSemaphoreGetWin32HandleInfoKHR semHandleInfo = VkSemaphoreGetWin32HandleInfoKHR.calloc(stack)
                        .sType(KHRExternalSemaphoreWin32.VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR)
                        .semaphore(vkRenderCompleteSemaphore)
                        .handleType(VK11.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);

                PointerBuffer pHandle = stack.mallocPointer(1);
                res = KHRExternalSemaphoreWin32.vkGetSemaphoreWin32HandleKHR(device, semHandleInfo, pHandle);
                if (res != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("Failed to export Vulkan render complete semaphore Win32 handle: " + res);
                }
                EXTSemaphoreWin32.glImportSemaphoreWin32HandleEXT(
                        glVkRenderCompleteSemaphore,
                        EXTSemaphoreWin32.GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                        pHandle.get(0)
                );
            } else {
                VkSemaphoreGetFdInfoKHR semFdInfo = VkSemaphoreGetFdInfoKHR.calloc(stack)
                        .sType(KHRExternalSemaphoreFd.VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR)
                        .semaphore(vkRenderCompleteSemaphore)
                        .handleType(VK11.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT);

                IntBuffer pFd = stack.mallocInt(1);
                res = KHRExternalSemaphoreFd.vkGetSemaphoreFdKHR(device, semFdInfo, pFd);
                if (res != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("Failed to export Vulkan render complete semaphore FD: " + res);
                }
                EXTSemaphoreFD.glImportSemaphoreFdEXT(
                        glVkRenderCompleteSemaphore,
                        EXTSemaphoreFD.GL_HANDLE_TYPE_OPAQUE_FD_EXT,
                        pFd.get(0)
                );
            }

            // 2. Create glReadySemaphore (OpenGL signals, Vulkan waits)
            res = VK10.vkCreateSemaphore(device, semInfo, null, pSem);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Vulkan GL ready semaphore: " + res);
            }
            vkGlReadySemaphore = pSem.get(0);

            glReadySemaphore = EXTSemaphore.glGenSemaphoresEXT();
            if (isWindows) {
                VkSemaphoreGetWin32HandleInfoKHR semHandleInfo = VkSemaphoreGetWin32HandleInfoKHR.calloc(stack)
                        .sType(KHRExternalSemaphoreWin32.VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR)
                        .semaphore(vkGlReadySemaphore)
                        .handleType(VK11.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);

                PointerBuffer pHandle = stack.mallocPointer(1);
                res = KHRExternalSemaphoreWin32.vkGetSemaphoreWin32HandleKHR(device, semHandleInfo, pHandle);
                if (res != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("Failed to export Vulkan GL ready semaphore Win32 handle: " + res);
                }
                EXTSemaphoreWin32.glImportSemaphoreWin32HandleEXT(
                        glReadySemaphore,
                        EXTSemaphoreWin32.GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                        pHandle.get(0)
                );
            } else {
                VkSemaphoreGetFdInfoKHR semFdInfo = VkSemaphoreGetFdInfoKHR.calloc(stack)
                        .sType(KHRExternalSemaphoreFd.VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR)
                        .semaphore(vkGlReadySemaphore)
                        .handleType(VK11.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT);

                IntBuffer pFd = stack.mallocInt(1);
                res = KHRExternalSemaphoreFd.vkGetSemaphoreFdKHR(device, semFdInfo, pFd);
                if (res != VK10.VK_SUCCESS) {
                    throw new IllegalStateException("Failed to export Vulkan GL ready semaphore FD: " + res);
                }
                EXTSemaphoreFD.glImportSemaphoreFdEXT(
                        glReadySemaphore,
                        EXTSemaphoreFD.GL_HANDLE_TYPE_OPAQUE_FD_EXT,
                        pFd.get(0)
                );
            }
        }
    }

    private void createSharedFramebuffer() {
        glSharedFbo = GL30C.glGenFramebuffers();
        GL30C.glBindFramebuffer(GL30C.GL_FRAMEBUFFER, glSharedFbo);
        GL30C.glFramebufferTexture2D(GL30C.GL_FRAMEBUFFER, GL30C.GL_COLOR_ATTACHMENT0, GL11C.GL_TEXTURE_2D, glSharedColorTex, 0);
        GL30C.glFramebufferTexture2D(GL30C.GL_FRAMEBUFFER, GL30C.GL_DEPTH_ATTACHMENT, GL11C.GL_TEXTURE_2D, glSharedDepthTex, 0);

        int status = GL30C.glCheckFramebufferStatus(GL30C.GL_FRAMEBUFFER);
        if (status != GL30C.GL_FRAMEBUFFER_COMPLETE) {
            throw new IllegalStateException("Shared OpenGL Framebuffer is incomplete, status: " + status);
        }
        GL30C.glBindFramebuffer(GL30C.GL_FRAMEBUFFER, 0);
    }

    private void transitionInitialLayouts(VkContext ctx) {
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageMemoryBarrier.Buffer barriers = VkImageMemoryBarrier.calloc(2, stack);

            // Barrier for color image: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
            barriers.get(0)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED)
                    .newLayout(VK10.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                    .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                    .image(colorImage)
                    .subresourceRange(r -> r
                            .aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .baseMipLevel(0)
                            .levelCount(1)
                            .baseArrayLayer(0)
                            .layerCount(1))
                    .srcAccessMask(0)
                    .dstAccessMask(VK10.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

            // Barrier for depth image: UNDEFINED -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            barriers.get(1)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                    .oldLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED)
                    .newLayout(VK10.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                    .image(depthImage)
                    .subresourceRange(r -> r
                            .aspectMask(VK10.VK_IMAGE_ASPECT_DEPTH_BIT)
                            .baseMipLevel(0)
                            .levelCount(1)
                            .baseArrayLayer(0)
                            .layerCount(1))
                    .srcAccessMask(0)
                    .dstAccessMask(VK10.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            VK10.vkCmdPipelineBarrier(
                    cmd,
                    VK10.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK10.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK10.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                    0,
                    null,
                    null,
                    barriers
            );
        }
        ctx.endSingleTimeCommands(cmd);
    }

    private void createFallbackImagesAndStaging(VkContext ctx, VkDevice device) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Color VkImage + View
            VkImageCreateInfo colorInfo = VkImageCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                    .imageType(VK10.VK_IMAGE_TYPE_2D)
                    .format(VK10.VK_FORMAT_R8G8B8A8_UNORM)
                    .extent(e -> e.width(width).height(height).depth(1))
                    .mipLevels(1).arrayLayers(1)
                    .samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK10.VK_IMAGE_USAGE_STORAGE_BIT
                            | VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK10.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK10.VK_IMAGE_USAGE_SAMPLED_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED);
            LongBuffer pImg = stack.mallocLong(1);
            if (VK10.vkCreateImage(device, colorInfo, null, pImg) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create fallback Vulkan color image");
            }
            colorImage = pImg.get(0);

            VkMemoryRequirements reqs = VkMemoryRequirements.malloc(stack);
            VK10.vkGetImageMemoryRequirements(device, colorImage, reqs);
            VkMemoryAllocateInfo alloc = VkMemoryAllocateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                    .allocationSize(reqs.size())
                    .memoryTypeIndex(ctx.findMemoryType(reqs.memoryTypeBits(), VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
            LongBuffer pMem = stack.mallocLong(1);
            if (VK10.vkAllocateMemory(device, alloc, null, pMem) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to allocate fallback Vulkan color memory");
            }
            colorMemory = pMem.get(0);
            VK10.vkBindImageMemory(device, colorImage, colorMemory, 0);

            VkImageViewCreateInfo cViewInfo = VkImageViewCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                    .image(colorImage).viewType(VK10.VK_IMAGE_VIEW_TYPE_2D).format(VK10.VK_FORMAT_R8G8B8A8_UNORM)
                    .subresourceRange(r -> r.aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1));
            LongBuffer pView = stack.mallocLong(1);
            VK10.vkCreateImageView(device, cViewInfo, null, pView);
            colorImageView = pView.get(0);

            // Depth VkImage + View
            VkImageCreateInfo depthInfo = VkImageCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                    .imageType(VK10.VK_IMAGE_TYPE_2D)
                    .format(VK10.VK_FORMAT_D32_SFLOAT)
                    .extent(e -> e.width(width).height(height).depth(1))
                    .mipLevels(1).arrayLayers(1)
                    .samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | VK10.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK10.VK_IMAGE_USAGE_SAMPLED_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED);
            if (VK10.vkCreateImage(device, depthInfo, null, pImg) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create fallback Vulkan depth image");
            }
            depthImage = pImg.get(0);

            VK10.vkGetImageMemoryRequirements(device, depthImage, reqs);
            alloc.allocationSize(reqs.size())
                    .memoryTypeIndex(ctx.findMemoryType(reqs.memoryTypeBits(), VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
            if (VK10.vkAllocateMemory(device, alloc, null, pMem) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to allocate fallback Vulkan depth memory");
            }
            depthMemory = pMem.get(0);
            VK10.vkBindImageMemory(device, depthImage, depthMemory, 0);

            VkImageViewCreateInfo dViewInfo = VkImageViewCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                    .image(depthImage).viewType(VK10.VK_IMAGE_VIEW_TYPE_2D).format(VK10.VK_FORMAT_D32_SFLOAT)
                    .subresourceRange(r -> r.aspectMask(VK10.VK_IMAGE_ASPECT_DEPTH_BIT).levelCount(1).layerCount(1));
            VK10.vkCreateImageView(device, dViewInfo, null, pView);
            depthImageView = pView.get(0);

            // Host-visible coherent readback staging buffer
            stagingColorBytes = width * height * 4;
            stagingTotalBytes = stagingColorBytes * 2; // RGBA8 (4B/px) + D32F (4B/px)
            VkBufferCreateInfo bufInfo = VkBufferCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(stagingTotalBytes)
                    .usage(VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer pBuf = stack.mallocLong(1);
            if (VK10.vkCreateBuffer(device, bufInfo, null, pBuf) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create staging readback buffer");
            }
            stagingBuffer = pBuf.get(0);

            VK10.vkGetBufferMemoryRequirements(device, stagingBuffer, reqs);
            alloc.allocationSize(reqs.size())
                    .memoryTypeIndex(ctx.findMemoryType(reqs.memoryTypeBits(),
                            VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
            if (VK10.vkAllocateMemory(device, alloc, null, pMem) != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to allocate staging readback memory");
            }
            stagingMemory = pMem.get(0);
            VK10.vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

            PointerBuffer ppMap = stack.mallocPointer(1);
            VK10.vkMapMemory(device, stagingMemory, 0, stagingTotalBytes, 0, ppMap);
            stagingMappedAddr = ppMap.get(0);

            // Standard OpenGL textures for staging upload + shared FBO blit
            int prevTex = GL11C.glGetInteger(GL11C.GL_TEXTURE_BINDING_2D);

            glSharedColorTex = GL11C.glGenTextures();
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glSharedColorTex);
            GL11C.glTexImage2D(GL11C.GL_TEXTURE_2D, 0, GL11C.GL_RGBA8, width, height, 0,
                    GL11C.GL_RGBA, GL11C.GL_UNSIGNED_BYTE, 0L);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MIN_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MAG_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_WRAP_S, GL12C.GL_CLAMP_TO_EDGE);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_WRAP_T, GL12C.GL_CLAMP_TO_EDGE);

            glSharedDepthTex = GL11C.glGenTextures();
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glSharedDepthTex);
            GL11C.glTexImage2D(GL11C.GL_TEXTURE_2D, 0, GL30C.GL_DEPTH_COMPONENT32F, width, height, 0,
                    GL11C.GL_DEPTH_COMPONENT, GL11C.GL_FLOAT, 0L);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MIN_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MAG_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_WRAP_S, GL12C.GL_CLAMP_TO_EDGE);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_WRAP_T, GL12C.GL_CLAMP_TO_EDGE);

            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, prevTex);
        }
    }

    /**
     * Composites the shared Vulkan color and depth buffer into Minecraft's main RenderTarget.
     */
    public void compositeToMainTarget(RenderTarget mainTarget) {
        if (mainTarget == null) {
            return;
        }
        compositeToMainTarget(mainTarget.frameBufferId, mainTarget.width, mainTarget.height);
    }

    /**
     * Composites the shared Vulkan color and depth buffer directly into the target OpenGL Framebuffer.
     */
    public void compositeToMainTarget(int targetFboId, int targetWidth, int targetHeight) {
        if (!initialized || glSharedFbo == 0) {
            return;
        }

        if (interopMode == InteropMode.ZERO_COPY_SEMAPHORE) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                IntBuffer textures = stack.ints(glSharedColorTex, glSharedDepthTex);
                IntBuffer layouts = stack.ints(
                        vkLayoutToGlExtLayout(vkExportedColorLayout),
                        vkLayoutToGlExtLayout(vkExportedDepthLayout)
                );
                EXTSemaphore.glWaitSemaphoreEXT(glVkRenderCompleteSemaphore, stack.ints(), textures, layouts);
            }
        } else if (interopMode == InteropMode.STAGING_READBACK) {
            VkContext ctx = VkContext.getInstance();
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkBufferImageCopy.Buffer colorCopy = VkBufferImageCopy.calloc(1, stack)
                        .bufferOffset(0).bufferRowLength(0).bufferImageHeight(0);
                colorCopy.imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
                colorCopy.imageExtent().set(width, height, 1);
                VK10.vkCmdCopyImageToBuffer(cmd, colorImage, VK10.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, colorCopy);

                VkBufferImageCopy.Buffer depthCopy = VkBufferImageCopy.calloc(1, stack)
                        .bufferOffset(stagingColorBytes).bufferRowLength(0).bufferImageHeight(0);
                depthCopy.imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_DEPTH_BIT).layerCount(1);
                depthCopy.imageExtent().set(width, height, 1);
                VK10.vkCmdCopyImageToBuffer(cmd, depthImage, VK10.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, depthCopy);
            }
            ctx.endSingleTimeCommands(cmd);

            int prevActiveTex = GL11C.glGetInteger(GL13C.GL_ACTIVE_TEXTURE);
            GL13C.glActiveTexture(GL13C.GL_TEXTURE0);
            int prevTex = GL11C.glGetInteger(GL11C.GL_TEXTURE_BINDING_2D);
            int prevUnpackBuffer = GL11C.glGetInteger(GL21C.GL_PIXEL_UNPACK_BUFFER_BINDING);
            int prevUnpackAlign = GL11C.glGetInteger(GL11C.GL_UNPACK_ALIGNMENT);
            int prevUnpackRow = GL11C.glGetInteger(GL11C.GL_UNPACK_ROW_LENGTH);
            if (prevUnpackBuffer != 0) {
                GL15C.glBindBuffer(GL21C.GL_PIXEL_UNPACK_BUFFER, 0);
            }
            GL11C.glPixelStorei(GL11C.GL_UNPACK_ALIGNMENT, 4);
            GL11C.glPixelStorei(GL11C.GL_UNPACK_ROW_LENGTH, 0);

            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glSharedColorTex);
            GL11C.glTexSubImage2D(GL11C.GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL11C.GL_RGBA, GL11C.GL_UNSIGNED_BYTE, stagingMappedAddr);

            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glSharedDepthTex);
            GL11C.glTexSubImage2D(GL11C.GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL11C.GL_DEPTH_COMPONENT, GL11C.GL_FLOAT, stagingMappedAddr + stagingColorBytes);

            GL11C.glPixelStorei(GL11C.GL_UNPACK_ALIGNMENT, prevUnpackAlign);
            GL11C.glPixelStorei(GL11C.GL_UNPACK_ROW_LENGTH, prevUnpackRow);
            if (prevUnpackBuffer != 0) {
                GL15C.glBindBuffer(GL21C.GL_PIXEL_UNPACK_BUFFER, prevUnpackBuffer);
            }
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, prevTex);
            GL13C.glActiveTexture(prevActiveTex);
        }

        try {
            // 2. Save previous OpenGL FBO / scissor state cleanly
            int prevReadFbo = GL11C.glGetInteger(GL30C.GL_READ_FRAMEBUFFER_BINDING);
            int prevDrawFbo = GL11C.glGetInteger(GL30C.GL_DRAW_FRAMEBUFFER_BINDING);
            boolean scissorWasEnabled = GL11C.glIsEnabled(GL11C.GL_SCISSOR_TEST);
            if (scissorWasEnabled) {
                GL11C.glDisable(GL11C.GL_SCISSOR_TEST);
            }

            // 3. Blit color buffer (format conversion allowed by OpenGL)
            if (GL.getCapabilities().OpenGL45) {
                GL45C.glBlitNamedFramebuffer(
                        glSharedFbo,
                        targetFboId,
                        0, 0, width, height,
                        0, 0, targetWidth, targetHeight,
                        GL11C.GL_COLOR_BUFFER_BIT,
                        GL11C.GL_NEAREST
                );
            } else {
                GL30C.glBindFramebuffer(GL30C.GL_READ_FRAMEBUFFER, glSharedFbo);
                GL30C.glBindFramebuffer(GL30C.GL_DRAW_FRAMEBUFFER, targetFboId);
                GL30C.glBlitFramebuffer(
                        0, 0, width, height,
                        0, 0, targetWidth, targetHeight,
                        GL11C.GL_COLOR_BUFFER_BIT,
                        GL11C.GL_NEAREST
                );
            }

            // 4. Copy depth buffer via fullscreen shader so any target depth format (D24, D24S8, D32F) works
            copyDepthToTargetFbo(targetFboId, targetWidth, targetHeight);

            // 4b. Composite Distant Horizons LODs behind Valkyrie terrain (into sky pixels) if DH rendered this frame
            compositeDhLodsToTargetFbo(targetFboId, targetWidth, targetHeight);

            // 5. Restore OpenGL FBO / scissor state cleanly
            if (scissorWasEnabled) {
                GL11C.glEnable(GL11C.GL_SCISSOR_TEST);
            }
            GL30C.glBindFramebuffer(GL30C.GL_READ_FRAMEBUFFER, prevReadFbo);
            GL30C.glBindFramebuffer(GL30C.GL_DRAW_FRAMEBUFFER, prevDrawFbo);
        } finally {
            // 6. Always synchronize back to Vulkan so a failed/partial GL composite cannot stall the next VK submit
            if (interopMode == InteropMode.ZERO_COPY_SEMAPHORE) {
                try (MemoryStack stack = MemoryStack.stackPush()) {
                    IntBuffer textures = stack.ints(glSharedColorTex, glSharedDepthTex);
                    IntBuffer layouts = stack.ints(EXTSemaphore.GL_LAYOUT_GENERAL_EXT, EXTSemaphore.GL_LAYOUT_GENERAL_EXT);
                    EXTSemaphore.glSignalSemaphoreEXT(glReadySemaphore, stack.ints(), textures, layouts);
                    glSemaphoreSignaled = true;
                }
            } else if (interopMode == InteropMode.ZERO_COPY_FENCE) {
                GL11C.glFinish();
            }
        }
    }

    private void ensureFullscreenVao() {
        if (glDepthCopyVao == 0) {
            glDepthCopyVao = GL30C.glGenVertexArrays();
        }
    }

    private int linkFullscreenProgram(String fragSrc, String errorLabel) {
        String vertSrc = """
                #version 330 core
                out vec2 vUv;
                void main() {
                    vec2 pos = vec2((gl_VertexID & 1) * 4 - 1, (gl_VertexID & 2) * 2 - 1);
                    vUv = pos * 0.5 + 0.5;
                    gl_Position = vec4(pos, 0.0, 1.0);
                }
                """;
        int vert = compileGlShader(GL20C.GL_VERTEX_SHADER, vertSrc);
        int frag = compileGlShader(GL20C.GL_FRAGMENT_SHADER, fragSrc);
        int prog = GL20C.glCreateProgram();
        GL20C.glAttachShader(prog, vert);
        GL20C.glAttachShader(prog, frag);
        GL20C.glLinkProgram(prog);
        GL20C.glDeleteShader(vert);
        GL20C.glDeleteShader(frag);
        if (GL20C.glGetProgrami(prog, GL20C.GL_LINK_STATUS) == GL11C.GL_FALSE) {
            String log = GL20C.glGetProgramInfoLog(prog);
            GL20C.glDeleteProgram(prog);
            throw new IllegalStateException("Failed to link OpenGL " + errorLabel + " program: " + log);
        }
        return prog;
    }

    private void ensureDepthCopyResources() {
        if (glDepthCopyProgram != 0) {
            return;
        }
        String fragSrc = """
                #version 330 core
                uniform sampler2D uDepthTex;
                in vec2 vUv;
                void main() {
                    gl_FragDepth = texture(uDepthTex, vUv).r;
                }
                """;
        glDepthCopyProgram = linkFullscreenProgram(fragSrc, "depth copy");
        glDepthCopyTexLoc = GL20C.glGetUniformLocation(glDepthCopyProgram, "uDepthTex");
        ensureFullscreenVao();
    }

    private void ensureDhCompositeResources() {
        if (glDhCompositeProgram != 0) {
            return;
        }
        String fragSrc = """
                #version 330 core
                uniform sampler2D uMcDepthTex;
                uniform sampler2D uDhColorTex;
                uniform sampler2D uDhDepthTex;
                in vec2 vUv;
                out vec4 fragColor;
                void main() {
                    float mcDepth = texture(uMcDepthTex, vUv).r;
                    if (mcDepth < 0.999999) {
                        discard;
                    }
                    float dhDepth = texture(uDhDepthTex, vUv).r;
                    if (dhDepth >= 0.999999) {
                        discard;
                    }
                    vec4 dhColor = texture(uDhColorTex, vUv);
                    if (dhColor.a <= 0.001) {
                        discard;
                    }
                    fragColor = dhColor;
                }
                """;
        glDhCompositeProgram = linkFullscreenProgram(fragSrc, "DH composite");
        glDhMcDepthLoc = GL20C.glGetUniformLocation(glDhCompositeProgram, "uMcDepthTex");
        glDhColorLoc = GL20C.glGetUniformLocation(glDhCompositeProgram, "uDhColorTex");
        glDhDepthLoc = GL20C.glGetUniformLocation(glDhCompositeProgram, "uDhDepthTex");
        ensureFullscreenVao();
    }

    private static int compileGlShader(int type, String src) {
        int shader = GL20C.glCreateShader(type);
        GL20C.glShaderSource(shader, src);
        GL20C.glCompileShader(shader);
        if (GL20C.glGetShaderi(shader, GL20C.GL_COMPILE_STATUS) == GL11C.GL_FALSE) {
            String log = GL20C.glGetShaderInfoLog(shader);
            GL20C.glDeleteShader(shader);
            throw new IllegalStateException("Failed to compile OpenGL depth copy shader: " + log);
        }
        return shader;
    }

    private void copyDepthToTargetFbo(int targetFboId, int targetWidth, int targetHeight) {
        if (glSharedDepthTex == 0) {
            return;
        }
        ensureDepthCopyResources();

        boolean depthMaskPrev = GL11C.glGetBoolean(GL11C.GL_DEPTH_WRITEMASK);
        boolean depthTestPrev = GL11C.glIsEnabled(GL11C.GL_DEPTH_TEST);
        int depthFuncPrev = GL11C.glGetInteger(GL11C.GL_DEPTH_FUNC);
        boolean cullPrev = GL11C.glIsEnabled(GL11C.GL_CULL_FACE);
        boolean blendPrev = GL11C.glIsEnabled(GL11C.GL_BLEND);
        boolean stencilPrev = GL11C.glIsEnabled(GL11C.GL_STENCIL_TEST);
        int programPrev = GL11C.glGetInteger(GL20C.GL_CURRENT_PROGRAM);
        int vaoPrev = GL11C.glGetInteger(GL30C.GL_VERTEX_ARRAY_BINDING);
        int activeTexPrev = GL11C.glGetInteger(GL13C.GL_ACTIVE_TEXTURE);

        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer vpPrev = stack.mallocInt(4);
            GL11C.glGetIntegerv(GL11C.GL_VIEWPORT, vpPrev);
            ByteBuffer colorMaskPrev = stack.malloc(4);
            GL11C.glGetBooleanv(GL11C.GL_COLOR_WRITEMASK, colorMaskPrev);

            GL13C.glActiveTexture(GL13C.GL_TEXTURE0);
            int tex0Prev = GL11C.glGetInteger(GL11C.GL_TEXTURE_BINDING_2D);
            int sampler0Prev = GL11C.glGetInteger(GL33C.GL_SAMPLER_BINDING);

            GL30C.glBindFramebuffer(GL30C.GL_DRAW_FRAMEBUFFER, targetFboId);
            GL11C.glViewport(0, 0, targetWidth, targetHeight);
            GL11C.glColorMask(false, false, false, false);
            GL11C.glDepthMask(true);
            GL11C.glEnable(GL11C.GL_DEPTH_TEST);
            GL11C.glDepthFunc(GL11C.GL_ALWAYS);
            GL11C.glDisable(GL11C.GL_CULL_FACE);
            GL11C.glDisable(GL11C.GL_BLEND);
            GL11C.glDisable(GL11C.GL_STENCIL_TEST);

            GL20C.glUseProgram(glDepthCopyProgram);
            GL30C.glBindVertexArray(glDepthCopyVao);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glSharedDepthTex);
            GL33C.glBindSampler(0, 0);
            GL20C.glUniform1i(glDepthCopyTexLoc, 0);
            GL11C.glDrawArrays(GL11C.GL_TRIANGLES, 0, 3);

            GL33C.glBindSampler(0, sampler0Prev);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, tex0Prev);
            GL11C.glColorMask(colorMaskPrev.get(0) != 0, colorMaskPrev.get(1) != 0, colorMaskPrev.get(2) != 0, colorMaskPrev.get(3) != 0);
            GL11C.glViewport(vpPrev.get(0), vpPrev.get(1), vpPrev.get(2), vpPrev.get(3));
        }

        GL13C.glActiveTexture(activeTexPrev);
        GL30C.glBindVertexArray(vaoPrev);
        GL20C.glUseProgram(programPrev);
        GL11C.glDepthMask(depthMaskPrev);
        GL11C.glDepthFunc(depthFuncPrev);
        if (!depthTestPrev) GL11C.glDisable(GL11C.GL_DEPTH_TEST);
        if (cullPrev) GL11C.glEnable(GL11C.GL_CULL_FACE);
        if (blendPrev) GL11C.glEnable(GL11C.GL_BLEND);
        if (stencilPrev) GL11C.glEnable(GL11C.GL_STENCIL_TEST);
    }

    private void compositeDhLodsToTargetFbo(int targetFboId, int targetWidth, int targetHeight) {
        if (glSharedDepthTex == 0) {
            return;
        }
        int[] dhTex = new int[2];
        if (!org.sakus.aerogelvk.compat.CompatManager.consumeDhFrameTextures(dhTex)) {
            return;
        }
        int dhColorTex = dhTex[0];
        int dhDepthTex = dhTex[1];
        if (dhColorTex <= 0 || dhDepthTex <= 0) {
            return;
        }

        ensureDhCompositeResources();

        boolean depthMaskPrev = GL11C.glGetBoolean(GL11C.GL_DEPTH_WRITEMASK);
        boolean depthTestPrev = GL11C.glIsEnabled(GL11C.GL_DEPTH_TEST);
        boolean cullPrev = GL11C.glIsEnabled(GL11C.GL_CULL_FACE);
        boolean blendPrev = GL11C.glIsEnabled(GL11C.GL_BLEND);
        int blendSrcRgbPrev = GL11C.glGetInteger(GL14C.GL_BLEND_SRC_RGB);
        int blendDstRgbPrev = GL11C.glGetInteger(GL14C.GL_BLEND_DST_RGB);
        int blendSrcAlphaPrev = GL11C.glGetInteger(GL14C.GL_BLEND_SRC_ALPHA);
        int blendDstAlphaPrev = GL11C.glGetInteger(GL14C.GL_BLEND_DST_ALPHA);
        boolean stencilPrev = GL11C.glIsEnabled(GL11C.GL_STENCIL_TEST);
        int programPrev = GL11C.glGetInteger(GL20C.GL_CURRENT_PROGRAM);
        int vaoPrev = GL11C.glGetInteger(GL30C.GL_VERTEX_ARRAY_BINDING);
        int activeTexPrev = GL11C.glGetInteger(GL13C.GL_ACTIVE_TEXTURE);

        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer vpPrev = stack.mallocInt(4);
            GL11C.glGetIntegerv(GL11C.GL_VIEWPORT, vpPrev);
            ByteBuffer colorMaskPrev = stack.malloc(4);
            GL11C.glGetBooleanv(GL11C.GL_COLOR_WRITEMASK, colorMaskPrev);

            GL13C.glActiveTexture(GL13C.GL_TEXTURE0);
            int tex0Prev = GL11C.glGetInteger(GL11C.GL_TEXTURE_BINDING_2D);
            int sampler0Prev = GL11C.glGetInteger(GL33C.GL_SAMPLER_BINDING);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glSharedDepthTex);
            GL33C.glBindSampler(0, 0);

            GL13C.glActiveTexture(GL13C.GL_TEXTURE1);
            int tex1Prev = GL11C.glGetInteger(GL11C.GL_TEXTURE_BINDING_2D);
            int sampler1Prev = GL11C.glGetInteger(GL33C.GL_SAMPLER_BINDING);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, dhColorTex);
            GL33C.glBindSampler(1, 0);

            GL13C.glActiveTexture(GL13C.GL_TEXTURE2);
            int tex2Prev = GL11C.glGetInteger(GL11C.GL_TEXTURE_BINDING_2D);
            int sampler2Prev = GL11C.glGetInteger(GL33C.GL_SAMPLER_BINDING);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, dhDepthTex);
            GL33C.glBindSampler(2, 0);

            GL30C.glBindFramebuffer(GL30C.GL_DRAW_FRAMEBUFFER, targetFboId);
            GL11C.glViewport(0, 0, targetWidth, targetHeight);
            GL11C.glColorMask(true, true, true, true);
            GL11C.glDepthMask(false);
            GL11C.glDisable(GL11C.GL_DEPTH_TEST);
            GL11C.glDisable(GL11C.GL_CULL_FACE);
            GL11C.glDisable(GL11C.GL_STENCIL_TEST);
            GL11C.glEnable(GL11C.GL_BLEND);
            GL14C.glBlendFuncSeparate(GL11C.GL_SRC_ALPHA, GL11C.GL_ONE_MINUS_SRC_ALPHA, GL11C.GL_ONE, GL11C.GL_ONE_MINUS_SRC_ALPHA);

            GL20C.glUseProgram(glDhCompositeProgram);
            GL30C.glBindVertexArray(glDepthCopyVao);
            GL20C.glUniform1i(glDhMcDepthLoc, 0);
            GL20C.glUniform1i(glDhColorLoc, 1);
            GL20C.glUniform1i(glDhDepthLoc, 2);
            GL11C.glDrawArrays(GL11C.GL_TRIANGLES, 0, 3);

            GL13C.glActiveTexture(GL13C.GL_TEXTURE2);
            GL33C.glBindSampler(2, sampler2Prev);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, tex2Prev);
            GL13C.glActiveTexture(GL13C.GL_TEXTURE1);
            GL33C.glBindSampler(1, sampler1Prev);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, tex1Prev);
            GL13C.glActiveTexture(GL13C.GL_TEXTURE0);
            GL33C.glBindSampler(0, sampler0Prev);
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, tex0Prev);
            GL11C.glColorMask(colorMaskPrev.get(0) != 0, colorMaskPrev.get(1) != 0, colorMaskPrev.get(2) != 0, colorMaskPrev.get(3) != 0);
            GL11C.glViewport(vpPrev.get(0), vpPrev.get(1), vpPrev.get(2), vpPrev.get(3));
        }

        GL13C.glActiveTexture(activeTexPrev);
        GL30C.glBindVertexArray(vaoPrev);
        GL20C.glUseProgram(programPrev);
        GL14C.glBlendFuncSeparate(blendSrcRgbPrev, blendDstRgbPrev, blendSrcAlphaPrev, blendDstAlphaPrev);
        if (!blendPrev) GL11C.glDisable(GL11C.GL_BLEND);
        GL11C.glDepthMask(depthMaskPrev);
        if (depthTestPrev) GL11C.glEnable(GL11C.GL_DEPTH_TEST);
        if (cullPrev) GL11C.glEnable(GL11C.GL_CULL_FACE);
        if (stencilPrev) GL11C.glEnable(GL11C.GL_STENCIL_TEST);
    }

    public synchronized boolean consumeGlSemaphoreSignaled() {
        boolean wasSignaled = glSemaphoreSignaled;
        glSemaphoreSignaled = false;
        return wasSignaled;
    }

    public synchronized boolean isGlSemaphoreSignaled() {
        return glSemaphoreSignaled;
    }

    private volatile int vkExportedColorLayout = VK10.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    private volatile int vkExportedDepthLayout = VK10.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    public void setVkExportedLayouts(int colorLayout, int depthLayout) {
        this.vkExportedColorLayout = colorLayout;
        this.vkExportedDepthLayout = depthLayout;
    }

    private static int vkLayoutToGlExtLayout(int vkLayout) {
        return switch (vkLayout) {
            case VK10.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL -> EXTSemaphore.GL_LAYOUT_COLOR_ATTACHMENT_EXT;
            case VK10.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> EXTSemaphore.GL_LAYOUT_DEPTH_STENCIL_ATTACHMENT_EXT;
            case VK10.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL -> EXTSemaphore.GL_LAYOUT_SHADER_READ_ONLY_EXT;
            case VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL -> EXTSemaphore.GL_LAYOUT_TRANSFER_DST_EXT;
            case VK10.VK_IMAGE_LAYOUT_GENERAL -> EXTSemaphore.GL_LAYOUT_GENERAL_EXT;
            default -> EXTSemaphore.GL_LAYOUT_TRANSFER_SRC_EXT;
        };
    }

    public boolean usesExternalSemaphores() {
        return interopMode == InteropMode.ZERO_COPY_SEMAPHORE;
    }

    public InteropMode getInteropMode() {
        return interopMode;
    }

    public boolean isGlInteropSupported() {
        return glInteropSupported;
    }

    private static boolean hasCurrentGlContext() {
        try {
            return GL.getCapabilities() != null && GLFW.glfwGetCurrentContext() != 0L;
        } catch (Throwable ignored) {
            return false;
        }
    }

    public synchronized void destroySharedTexturesAndFbo() {
        VkDevice device = VkContext.getInstance().getVkDevice();
        if (device != null) {
            VK10.vkDeviceWaitIdle(device);
        }
        boolean glReady = hasCurrentGlContext();

        if (glReady) {
            if (glSharedFbo != 0) {
                GL30C.glDeleteFramebuffers(glSharedFbo);
            }
            if (glSharedColorTex != 0) {
                GL11C.glDeleteTextures(glSharedColorTex);
            }
            if (glColorMemObject != 0) {
                EXTMemoryObject.glDeleteMemoryObjectsEXT(glColorMemObject);
            }
            if (glSharedDepthTex != 0) {
                GL11C.glDeleteTextures(glSharedDepthTex);
            }
            if (glDepthMemObject != 0) {
                EXTMemoryObject.glDeleteMemoryObjectsEXT(glDepthMemObject);
            }
        }
        glSharedFbo = 0;
        glSharedColorTex = 0;
        glColorMemObject = 0;
        glSharedDepthTex = 0;
        glDepthMemObject = 0;

        if (device != null) {
            if (stagingMappedAddr != 0 && stagingMemory != 0) {
                VK10.vkUnmapMemory(device, stagingMemory);
                stagingMappedAddr = 0;
            }
            if (stagingBuffer != 0) {
                VK10.vkDestroyBuffer(device, stagingBuffer, null);
                stagingBuffer = 0;
            }
            if (stagingMemory != 0) {
                VK10.vkFreeMemory(device, stagingMemory, null);
                stagingMemory = 0;
            }
            stagingColorBytes = 0;
            stagingTotalBytes = 0;

            if (colorImageView != 0) {
                VK10.vkDestroyImageView(device, colorImageView, null);
                colorImageView = 0;
            }
            if (colorImage != 0) {
                VK10.vkDestroyImage(device, colorImage, null);
                colorImage = 0;
            }
            if (colorMemory != 0) {
                VK10.vkFreeMemory(device, colorMemory, null);
                colorMemory = 0;
            }

            if (depthImageView != 0) {
                VK10.vkDestroyImageView(device, depthImageView, null);
                depthImageView = 0;
            }
            if (depthImage != 0) {
                VK10.vkDestroyImage(device, depthImage, null);
                depthImage = 0;
            }
            if (depthMemory != 0) {
                VK10.vkFreeMemory(device, depthMemory, null);
                depthMemory = 0;
            }
        }

        initialized = false;
    }

    private void destroyExternalSemaphores(VkDevice device) {
        if (hasCurrentGlContext()) {
            if (glVkRenderCompleteSemaphore != 0) {
                EXTSemaphore.glDeleteSemaphoresEXT(glVkRenderCompleteSemaphore);
            }
            if (glReadySemaphore != 0) {
                EXTSemaphore.glDeleteSemaphoresEXT(glReadySemaphore);
            }
        }
        glVkRenderCompleteSemaphore = 0;
        glReadySemaphore = 0;

        if (device != null) {
            if (vkRenderCompleteSemaphore != 0) {
                VK10.vkDestroySemaphore(device, vkRenderCompleteSemaphore, null);
                vkRenderCompleteSemaphore = 0;
            }
            if (vkGlReadySemaphore != 0) {
                VK10.vkDestroySemaphore(device, vkGlReadySemaphore, null);
                vkGlReadySemaphore = 0;
            }
        }
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public long getColorImage() {
        return colorImage;
    }

    public long getSharedColorImage() {
        return colorImage;
    }

    public long getColorImageView() {
        return colorImageView;
    }

    public long getSharedColorImageView() {
        return colorImageView;
    }

    public int getGlSharedColorTex() {
        return glSharedColorTex;
    }

    public long getDepthImage() {
        return depthImage;
    }

    public long getSharedDepthImage() {
        return depthImage;
    }

    public long getDepthImageView() {
        return depthImageView;
    }

    public long getSharedDepthImageView() {
        return depthImageView;
    }

    public int getGlSharedDepthTex() {
        return glSharedDepthTex;
    }

    public int getGlSharedFbo() {
        return glSharedFbo;
    }

    public long getVkRenderCompleteSemaphore() {
        return vkRenderCompleteSemaphore;
    }

    public int getGlVkRenderCompleteSemaphore() {
        return glVkRenderCompleteSemaphore;
    }

    public long getVkGlReadySemaphore() {
        return vkGlReadySemaphore;
    }

    public int getGlReadySemaphore() {
        return glReadySemaphore;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public synchronized void cleanup() {
        close();
    }

    @Override
    public synchronized void close() {
        destroySharedTexturesAndFbo();
        if (hasCurrentGlContext()) {
            if (glDepthCopyProgram != 0) {
                GL20C.glDeleteProgram(glDepthCopyProgram);
            }
            if (glDhCompositeProgram != 0) {
                GL20C.glDeleteProgram(glDhCompositeProgram);
            }
            if (glDepthCopyVao != 0) {
                GL30C.glDeleteVertexArrays(glDepthCopyVao);
            }
        }
        glDepthCopyProgram = 0;
        glDepthCopyVao = 0;
        glDepthCopyTexLoc = -1;
        glDhCompositeProgram = 0;
        glDhMcDepthLoc = -1;
        glDhColorLoc = -1;
        glDhDepthLoc = -1;
        VkDevice device = VkContext.getInstance().getVkDevice();
        destroyExternalSemaphores(device);

        LOGGER.info("[Valkyrie] VkInteropManager closed and all interop resources freed.");
    }
}
