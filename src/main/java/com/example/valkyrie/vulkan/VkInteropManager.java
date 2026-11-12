package com.example.valkyrie.vulkan;

import com.mojang.blaze3d.pipeline.RenderTarget;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opengl.*;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.Platform;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.IntBuffer;
import java.nio.LongBuffer;

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

    private int width = 0;
    private int height = 0;
    private boolean initialized = false;

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

        createSharedColorImage(ctx, device, isWindows);
        createSharedDepthImage(ctx, device, isWindows);
        initSharedSemaphores(device, isWindows);
        createSharedFramebuffer();

        // Transition Vulkan images from UNDEFINED to COLOR_ATTACHMENT / DEPTH_ATTACHMENT initial layouts
        transitionInitialLayouts(ctx);

        initialized = true;
        LOGGER.info("[Valkyrie] Vulkan-OpenGL Interop resources ready ({}x{}).", width, height);
    }

    private void createSharedColorImage(VkContext ctx, VkDevice device, boolean isWindows) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int handleType = isWindows
                    ? KHRExternalMemoryWin32.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
                    : KHRExternalMemoryFd.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

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

            VkMemoryRequirements memReqs = VkMemoryRequirements.malloc(stack);
            VK10.vkGetImageMemoryRequirements(device, colorImage, memReqs);

            VkExportMemoryAllocateInfo exportAllocInfo = VkExportMemoryAllocateInfo.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO)
                    .handleTypes(handleType);

            VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                    .pNext(exportAllocInfo.address())
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
                        .handleType(KHRExternalMemoryWin32.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR);

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
                        .handleType(KHRExternalMemoryFd.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);

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
                    ? KHRExternalMemoryWin32.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
                    : KHRExternalMemoryFd.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

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

            VkMemoryRequirements memReqs = VkMemoryRequirements.malloc(stack);
            VK10.vkGetImageMemoryRequirements(device, depthImage, memReqs);

            VkExportMemoryAllocateInfo exportAllocInfo = VkExportMemoryAllocateInfo.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO)
                    .handleTypes(handleType);

            VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                    .pNext(exportAllocInfo.address())
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
                        .handleType(KHRExternalMemoryWin32.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR);

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
                        .handleType(KHRExternalMemoryFd.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);

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
                    ? KHRExternalSemaphoreWin32.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
                    : KHRExternalSemaphoreFd.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

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
                        .handleType(KHRExternalSemaphoreWin32.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR);

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
                        .handleType(KHRExternalSemaphoreFd.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);

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
                        .handleType(KHRExternalSemaphoreWin32.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR);

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
                        .handleType(KHRExternalSemaphoreFd.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);

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

        // 1. OpenGL waits on Vulkan render complete semaphore
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer textures = stack.ints(glSharedColorTex, glSharedDepthTex);
            IntBuffer layouts = stack.ints(EXTMemoryObject.GL_LAYOUT_TRANSFER_SRC_EXT, EXTMemoryObject.GL_LAYOUT_TRANSFER_SRC_EXT);
            EXTSemaphore.glWaitSemaphoreEXT(glVkRenderCompleteSemaphore, null, textures, layouts);
        }

        // 2. Save previous OpenGL FBO bindings
        int prevReadFbo = GL11C.glGetInteger(GL30C.GL_READ_FRAMEBUFFER_BINDING);
        int prevDrawFbo = GL11C.glGetInteger(GL30C.GL_DRAW_FRAMEBUFFER_BINDING);

        // 3. Direct zero-copy blit of color and depth buffers
        GL45C.glBlitNamedFramebuffer(
                glSharedFbo,
                targetFboId,
                0, 0, width, height,
                0, 0, targetWidth, targetHeight,
                GL11C.GL_COLOR_BUFFER_BIT | GL11C.GL_DEPTH_BUFFER_BIT,
                GL11C.GL_NEAREST
        );

        // 4. Restore OpenGL state cleanly
        GL30C.glBindFramebuffer(GL30C.GL_READ_FRAMEBUFFER, prevReadFbo);
        GL30C.glBindFramebuffer(GL30C.GL_DRAW_FRAMEBUFFER, prevDrawFbo);

        // 5. Signal GL ready semaphore so Vulkan can proceed on next frame
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer textures = stack.ints(glSharedColorTex, glSharedDepthTex);
            IntBuffer layouts = stack.ints(EXTMemoryObject.GL_LAYOUT_COLOR_ATTACHMENT_EXT, EXTMemoryObject.GL_LAYOUT_DEPTH_STENCIL_ATTACHMENT_EXT);
            EXTSemaphore.glSignalSemaphoreEXT(glReadySemaphore, null, textures, layouts);
        }
    }

    public synchronized void destroySharedTexturesAndFbo() {
        VkDevice device = VkContext.getInstance().getVkDevice();

        if (glSharedFbo != 0) {
            GL30C.glDeleteFramebuffers(glSharedFbo);
            glSharedFbo = 0;
        }

        if (glSharedColorTex != 0) {
            GL11C.glDeleteTextures(glSharedColorTex);
            glSharedColorTex = 0;
        }

        if (glColorMemObject != 0) {
            EXTMemoryObject.glDeleteMemoryObjectsEXT(glColorMemObject);
            glColorMemObject = 0;
        }

        if (glSharedDepthTex != 0) {
            GL11C.glDeleteTextures(glSharedDepthTex);
            glSharedDepthTex = 0;
        }

        if (glDepthMemObject != 0) {
            EXTMemoryObject.glDeleteMemoryObjectsEXT(glDepthMemObject);
            glDepthMemObject = 0;
        }

        if (device != null) {
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

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public long getColorImage() {
        return colorImage;
    }

    public long getColorImageView() {
        return colorImageView;
    }

    public int getGlSharedColorTex() {
        return glSharedColorTex;
    }

    public long getDepthImage() {
        return depthImage;
    }

    public long getDepthImageView() {
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

        VkDevice device = VkContext.getInstance().getVkDevice();

        if (glVkRenderCompleteSemaphore != 0) {
            EXTSemaphore.glDeleteSemaphoresEXT(glVkRenderCompleteSemaphore);
            glVkRenderCompleteSemaphore = 0;
        }

        if (glReadySemaphore != 0) {
            EXTSemaphore.glDeleteSemaphoresEXT(glReadySemaphore);
            glReadySemaphore = 0;
        }

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

        LOGGER.info("[Valkyrie] VkInteropManager closed and all interop resources freed.");
    }
}
