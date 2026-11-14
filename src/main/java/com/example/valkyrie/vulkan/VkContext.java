package com.example.valkyrie.vulkan;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.Platform;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocatorCreateInfo;
import org.lwjgl.util.vma.VmaVulkanFunctions;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Core Vulkan 1.3 Context and Resource Manager for ValkyrieRender.
 * Manages VkInstance, VkPhysicalDevice, VkDevice, graphics/compute VkQueue,
 * command pools, VMA allocator, and memory type queries.
 */
public class VkContext implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(VkContext.class);
    private static VkContext instance;

    private VkInstance vkInstance;
    private VkPhysicalDevice physicalDevice;
    private VkDevice vkDevice;
    private VkQueue graphicsQueue;
    private int queueFamilyIndex = -1;
    private long commandPool;
    private long vmaAllocator;
    private VkPhysicalDeviceMemoryProperties memoryProperties;
    private String deviceName = "Unknown";
    private byte[] deviceUuid;
    private boolean externalMemorySupported = false;
    private boolean externalSemaphoreSupported = false;
    private boolean initialized = false;

    public static synchronized VkContext getInstance() {
        if (instance == null) {
            instance = new VkContext();
        }
        return instance;
    }

    public static VkContext get() {
        return getInstance();
    }

    public synchronized void init() {
        if (initialized) {
            LOGGER.warn("[Valkyrie] VkContext already initialized.");
            return;
        }

        LOGGER.info("[Valkyrie] Initializing Vulkan 1.3 Context...");
        createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        initVmaAllocator();

        initialized = true;
        LOGGER.info("[Valkyrie] Vulkan 1.3 Context successfully initialized.");
    }

    private void createInstance() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkApplicationInfo appInfo = VkApplicationInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_APPLICATION_INFO)
                    .pApplicationName(stack.UTF8("ValkyrieRender"))
                    .applicationVersion(VK10.VK_MAKE_VERSION(1, 0, 0))
                    .pEngineName(stack.UTF8("ValkyrieEngine"))
                    .engineVersion(VK10.VK_MAKE_VERSION(1, 0, 0))
                    .apiVersion(VK13.VK_API_VERSION_1_3);

            // Query supported instance extensions
            IntBuffer extCount = stack.mallocInt(1);
            VK10.vkEnumerateInstanceExtensionProperties((ByteBuffer) null, extCount, null);
            VkExtensionProperties.Buffer availableExts = VkExtensionProperties.malloc(extCount.get(0), stack);
            VK10.vkEnumerateInstanceExtensionProperties((ByteBuffer) null, extCount, availableExts);

            Set<String> supportedExtNames = new HashSet<>();
            for (int i = 0; i < availableExts.capacity(); i++) {
                supportedExtNames.add(availableExts.get(i).extensionNameString());
            }

            List<String> enabledExtensions = new ArrayList<>();
            // Enable external memory and semaphore capabilities if present
            if (supportedExtNames.contains(KHRExternalMemoryCapabilities.VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME)) {
                enabledExtensions.add(KHRExternalMemoryCapabilities.VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
            }
            if (supportedExtNames.contains(KHRExternalSemaphoreCapabilities.VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME)) {
                enabledExtensions.add(KHRExternalSemaphoreCapabilities.VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
            }

            PointerBuffer ppExtensions = stack.mallocPointer(enabledExtensions.size());
            for (String ext : enabledExtensions) {
                LOGGER.info("[Valkyrie] Enabling Vulkan Instance extension: {}", ext);
                ppExtensions.put(stack.UTF8(ext));
            }
            ppExtensions.flip();

            VkInstanceCreateInfo createInfo = VkInstanceCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                    .pApplicationInfo(appInfo)
                    .ppEnabledExtensionNames(ppExtensions);

            PointerBuffer pInstance = stack.mallocPointer(1);
            int res = VK10.vkCreateInstance(createInfo, null, pInstance);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Vulkan 1.3 Instance, error code: " + res);
            }
            vkInstance = new VkInstance(pInstance.get(0), createInfo);
        }
    }

    private void pickPhysicalDevice() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer pDeviceCount = stack.mallocInt(1);
            VK10.vkEnumeratePhysicalDevices(vkInstance, pDeviceCount, null);
            int deviceCount = pDeviceCount.get(0);
            if (deviceCount == 0) {
                throw new IllegalStateException("Failed to find GPUs with Vulkan support!");
            }

            PointerBuffer pDevices = stack.mallocPointer(deviceCount);
            VK10.vkEnumeratePhysicalDevices(vkInstance, pDeviceCount, pDevices);

            VkPhysicalDevice bestDevice = null;
            int bestScore = -1;
            int bestQueueIndex = -1;

            for (int i = 0; i < deviceCount; i++) {
                VkPhysicalDevice device = new VkPhysicalDevice(pDevices.get(i), vkInstance);
                VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.malloc(stack);
                VK10.vkGetPhysicalDeviceProperties(device, properties);

                int qFamily = findQueueFamily(device, stack);
                if (qFamily < 0) {
                    continue; // No graphics+compute queue family
                }

                // Check Vulkan 1.3 features support
                if (!checkDeviceFeatures(device, stack)) {
                    LOGGER.warn("[Valkyrie] GPU {} does not satisfy required Vulkan 1.2/1.3 features.", properties.deviceNameString());
                    continue;
                }

                int score = 100;
                if (properties.deviceType() == VK10.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    score += 1000;
                } else if (properties.deviceType() == VK10.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                    score += 500;
                }

                LOGGER.info("[Valkyrie] Evaluated GPU [{}]: {} (Score: {})", i, properties.deviceNameString(), score);

                if (score > bestScore) {
                    bestScore = score;
                    bestDevice = device;
                    bestQueueIndex = qFamily;
                }
            }

            if (bestDevice == null) {
                throw new IllegalStateException("Failed to find a suitable GPU that supports required Vulkan 1.3 features!");
            }

            this.physicalDevice = bestDevice;
            this.queueFamilyIndex = bestQueueIndex;

            // Cache memory properties (calloc via MemoryUtil so close() can free() cleanly)
            this.memoryProperties = VkPhysicalDeviceMemoryProperties.calloc();
            VK10.vkGetPhysicalDeviceMemoryProperties(this.physicalDevice, this.memoryProperties);

            VkPhysicalDeviceIDProperties idProps = VkPhysicalDeviceIDProperties.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
            VkPhysicalDeviceProperties2 props2 = VkPhysicalDeviceProperties2.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2)
                    .pNext(idProps.address());
            VK11.vkGetPhysicalDeviceProperties2(this.physicalDevice, props2);
            this.deviceName = props2.properties().deviceNameString();
            byte[] uuid = new byte[VK10.VK_UUID_SIZE];
            idProps.deviceUUID().get(uuid);
            this.deviceUuid = uuid;
            LOGGER.info("[Valkyrie] Selected Physical Device: {} (Queue Family: {})", this.deviceName, queueFamilyIndex);
        }
    }

    private int findQueueFamily(VkPhysicalDevice device, MemoryStack stack) {
        IntBuffer pQueueCount = stack.mallocInt(1);
        VK10.vkGetPhysicalDeviceQueueFamilyProperties(device, pQueueCount, null);
        int count = pQueueCount.get(0);
        if (count == 0) return -1;

        VkQueueFamilyProperties.Buffer qProps = VkQueueFamilyProperties.malloc(count, stack);
        VK10.vkGetPhysicalDeviceQueueFamilyProperties(device, pQueueCount, qProps);

        for (int i = 0; i < count; i++) {
            int flags = qProps.get(i).queueFlags();
            if ((flags & VK10.VK_QUEUE_GRAPHICS_BIT) != 0 && (flags & VK10.VK_QUEUE_COMPUTE_BIT) != 0) {
                return i;
            }
        }
        return -1;
    }

    private boolean checkDeviceFeatures(VkPhysicalDevice device, MemoryStack stack) {
        VkPhysicalDeviceVulkan13Features vk13Features = VkPhysicalDeviceVulkan13Features.calloc(stack)
                .sType(VK13.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);

        VkPhysicalDeviceVulkan12Features vk12Features = VkPhysicalDeviceVulkan12Features.calloc(stack)
                .sType(VK12.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                .pNext(vk13Features.address());

        VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                .sType(VK11.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                .pNext(vk12Features.address());

        VK11.vkGetPhysicalDeviceFeatures2(device, features2);

        boolean hasDynamicRendering = vk13Features.dynamicRendering();
        boolean hasSync2 = vk13Features.synchronization2();
        boolean hasScalarBlockLayout = vk12Features.scalarBlockLayout();
        boolean hasDrawIndirectCount = vk12Features.drawIndirectCount();
        boolean hasMultiDrawIndirect = features2.features().multiDrawIndirect();

        return hasDynamicRendering && hasSync2 && hasScalarBlockLayout && hasDrawIndirectCount && hasMultiDrawIndirect;
    }

    private void createLogicalDevice() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Check available device extensions
            IntBuffer pDevExtCount = stack.mallocInt(1);
            VK10.vkEnumerateDeviceExtensionProperties(physicalDevice, (ByteBuffer) null, pDevExtCount, null);
            int extCount = pDevExtCount.get(0);
            VkExtensionProperties.Buffer devExts = VkExtensionProperties.malloc(extCount, stack);
            VK10.vkEnumerateDeviceExtensionProperties(physicalDevice, (ByteBuffer) null, pDevExtCount, devExts);

            Set<String> supportedDevExts = new HashSet<>();
            for (int i = 0; i < extCount; i++) {
                supportedDevExts.add(devExts.get(i).extensionNameString());
            }

            boolean isWindows = Platform.get() == Platform.WINDOWS;
            String extMemPlatform = isWindows
                    ? KHRExternalMemoryWin32.VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
                    : KHRExternalMemoryFd.VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
            String extSemPlatform = isWindows
                    ? KHRExternalSemaphoreWin32.VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
                    : KHRExternalSemaphoreFd.VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;

            List<String> extensionsToEnable = new ArrayList<>();
            externalMemorySupported = supportedDevExts.contains(KHRExternalMemory.VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME)
                    && supportedDevExts.contains(extMemPlatform);
            if (externalMemorySupported) {
                extensionsToEnable.add(KHRExternalMemory.VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
                extensionsToEnable.add(extMemPlatform);
            } else {
                LOGGER.warn("[Valkyrie] External memory extensions ({}, {}) not supported on selected GPU; GL interop disabled.",
                        KHRExternalMemory.VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, extMemPlatform);
            }

            externalSemaphoreSupported = supportedDevExts.contains(KHRExternalSemaphore.VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME)
                    && supportedDevExts.contains(extSemPlatform);
            if (externalSemaphoreSupported) {
                extensionsToEnable.add(KHRExternalSemaphore.VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
                extensionsToEnable.add(extSemPlatform);
            } else {
                LOGGER.warn("[Valkyrie] External semaphore extensions ({}, {}) not supported on selected GPU; GL semaphore sync disabled.",
                        KHRExternalSemaphore.VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, extSemPlatform);
            }

            PointerBuffer ppExtensions = stack.mallocPointer(extensionsToEnable.size());
            for (String ext : extensionsToEnable) {
                LOGGER.info("[Valkyrie] Enabling Vulkan Device extension: {}", ext);
                ppExtensions.put(stack.UTF8(ext));
            }
            ppExtensions.flip();

            // Query supported core features so we only enable optional features (like shaderInt64) if present
            VkPhysicalDeviceFeatures supportedCore = VkPhysicalDeviceFeatures.calloc(stack);
            VK10.vkGetPhysicalDeviceFeatures(physicalDevice, supportedCore);

            // Queue setup
            FloatBuffer pQueuePriorities = stack.floats(1.0f);
            VkDeviceQueueCreateInfo.Buffer queueCreateInfo = VkDeviceQueueCreateInfo.calloc(1, stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                    .queueFamilyIndex(queueFamilyIndex)
                    .pQueuePriorities(pQueuePriorities);

            // Enable Vulkan 1.2 and 1.3 features
            VkPhysicalDeviceVulkan13Features vk13Features = VkPhysicalDeviceVulkan13Features.calloc(stack)
                    .sType(VK13.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
                    .dynamicRendering(true)
                    .synchronization2(true);

            VkPhysicalDeviceVulkan12Features vk12Features = VkPhysicalDeviceVulkan12Features.calloc(stack)
                    .sType(VK12.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                    .scalarBlockLayout(true)
                    .drawIndirectCount(true)
                    .runtimeDescriptorArray(true)
                    .pNext(vk13Features.address());

            VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                    .pNext(vk12Features.address());
            features2.features().multiDrawIndirect(true);
            features2.features().shaderInt64(supportedCore.shaderInt64());
            features2.features().shaderStorageImageArrayDynamicIndexing(supportedCore.shaderStorageImageArrayDynamicIndexing());

            VkDeviceCreateInfo deviceCreateInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                    .pNext(features2.address())
                    .pQueueCreateInfos(queueCreateInfo)
                    .ppEnabledExtensionNames(ppExtensions);

            PointerBuffer pDevice = stack.mallocPointer(1);
            int res = VK10.vkCreateDevice(physicalDevice, deviceCreateInfo, null, pDevice);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Vulkan Logical Device, error code: " + res);
            }
            vkDevice = new VkDevice(pDevice.get(0), physicalDevice, deviceCreateInfo);

            PointerBuffer pQueue = stack.mallocPointer(1);
            VK10.vkGetDeviceQueue(vkDevice, queueFamilyIndex, 0, pQueue);
            graphicsQueue = new VkQueue(pQueue.get(0), vkDevice);
        }
    }

    private void createCommandPool() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                    .flags(VK10.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                    .queueFamilyIndex(queueFamilyIndex);

            java.nio.LongBuffer pPool = stack.mallocLong(1);
            int res = VK10.vkCreateCommandPool(vkDevice, poolInfo, null, pPool);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Vulkan Command Pool, error code: " + res);
            }
            commandPool = pPool.get(0);
        }
    }

    private void initVmaAllocator() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VmaVulkanFunctions vmaFunctions = VmaVulkanFunctions.calloc(stack);
            vmaFunctions.set(vkInstance, vkDevice);

            VmaAllocatorCreateInfo allocatorInfo = VmaAllocatorCreateInfo.calloc(stack)
                    .flags(0)
                    .physicalDevice(physicalDevice)
                    .device(vkDevice)
                    .instance(vkInstance)
                    .pVulkanFunctions(vmaFunctions)
                    .vulkanApiVersion(VK13.VK_API_VERSION_1_3);

            PointerBuffer pAllocator = stack.mallocPointer(1);
            int res = Vma.vmaCreateAllocator(allocatorInfo, pAllocator);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create VMA Allocator, error code: " + res);
            }
            vmaAllocator = pAllocator.get(0);
            LOGGER.info("[Valkyrie] VMA Allocator initialized successfully.");
        }
    }

    public VkCommandBuffer allocateCommandBuffer() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                    .commandPool(commandPool)
                    .level(VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(1);

            PointerBuffer pCmd = stack.mallocPointer(1);
            int res = VK10.vkAllocateCommandBuffers(vkDevice, allocInfo, pCmd);
            if (res != VK10.VK_SUCCESS) {
                throw new RuntimeException("Failed to allocate command buffer: " + res);
            }
            return new VkCommandBuffer(pCmd.get(0), vkDevice);
        }
    }

    public void freeCommandBuffer(VkCommandBuffer cmd) {
        if (cmd != null && vkDevice != null && commandPool != 0) {
            VK10.vkFreeCommandBuffers(vkDevice, commandPool, cmd);
        }
    }

    public VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBuffer commandBuffer = allocateCommandBuffer();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            int res = VK10.vkBeginCommandBuffer(commandBuffer, beginInfo);
            if (res != VK10.VK_SUCCESS) {
                freeCommandBuffer(commandBuffer);
                throw new RuntimeException("Failed to begin single time command buffer: " + res);
            }
        }
        return commandBuffer;
    }

    public void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        if (commandBuffer == null) return;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int res = VK10.vkEndCommandBuffer(commandBuffer);
            if (res != VK10.VK_SUCCESS) {
                throw new RuntimeException("Failed to end single time command buffer: " + res);
            }

            PointerBuffer pCmd = stack.pointers(commandBuffer);
            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(pCmd);

            res = VK10.vkQueueSubmit(graphicsQueue, submitInfo, VK10.VK_NULL_HANDLE);
            if (res != VK10.VK_SUCCESS) {
                throw new RuntimeException("Failed to submit single time command buffer: " + res);
            }
            VK10.vkQueueWaitIdle(graphicsQueue);
        } finally {
            freeCommandBuffer(commandBuffer);
        }
    }

    public int findMemoryType(int typeFilter, int properties) {
        if (memoryProperties == null) {
            throw new IllegalStateException("Vulkan memory properties not initialized!");
        }
        for (int i = 0; i < memoryProperties.memoryTypeCount(); i++) {
            if ((typeFilter & (1 << i)) != 0 &&
                    (memoryProperties.memoryTypes(i).propertyFlags() & properties) == properties) {
                return i;
            }
        }
        throw new RuntimeException("Failed to find suitable Vulkan memory type! Filter: " + typeFilter + ", Properties: " + properties);
    }

    public VkInstance getVkInstance() {
        return vkInstance;
    }

    public VkPhysicalDevice getPhysicalDevice() {
        return physicalDevice;
    }

    public VkDevice getVkDevice() {
        return vkDevice;
    }

    public VkQueue getGraphicsQueue() {
        return graphicsQueue;
    }

    public int getQueueFamilyIndex() {
        return queueFamilyIndex;
    }

    public long getCommandPool() {
        return commandPool;
    }

    public long getVmaAllocator() {
        return vmaAllocator;
    }

    public VkPhysicalDeviceMemoryProperties getMemoryProperties() {
        return memoryProperties;
    }

    public String getDeviceName() {
        return deviceName;
    }

    public byte[] getDeviceUuid() {
        return deviceUuid != null ? deviceUuid.clone() : null;
    }

    public boolean isExternalMemorySupported() {
        return externalMemorySupported;
    }

    public boolean isExternalSemaphoreSupported() {
        return externalSemaphoreSupported;
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

        LOGGER.info("[Valkyrie] Cleaning up Vulkan 1.3 Context...");
        if (vkDevice != null) {
            VK10.vkDeviceWaitIdle(vkDevice);
        }

        if (vmaAllocator != 0) {
            Vma.vmaDestroyAllocator(vmaAllocator);
            vmaAllocator = 0;
        }

        if (commandPool != 0) {
            VK10.vkDestroyCommandPool(vkDevice, commandPool, null);
            commandPool = 0;
        }

        if (vkDevice != null) {
            VK10.vkDestroyDevice(vkDevice, null);
            vkDevice = null;
        }

        if (vkInstance != null) {
            VK10.vkDestroyInstance(vkInstance, null);
            vkInstance = null;
        }

        if (memoryProperties != null) {
            memoryProperties.free();
            memoryProperties = null;
        }

        initialized = false;
        LOGGER.info("[Valkyrie] Vulkan 1.3 Context destroyed cleanly.");
    }
}
