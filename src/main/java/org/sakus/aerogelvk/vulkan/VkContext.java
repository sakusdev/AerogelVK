package org.sakus.aerogelvk.vulkan;

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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Core Vulkan 1.3 Context and Resource Manager for ValkyrieRender.
 * Manages VkInstance, VkPhysicalDevice, VkDevice, graphics/transfer/compute VkQueues,
 * command pools, VMA allocator, and memory type queries.
 */
public class VkContext implements AutoCloseable {
    public static final int FRAMES_IN_FLIGHT = 2;
    private static final String VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
    private static final Logger LOGGER = LoggerFactory.getLogger(VkContext.class);
    private static VkContext instance;

    private VkInstance vkInstance;
    private VkPhysicalDevice physicalDevice;
    private VkDevice vkDevice;
    private VkQueue graphicsQueue;
    private VkQueue transferQueue;
    private VkQueue computeQueue;
    private int queueFamilyIndex = -1;
    private int transferQueueFamilyIndex = -1;
    private int transferQueueIndex = 0;
    private int computeQueueFamilyIndex = -1;
    private int computeQueueIndex = 0;
    private boolean dedicatedTransferQueue = false;
    private boolean dedicatedComputeQueue = false;
    private long commandPool;
    private long transferCommandPool;
    private long computeCommandPool;
    private long vmaAllocator;
    private VkPhysicalDeviceMemoryProperties memoryProperties;
    private String deviceName = "Unknown";
    private byte[] deviceUuid;
    private boolean externalMemorySupported = false;
    private boolean externalSemaphoreSupported = false;
    private boolean depthClampSupported = false;
    private boolean independentBlendSupported = false;
    private boolean maintenance4Supported = false;
    private long maxStorageBufferRange = 128L * 1024 * 1024;
    private boolean validationEnabled = false;
    private long debugMessenger = VK10.VK_NULL_HANDLE;
    private VkDebugUtilsMessengerCallbackEXT debugCallback;
    private final AtomicInteger validationErrorCount = new AtomicInteger(0);
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
        try {
            createInstance();
            pickPhysicalDevice();
            createLogicalDevice();
            createCommandPool();
            initVmaAllocator();

            initialized = true;
            LOGGER.info("[Valkyrie] Vulkan 1.3 Context successfully initialized.");
        } catch (Throwable t) {
            closePartialResources();
            throw t;
        }
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

            boolean wantValidation = Boolean.parseBoolean(
                    System.getProperty("aerogelvk.vk.validation",
                            System.getenv().getOrDefault("AEROGELVK_VK_VALIDATION", "false")));
            List<String> enabledLayers = new ArrayList<>();
            validationEnabled = false;
            validationErrorCount.set(0);
            if (wantValidation) {
                IntBuffer layerCount = stack.mallocInt(1);
                VK10.vkEnumerateInstanceLayerProperties(layerCount, null);
                VkLayerProperties.Buffer availableLayers = VkLayerProperties.malloc(layerCount.get(0), stack);
                VK10.vkEnumerateInstanceLayerProperties(layerCount, availableLayers);
                boolean hasValidationLayer = false;
                for (int i = 0; i < availableLayers.capacity(); i++) {
                    if (VALIDATION_LAYER_NAME.equals(availableLayers.get(i).layerNameString())) {
                        hasValidationLayer = true;
                        break;
                    }
                }
                if (hasValidationLayer) {
                    enabledLayers.add(VALIDATION_LAYER_NAME);
                    validationEnabled = true;
                    if (supportedExtNames.contains(EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                        enabledExtensions.add(EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                    }
                    LOGGER.info("[Valkyrie] Enabling Vulkan Validation Layer: {}", VALIDATION_LAYER_NAME);
                } else {
                    LOGGER.warn("[Valkyrie] AEROGELVK_VK_VALIDATION=true requested, but {} is not installed.", VALIDATION_LAYER_NAME);
                }
            }

            PointerBuffer ppExtensions = stack.mallocPointer(enabledExtensions.size());
            for (String ext : enabledExtensions) {
                LOGGER.info("[Valkyrie] Enabling Vulkan Instance extension: {}", ext);
                ppExtensions.put(stack.UTF8(ext));
            }
            ppExtensions.flip();

            PointerBuffer ppLayers = null;
            if (!enabledLayers.isEmpty()) {
                ppLayers = stack.mallocPointer(enabledLayers.size());
                for (String layer : enabledLayers) {
                    ppLayers.put(stack.UTF8(layer));
                }
                ppLayers.flip();
            }

            VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = null;
            if (validationEnabled && enabledExtensions.contains(EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                debugCallback = VkDebugUtilsMessengerCallbackEXT.create((messageSeverity, messageTypes, pCallbackData, pUserData) -> {
                    VkDebugUtilsMessengerCallbackDataEXT data = VkDebugUtilsMessengerCallbackDataEXT.create(pCallbackData);
                    String msg = data.pMessageString();
                    if ((messageSeverity & EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
                        validationErrorCount.incrementAndGet();
                        LOGGER.error("[VulkanValidation] {}", msg);
                    } else if ((messageSeverity & EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
                        LOGGER.warn("[VulkanValidation] {}", msg);
                    } else {
                        LOGGER.debug("[VulkanValidation] {}", msg);
                    }
                    return VK10.VK_FALSE;
                });
                debugCreateInfo = VkDebugUtilsMessengerCreateInfoEXT.calloc(stack)
                        .sType(EXTDebugUtils.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT)
                        .messageSeverity(EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                        .messageType(EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                                | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
                        .pfnUserCallback(debugCallback);
            }

            VkInstanceCreateInfo createInfo = VkInstanceCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                    .pApplicationInfo(appInfo)
                    .ppEnabledExtensionNames(ppExtensions);
            if (ppLayers != null) {
                createInfo.ppEnabledLayerNames(ppLayers);
            }
            if (debugCreateInfo != null) {
                createInfo.pNext(debugCreateInfo.address());
            }

            PointerBuffer pInstance = stack.mallocPointer(1);
            int res = VK10.vkCreateInstance(createInfo, null, pInstance);
            if (res != VK10.VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Vulkan 1.3 Instance, error code: " + res);
            }
            vkInstance = new VkInstance(pInstance.get(0), createInfo);

            if (debugCreateInfo != null) {
                java.nio.LongBuffer pMessenger = stack.mallocLong(1);
                if (EXTDebugUtils.vkCreateDebugUtilsMessengerEXT(vkInstance, debugCreateInfo, null, pMessenger) == VK10.VK_SUCCESS) {
                    debugMessenger = pMessenger.get(0);
                }
            }
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
                String preferredDevice = System.getProperty("aerogelvk.vk.device", System.getenv("AEROGELVK_VK_DEVICE"));
                if (preferredDevice != null && !preferredDevice.isBlank()
                        && properties.deviceNameString().toLowerCase(java.util.Locale.ROOT)
                        .contains(preferredDevice.trim().toLowerCase(java.util.Locale.ROOT))) {
                    score += 10000;
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
            selectAuxQueues(this.physicalDevice, this.queueFamilyIndex, stack);

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
            this.maxStorageBufferRange = Integer.toUnsignedLong(props2.properties().limits().maxStorageBufferRange());
            byte[] uuid = new byte[VK10.VK_UUID_SIZE];
            idProps.deviceUUID().get(uuid);
            this.deviceUuid = uuid;
            LOGGER.info("[Valkyrie] Selected Physical Device: {} (Graphics Family: {}, Transfer: {}/{} dedicated={}, Compute: {}/{} dedicated={})",
                    this.deviceName, queueFamilyIndex,
                    transferQueueFamilyIndex, transferQueueIndex, dedicatedTransferQueue,
                    computeQueueFamilyIndex, computeQueueIndex, dedicatedComputeQueue);
        }
    }

    private void selectAuxQueues(VkPhysicalDevice device, int gfxFamily, MemoryStack stack) {
        IntBuffer pQueueCount = stack.mallocInt(1);
        VK10.vkGetPhysicalDeviceQueueFamilyProperties(device, pQueueCount, null);
        int count = pQueueCount.get(0);
        VkQueueFamilyProperties.Buffer qProps = VkQueueFamilyProperties.malloc(count, stack);
        VK10.vkGetPhysicalDeviceQueueFamilyProperties(device, pQueueCount, qProps);

        // 1. Transfer queue: prefer dedicated TRANSFER-only family, else TRANSFER without GRAPHICS,
        //    else second queue in graphics family, else fallback to graphicsQueue.
        int bestTransferFamily = -1;
        int bestTransferIdx = 0;
        boolean foundDedicatedTransfer = false;
        for (int i = 0; i < count; i++) {
            int flags = qProps.get(i).queueFlags();
            if ((flags & VK10.VK_QUEUE_TRANSFER_BIT) != 0
                    && (flags & VK10.VK_QUEUE_GRAPHICS_BIT) == 0
                    && (flags & VK10.VK_QUEUE_COMPUTE_BIT) == 0) {
                bestTransferFamily = i;
                foundDedicatedTransfer = true;
                break;
            }
        }
        if (!foundDedicatedTransfer) {
            for (int i = 0; i < count; i++) {
                int flags = qProps.get(i).queueFlags();
                if ((flags & VK10.VK_QUEUE_TRANSFER_BIT) != 0 && (flags & VK10.VK_QUEUE_GRAPHICS_BIT) == 0) {
                    bestTransferFamily = i;
                    foundDedicatedTransfer = true;
                    break;
                }
            }
        }
        if (!foundDedicatedTransfer && qProps.get(gfxFamily).queueCount() > 1) {
            bestTransferFamily = gfxFamily;
            bestTransferIdx = 1;
            foundDedicatedTransfer = true;
        }
        if (foundDedicatedTransfer) {
            this.transferQueueFamilyIndex = bestTransferFamily;
            this.transferQueueIndex = bestTransferIdx;
            this.dedicatedTransferQueue = true;
        } else {
            this.transferQueueFamilyIndex = gfxFamily;
            this.transferQueueIndex = 0;
            this.dedicatedTransferQueue = false;
        }

        // 2. Compute queue: prefer dedicated COMPUTE without GRAPHICS, else unused queue in graphics family,
        //    else fallback to graphicsQueue.
        int bestComputeFamily = -1;
        int bestComputeIdx = 0;
        boolean foundDedicatedCompute = false;
        for (int i = 0; i < count; i++) {
            int flags = qProps.get(i).queueFlags();
            if ((flags & VK10.VK_QUEUE_COMPUTE_BIT) != 0 && (flags & VK10.VK_QUEUE_GRAPHICS_BIT) == 0) {
                bestComputeFamily = i;
                // If transfer also picked this same family at idx 0 and queueCount > 1, use idx 1
                if (bestComputeFamily == transferQueueFamilyIndex && transferQueueIndex == 0 && qProps.get(i).queueCount() > 1) {
                    bestComputeIdx = 1;
                }
                foundDedicatedCompute = true;
                break;
            }
        }
        if (!foundDedicatedCompute) {
            int gfxCount = qProps.get(gfxFamily).queueCount();
            int nextGfxIdx = (dedicatedTransferQueue && transferQueueFamilyIndex == gfxFamily) ? 2 : 1;
            if (gfxCount > nextGfxIdx) {
                bestComputeFamily = gfxFamily;
                bestComputeIdx = nextGfxIdx;
                foundDedicatedCompute = true;
            }
        }
        if (foundDedicatedCompute) {
            this.computeQueueFamilyIndex = bestComputeFamily;
            this.computeQueueIndex = bestComputeIdx;
            this.dedicatedComputeQueue = true;
        } else {
            this.computeQueueFamilyIndex = gfxFamily;
            this.computeQueueIndex = 0;
            this.dedicatedComputeQueue = false;
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

            // Query supported core and Vulkan 1.3 features so optional features are only enabled if supported
            VkPhysicalDeviceVulkan13Features supportedVk13 = VkPhysicalDeviceVulkan13Features.calloc(stack)
                    .sType(VK13.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
            VkPhysicalDeviceFeatures2 supportedFeatures2 = VkPhysicalDeviceFeatures2.calloc(stack)
                    .sType(VK11.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                    .pNext(supportedVk13.address());
            VK11.vkGetPhysicalDeviceFeatures2(physicalDevice, supportedFeatures2);
            VkPhysicalDeviceFeatures supportedCore = supportedFeatures2.features();
            this.depthClampSupported = supportedCore.depthClamp();
            this.independentBlendSupported = supportedCore.independentBlend();
            this.maintenance4Supported = supportedVk13.maintenance4();

            // Queue setup (unique families with required queue counts)
            Map<Integer, Integer> familyQueueCounts = new LinkedHashMap<>();
            familyQueueCounts.put(queueFamilyIndex, 1);
            familyQueueCounts.merge(transferQueueFamilyIndex, transferQueueIndex + 1, Math::max);
            familyQueueCounts.merge(computeQueueFamilyIndex, computeQueueIndex + 1, Math::max);

            VkDeviceQueueCreateInfo.Buffer queueCreateInfos = VkDeviceQueueCreateInfo.calloc(familyQueueCounts.size(), stack);
            int qi = 0;
            for (Map.Entry<Integer, Integer> entry : familyQueueCounts.entrySet()) {
                int qCount = entry.getValue();
                FloatBuffer priorities = stack.mallocFloat(qCount);
                for (int k = 0; k < qCount; k++) {
                    priorities.put(k, 1.0f);
                }
                queueCreateInfos.get(qi++)
                        .sType(VK10.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                        .queueFamilyIndex(entry.getKey())
                        .pQueuePriorities(priorities);
            }

            // Enable Vulkan 1.2 and 1.3 features
            VkPhysicalDeviceVulkan13Features vk13Features = VkPhysicalDeviceVulkan13Features.calloc(stack)
                    .sType(VK13.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
                    .dynamicRendering(true)
                    .synchronization2(true)
                    .maintenance4(maintenance4Supported);

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
            features2.features().depthClamp(depthClampSupported);
            features2.features().independentBlend(independentBlendSupported);
            features2.features().shaderInt64(supportedCore.shaderInt64());
            features2.features().shaderStorageImageArrayDynamicIndexing(supportedCore.shaderStorageImageArrayDynamicIndexing());

            VkDeviceCreateInfo deviceCreateInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                    .pNext(features2.address())
                    .pQueueCreateInfos(queueCreateInfos)
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

            if (transferQueueFamilyIndex == queueFamilyIndex && transferQueueIndex == 0) {
                transferQueue = graphicsQueue;
            } else {
                VK10.vkGetDeviceQueue(vkDevice, transferQueueFamilyIndex, transferQueueIndex, pQueue);
                transferQueue = new VkQueue(pQueue.get(0), vkDevice);
            }

            if (computeQueueFamilyIndex == queueFamilyIndex && computeQueueIndex == 0) {
                computeQueue = graphicsQueue;
            } else if (computeQueueFamilyIndex == transferQueueFamilyIndex && computeQueueIndex == transferQueueIndex) {
                computeQueue = transferQueue;
            } else {
                VK10.vkGetDeviceQueue(vkDevice, computeQueueFamilyIndex, computeQueueIndex, pQueue);
                computeQueue = new VkQueue(pQueue.get(0), vkDevice);
            }
        }
    }

    private void createCommandPool() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            commandPool = createCommandPoolForFamily(queueFamilyIndex, stack);
            transferCommandPool = createCommandPoolForFamily(transferQueueFamilyIndex, stack);
            computeCommandPool = createCommandPoolForFamily(computeQueueFamilyIndex, stack);
        }
    }

    private long createCommandPoolForFamily(int familyIdx, MemoryStack stack) {
        VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType(VK10.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                .flags(VK10.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(familyIdx);

        java.nio.LongBuffer pPool = stack.mallocLong(1);
        int res = VK10.vkCreateCommandPool(vkDevice, poolInfo, null, pPool);
        if (res != VK10.VK_SUCCESS) {
            throw new IllegalStateException("Failed to create Vulkan Command Pool for family " + familyIdx + ", error code: " + res);
        }
        return pPool.get(0);
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

    public VkCommandBuffer allocateTransferCommandBuffer() {
        return allocateCommandBufferFromPool(transferCommandPool);
    }

    public void freeTransferCommandBuffer(VkCommandBuffer cmd) {
        if (cmd != null && vkDevice != null && transferCommandPool != 0) {
            VK10.vkFreeCommandBuffers(vkDevice, transferCommandPool, cmd);
        }
    }

    public VkCommandBuffer allocateComputeCommandBuffer() {
        return allocateCommandBufferFromPool(computeCommandPool);
    }

    public void freeComputeCommandBuffer(VkCommandBuffer cmd) {
        if (cmd != null && vkDevice != null && computeCommandPool != 0) {
            VK10.vkFreeCommandBuffers(vkDevice, computeCommandPool, cmd);
        }
    }

    private VkCommandBuffer allocateCommandBufferFromPool(long pool) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType(VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                    .commandPool(pool)
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

    public VkQueue getTransferQueue() {
        return transferQueue;
    }

    public VkQueue getComputeQueue() {
        return computeQueue;
    }

    public int getQueueFamilyIndex() {
        return queueFamilyIndex;
    }

    public int getTransferQueueFamilyIndex() {
        return transferQueueFamilyIndex;
    }

    public int getComputeQueueFamilyIndex() {
        return computeQueueFamilyIndex;
    }

    public boolean hasDedicatedTransferQueue() {
        return dedicatedTransferQueue;
    }

    public boolean hasSeparateTransferQueueFamily() {
        return dedicatedTransferQueue && transferQueueFamilyIndex != queueFamilyIndex;
    }

    public boolean hasDedicatedComputeQueue() {
        return dedicatedComputeQueue;
    }

    public boolean hasSeparateComputeQueueFamily() {
        return dedicatedComputeQueue && computeQueueFamilyIndex != queueFamilyIndex;
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

    public boolean isDepthClampSupported() {
        return depthClampSupported;
    }

    public boolean isIndependentBlendSupported() {
        return independentBlendSupported;
    }

    public long getMaxStorageBufferRange() {
        return maxStorageBufferRange;
    }

    public boolean supportsColorAttachmentBlend(int vkFormat) {
        if (physicalDevice == null) {
            return true;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkFormatProperties props = VkFormatProperties.calloc(stack);
            VK10.vkGetPhysicalDeviceFormatProperties(physicalDevice, vkFormat, props);
            return (props.optimalTilingFeatures() & VK10.VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
        }
    }

    public boolean isValidationEnabled() {
        return validationEnabled;
    }

    public int getValidationErrorCount() {
        return validationErrorCount.get();
    }

    public boolean isInitialized() {
        return initialized;
    }

    public synchronized void cleanup() {
        close();
    }

    @Override
    public synchronized void close() {
        if (!initialized && vkInstance == null && vkDevice == null && vmaAllocator == 0) {
            return;
        }

        LOGGER.info("[Valkyrie] Cleaning up Vulkan 1.3 Context...");
        closePartialResources();
        LOGGER.info("[Valkyrie] Vulkan 1.3 Context destroyed cleanly.");
    }

    private void closePartialResources() {
        if (vkDevice != null) {
            try {
                VK10.vkDeviceWaitIdle(vkDevice);
            } catch (Throwable ignored) {
            }
        }

        if (vmaAllocator != 0) {
            Vma.vmaDestroyAllocator(vmaAllocator);
            vmaAllocator = 0;
        }

        if (computeCommandPool != 0 && vkDevice != null) {
            VK10.vkDestroyCommandPool(vkDevice, computeCommandPool, null);
            computeCommandPool = 0;
        }

        if (transferCommandPool != 0 && vkDevice != null) {
            VK10.vkDestroyCommandPool(vkDevice, transferCommandPool, null);
            transferCommandPool = 0;
        }

        if (commandPool != 0 && vkDevice != null) {
            VK10.vkDestroyCommandPool(vkDevice, commandPool, null);
            commandPool = 0;
        }

        if (vkDevice != null) {
            VK10.vkDestroyDevice(vkDevice, null);
            vkDevice = null;
        }

        if (debugMessenger != VK10.VK_NULL_HANDLE && vkInstance != null) {
            EXTDebugUtils.vkDestroyDebugUtilsMessengerEXT(vkInstance, debugMessenger, null);
            debugMessenger = VK10.VK_NULL_HANDLE;
        }
        if (debugCallback != null) {
            debugCallback.free();
            debugCallback = null;
        }

        if (vkInstance != null) {
            VK10.vkDestroyInstance(vkInstance, null);
            vkInstance = null;
        }

        if (memoryProperties != null) {
            memoryProperties.free();
            memoryProperties = null;
        }

        physicalDevice = null;
        initialized = false;
    }
}
