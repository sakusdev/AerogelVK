package org.sakus.aerogelvk.compat;

import org.sakus.aerogelvk.config.ValkyrieConfig;
import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Coordinates optional mod integrations across rendering, lighting, chunking, memory,
 * and lifecycle optimization mods so Valkyrie runs cleanly alongside them.
 */
public final class CompatManager {

    private static final Logger LOGGER = LoggerFactory.getLogger(CompatManager.class);

    private static final List<String> KNOWN_COMPAT_MODS = List.of(
            "sodium",
            "sodium-extra",
            "reeses-sodium-options",
            "immediatelyfast",
            "entityculling",
            "moreculling",
            "enhancedblockentities",
            "iris",
            "lithium",
            "c2me",
            "scalablelux",
            "noisium",
            "alternate-current",
            "ferritecore",
            "modernfix",
            "krypton",
            "dynamic_fps",
            "fastquit",
            "distanthorizons"
    );

    private static final Map<String, Boolean> DETECTED_MODS = new LinkedHashMap<>();

    private static boolean sodiumLoaded;
    private static boolean irisLoaded;
    private static boolean lithiumLoaded;
    private static boolean dhLoaded;
    private static boolean initialized;

    private CompatManager() {
    }

    public static synchronized void init() {
        if (initialized) {
            return;
        }
        FabricLoader loader = FabricLoader.getInstance();
        DETECTED_MODS.clear();
        StringBuilder detectedSummary = new StringBuilder();
        for (String modId : KNOWN_COMPAT_MODS) {
            boolean present = loader.isModLoaded(modId);
            DETECTED_MODS.put(modId, present);
            if (present) {
                if (detectedSummary.length() > 0) {
                    detectedSummary.append(", ");
                }
                detectedSummary.append(modId);
            }
        }

        sodiumLoaded = DETECTED_MODS.getOrDefault("sodium", false);
        irisLoaded = DETECTED_MODS.getOrDefault("iris", false);
        lithiumLoaded = DETECTED_MODS.getOrDefault("lithium", false);
        dhLoaded = DETECTED_MODS.getOrDefault("distanthorizons", false);

        LOGGER.info("[Valkyrie] Compat detection: sodium={}, lithium={}, distanthorizons={}, iris={}, active=[{}]",
                sodiumLoaded, lithiumLoaded, dhLoaded, irisLoaded,
                detectedSummary.isEmpty() ? "none" : detectedSummary);

        ValkyrieConfig config = ValkyrieConfig.get();
        config.loadOrCreate();

        if (dhLoaded && config.isDhCompatEnabled()) {
            try {
                DistantHorizonsCompat.register();
            } catch (Throwable t) {
                LOGGER.warn("[Valkyrie] Failed to initialize Distant Horizons compat bridge", t);
            }
        }

        initialized = true;
    }

    public static boolean isModDetected(String modId) {
        return DETECTED_MODS.getOrDefault(modId, false);
    }

    public static boolean isSodiumLoaded() {
        return sodiumLoaded;
    }

    public static boolean isIrisLoaded() {
        return irisLoaded;
    }

    public static boolean isLithiumLoaded() {
        return lithiumLoaded;
    }

    public static boolean isDhLoaded() {
        return dhLoaded;
    }

    /**
     * Shader rendering priority order:
     * <ol>
     *     <li><b>External Iris OpenGL shaderpack</b>: when Iris is installed, {@code irisCompatEnabled}
     *         is true, and {@link IrisCompat#isShaderPackInUse()} is true, Valkyrie yields world/terrain
     *         rendering cleanly to Iris/Sodium for that frame.</li>
     *     <li><b>Native Valkyrie Vulkan shaderpack</b>: when not yielding to Iris and
     *         {@link ValkyrieConfig#isShaderPackConfigured()} is true, Valkyrie executes the active
     *         OptiFine/Iris shaderpack natively via {@code VkShaderPackPipeline}.</li>
     *     <li><b>Valkyrie built-in deferred PBR pipeline</b>: default fallback when no shaderpack is
     *         active or if a native shaderpack fails to load/compile.</li>
     * </ol>
     */
    public static boolean shouldYieldToExternalShaderPipeline() {
        return irisLoaded
                && ValkyrieConfig.get().isIrisCompatEnabled()
                && IrisCompat.isShaderPackInUse();
    }

    /**
     * True when Valkyrie should execute a configured native Vulkan shaderpack
     * (i.e., not yielding to an active Iris OpenGL shaderpack and a Valkyrie shaderpack is enabled).
     */
    public static boolean isNativeShaderPackAllowed() {
        return !shouldYieldToExternalShaderPipeline()
                && ValkyrieConfig.get().isShaderPackConfigured();
    }

    /**
     * True when Sodium's chunk meshing and GL terrain draws should be suppressed in favor of Valkyrie.
     */
    public static boolean isSodiumTerrainBypassed() {
        return sodiumLoaded
                && ValkyrieConfig.get().isSodiumCompatEnabled()
                && !shouldYieldToExternalShaderPipeline()
                && !org.sakus.aerogelvk.ValkyrieRenderer.getInstance().isUnavailable();
    }

    public static boolean consumeDhFrameTextures(int[] outColorAndDepthTexIds) {
        if (!dhLoaded || !ValkyrieConfig.get().isDhCompatEnabled() || shouldYieldToExternalShaderPipeline()) {
            return false;
        }
        return DistantHorizonsCompat.consumeDhFrameTextures(outColorAndDepthTexIds);
    }
}
