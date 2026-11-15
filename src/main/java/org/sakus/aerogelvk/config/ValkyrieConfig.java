package org.sakus.aerogelvk.config;

import net.minecraft.client.Minecraft;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

/**
 * Runtime configuration for ValkyrieRender, tuned for integrated and discrete GPUs.
 * Persisted to config/valkyrie.properties when a Minecraft game directory is available.
 */
public final class ValkyrieConfig {

    public enum Preset {
        LOW,
        MEDIUM,
        HIGH
    }

    private static final Logger LOGGER = LoggerFactory.getLogger(ValkyrieConfig.class);
    private static final ValkyrieConfig INSTANCE = new ValkyrieConfig();

    public static ValkyrieConfig get() {
        return INSTANCE;
    }

    private boolean csmEnabled = true;
    private int csmResolution = 1024;
    private int csmCascades = 3;
    private int csmUpdateInterval = 6;
    private boolean fastShadowPcf = true;
    private boolean ssrEnabled = true;
    private boolean fxaaEnabled = true;
    private boolean hiZOcclusionEnabled = true;
    private float exposure = 1.0f;
    private boolean sodiumCompatEnabled = true;
    private boolean dhCompatEnabled = true;
    private boolean irisCompatEnabled = true;
    private String shaderPack = "off";
    private boolean shaderPackEnabled = false;

    private static final Set<String> KNOWN_KEYS = Set.of(
            "csmEnabled", "csmResolution", "csmCascades", "csmUpdateInterval",
            "fastShadowPcf", "ssrEnabled", "fxaaEnabled", "hiZOcclusionEnabled",
            "exposure", "sodiumCompatEnabled", "dhCompatEnabled", "irisCompatEnabled",
            "shaderPack", "shaderPackEnabled"
    );

    private final Map<String, String> unknownProperties = new LinkedHashMap<>();
    private boolean loaded = false;

    private ValkyrieConfig() {
    }

    public synchronized void loadOrCreate() {
        if (loaded) {
            return;
        }
        Path configPath = resolveConfigPath();
        if (configPath == null) {
            loaded = true;
            return;
        }
        loadFrom(configPath, true);
        loaded = true;
    }

    public synchronized void reload() {
        loaded = false;
        loadOrCreate();
    }

    public synchronized void loadFrom(Path configPath, boolean createIfMissing) {
        if (configPath == null) {
            return;
        }
        Properties props = new Properties();
        if (Files.exists(configPath)) {
            try (InputStream in = Files.newInputStream(configPath)) {
                props.load(in);
                applyProperties(props);
                LOGGER.info("[Valkyrie] Loaded config from {} (csm={} {}px x{} cascades, fastPcf={}, ssr={}, fxaa={}, hiZ={}, shaderPack={} enabled={})",
                        configPath, csmEnabled, csmResolution, csmCascades, fastShadowPcf, ssrEnabled, fxaaEnabled, hiZOcclusionEnabled, shaderPack, shaderPackEnabled);
            } catch (IOException e) {
                LOGGER.warn("[Valkyrie] Failed reading config {}, using defaults.", configPath, e);
            }
        } else if (createIfMissing) {
            save(configPath);
        }
    }

    public synchronized void saveTo(Path configPath) {
        if (configPath != null) {
            save(configPath);
        }
    }

    public synchronized void save() {
        Path configPath = resolveConfigPath();
        if (configPath != null) {
            save(configPath);
        }
    }

    private void save(Path configPath) {
        try {
            Files.createDirectories(configPath.getParent());
            Properties props = new Properties();
            if (Files.exists(configPath)) {
                try (InputStream existing = Files.newInputStream(configPath)) {
                    props.load(existing);
                } catch (IOException ignored) {
                }
            }
            for (Map.Entry<String, String> entry : unknownProperties.entrySet()) {
                props.putIfAbsent(entry.getKey(), entry.getValue());
            }
            props.setProperty("csmEnabled", Boolean.toString(csmEnabled));
            props.setProperty("csmResolution", Integer.toString(csmResolution));
            props.setProperty("csmCascades", Integer.toString(csmCascades));
            props.setProperty("csmUpdateInterval", Integer.toString(csmUpdateInterval));
            props.setProperty("fastShadowPcf", Boolean.toString(fastShadowPcf));
            props.setProperty("ssrEnabled", Boolean.toString(ssrEnabled));
            props.setProperty("fxaaEnabled", Boolean.toString(fxaaEnabled));
            props.setProperty("hiZOcclusionEnabled", Boolean.toString(hiZOcclusionEnabled));
            props.setProperty("exposure", Float.toString(exposure));
            props.setProperty("sodiumCompatEnabled", Boolean.toString(sodiumCompatEnabled));
            props.setProperty("dhCompatEnabled", Boolean.toString(dhCompatEnabled));
            props.setProperty("irisCompatEnabled", Boolean.toString(irisCompatEnabled));
            props.setProperty("shaderPack", shaderPack);
            props.setProperty("shaderPackEnabled", Boolean.toString(shaderPackEnabled));
            try (OutputStream out = Files.newOutputStream(configPath)) {
                props.store(out, "ValkyrieRender Configuration (Presets: LOW/MEDIUM/HIGH)");
            }
            LOGGER.info("[Valkyrie] Saved config to {}", configPath);
        } catch (IOException e) {
            LOGGER.warn("[Valkyrie] Could not save config to {}", configPath, e);
        }
    }

    private void applyProperties(Properties props) {
        unknownProperties.clear();
        for (String name : props.stringPropertyNames()) {
            if (!KNOWN_KEYS.contains(name)) {
                unknownProperties.put(name, props.getProperty(name));
            }
        }
        csmEnabled = parseBool(props.getProperty("csmEnabled"), csmEnabled);
        csmResolution = sanitizeResolution(parseInt(props.getProperty("csmResolution"), csmResolution));
        csmCascades = clampInt(parseInt(props.getProperty("csmCascades"), csmCascades), 1, 3);
        csmUpdateInterval = clampInt(parseInt(props.getProperty("csmUpdateInterval"), csmUpdateInterval), 1, 30);
        fastShadowPcf = parseBool(props.getProperty("fastShadowPcf"), fastShadowPcf);
        ssrEnabled = parseBool(props.getProperty("ssrEnabled"), ssrEnabled);
        fxaaEnabled = parseBool(props.getProperty("fxaaEnabled"), fxaaEnabled);
        hiZOcclusionEnabled = parseBool(props.getProperty("hiZOcclusionEnabled"), hiZOcclusionEnabled);
        exposure = Math.max(0.1f, Math.min(4.0f, parseFloat(props.getProperty("exposure"), exposure)));
        sodiumCompatEnabled = parseBool(props.getProperty("sodiumCompatEnabled"), sodiumCompatEnabled);
        dhCompatEnabled = parseBool(props.getProperty("dhCompatEnabled"), dhCompatEnabled);
        irisCompatEnabled = parseBool(props.getProperty("irisCompatEnabled"), irisCompatEnabled);
        String rawPack = props.getProperty("shaderPack");
        if (rawPack != null) {
            shaderPack = sanitizeShaderPackName(rawPack);
        }
        String rawEnabled = props.getProperty("shaderPackEnabled");
        if (rawEnabled != null) {
            shaderPackEnabled = parseBool(rawEnabled, shaderPackEnabled);
        } else if (rawPack != null) {
            shaderPackEnabled = !isOffToken(shaderPack);
        }
        if (shaderPackEnabled && isOffToken(shaderPack)) {
            shaderPack = "auto";
        }
    }

    public synchronized void applyPreset(Preset preset) {
        switch (preset) {
            case LOW -> {
                csmEnabled = false;
                csmResolution = 512;
                csmCascades = 1;
                csmUpdateInterval = 8;
                fastShadowPcf = true;
                ssrEnabled = false;
                fxaaEnabled = false;
                hiZOcclusionEnabled = true;
            }
            case MEDIUM -> {
                csmEnabled = true;
                csmResolution = 1024;
                csmCascades = 2;
                csmUpdateInterval = 6;
                fastShadowPcf = true;
                ssrEnabled = true;
                fxaaEnabled = true;
                hiZOcclusionEnabled = true;
            }
            case HIGH -> {
                csmEnabled = true;
                csmResolution = 2048;
                csmCascades = 3;
                csmUpdateInterval = 4;
                fastShadowPcf = false;
                ssrEnabled = true;
                fxaaEnabled = true;
                hiZOcclusionEnabled = true;
            }
        }
    }

    private static Path resolveConfigPath() {
        try {
            Minecraft mc = Minecraft.getInstance();
            if (mc != null && mc.gameDirectory != null) {
                Path configDir = mc.gameDirectory.toPath().resolve("config");
                Path primary = configDir.resolve("aerogelvk.properties");
                Path legacy = configDir.resolve("valkyrie.properties");
                return (!Files.exists(primary) && Files.exists(legacy)) ? legacy : primary;
            }
        } catch (Throwable ignored) {
            // Headless test environment without initialized Minecraft client
        }
        return null;
    }

    private static int sanitizeResolution(int res) {
        if (res <= 512) return 512;
        if (res <= 1024) return 1024;
        return 2048;
    }

    private static int clampInt(int v, int min, int max) {
        return Math.max(min, Math.min(max, v));
    }

    private static boolean parseBool(String s, boolean def) {
        return (s != null) ? Boolean.parseBoolean(s.trim()) : def;
    }

    private static int parseInt(String s, int def) {
        if (s == null) return def;
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static float parseFloat(String s, float def) {
        if (s == null) return def;
        try {
            return Float.parseFloat(s.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static String sanitizeShaderPackName(String raw) {
        if (raw == null) {
            return "off";
        }
        String trimmed = raw.trim();
        return trimmed.isEmpty() ? "off" : trimmed;
    }

    private static boolean isOffToken(String token) {
        if (token == null) {
            return true;
        }
        String t = token.trim();
        return t.isEmpty()
                || "off".equalsIgnoreCase(t)
                || "none".equalsIgnoreCase(t)
                || "false".equalsIgnoreCase(t)
                || "(none)".equalsIgnoreCase(t);
    }

    public boolean isCsmEnabled() {
        return csmEnabled;
    }

    public int getCsmResolution() {
        return csmResolution;
    }

    public int getCsmCascades() {
        return csmCascades;
    }

    public int getCsmUpdateInterval() {
        return csmUpdateInterval;
    }

    public boolean isFastShadowPcf() {
        return fastShadowPcf;
    }

    public boolean isSsrEnabled() {
        return ssrEnabled;
    }

    public boolean isFxaaEnabled() {
        return fxaaEnabled;
    }

    public boolean isHiZOcclusionEnabled() {
        return hiZOcclusionEnabled;
    }

    public float getExposure() {
        return exposure;
    }

    public boolean isSodiumCompatEnabled() {
        return sodiumCompatEnabled;
    }

    public boolean isDhCompatEnabled() {
        return dhCompatEnabled;
    }

    public boolean isIrisCompatEnabled() {
        return irisCompatEnabled;
    }

    public synchronized void setCsmEnabled(boolean csmEnabled) {
        this.csmEnabled = csmEnabled;
    }

    public synchronized void setCsmResolution(int csmResolution) {
        this.csmResolution = sanitizeResolution(csmResolution);
    }

    public synchronized void setCsmCascades(int csmCascades) {
        this.csmCascades = clampInt(csmCascades, 1, 3);
    }

    public synchronized void setCsmUpdateInterval(int csmUpdateInterval) {
        this.csmUpdateInterval = clampInt(csmUpdateInterval, 1, 30);
    }

    public synchronized void setFastShadowPcf(boolean fastShadowPcf) {
        this.fastShadowPcf = fastShadowPcf;
    }

    public synchronized void setSsrEnabled(boolean ssrEnabled) {
        this.ssrEnabled = ssrEnabled;
    }

    public synchronized void setFxaaEnabled(boolean fxaaEnabled) {
        this.fxaaEnabled = fxaaEnabled;
    }

    public synchronized void setHiZOcclusionEnabled(boolean hiZOcclusionEnabled) {
        this.hiZOcclusionEnabled = hiZOcclusionEnabled;
    }

    public synchronized void setExposure(float exposure) {
        if (!Float.isFinite(exposure)) {
            return;
        }
        this.exposure = Math.max(0.1f, Math.min(4.0f, exposure));
    }

    public synchronized void setSodiumCompatEnabled(boolean sodiumCompatEnabled) {
        this.sodiumCompatEnabled = sodiumCompatEnabled;
    }

    public synchronized void setDhCompatEnabled(boolean dhCompatEnabled) {
        this.dhCompatEnabled = dhCompatEnabled;
    }

    public synchronized void setIrisCompatEnabled(boolean irisCompatEnabled) {
        this.irisCompatEnabled = irisCompatEnabled;
    }

    public synchronized String getShaderPack() {
        return shaderPack;
    }

    public synchronized void setShaderPack(String shaderPack) {
        String sanitized = sanitizeShaderPackName(shaderPack);
        if (isOffToken(sanitized)) {
            this.shaderPack = "off";
            this.shaderPackEnabled = false;
        } else {
            this.shaderPack = sanitized;
            this.shaderPackEnabled = true;
        }
    }

    public synchronized boolean isShaderPackEnabled() {
        return shaderPackEnabled;
    }

    public synchronized void setShaderPackEnabled(boolean enabled) {
        this.shaderPackEnabled = enabled;
        if (enabled && isOffToken(this.shaderPack)) {
            this.shaderPack = "auto";
        }
    }

    public synchronized void disableShaderPack() {
        this.shaderPack = "off";
        this.shaderPackEnabled = false;
    }

    public synchronized boolean isShaderPackConfigured() {
        return shaderPackEnabled && !isOffToken(shaderPack);
    }

    public float getMaxShadowDistance() {
        if (!csmEnabled) return 0f;
        return switch (csmCascades) {
            case 1 -> 32.0f;
            case 2 -> 96.0f;
            default -> 256.0f;
        };
    }
}
