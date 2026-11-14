package org.sakus.aerogelvk.shaderpack;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses and stores OptiFine / Iris {@code shaders.properties} settings as well as
 * GLSL comment and {@code const} directives:
 * <ul>
 *     <li>{@code /* DRAWBUFFERS:... * /} and {@code /* RENDERTARGETS:... * /}</li>
 *     <li>{@code colortex0Format}..{@code colortex15Format} (and legacy {@code gcolorFormat}..{@code gaux4Format})</li>
 *     <li>{@code const bool *MipmapEnabled} / {@code *Mipmap}</li>
 *     <li>{@code const bool *Clear} / {@code const vec4 *ClearColor}</li>
 *     <li>{@code const int shadowMapResolution}, {@code const float shadowDistance}, {@code const float sunPathRotation}</li>
 *     <li>{@code scale.<pass>}, {@code alphaTest.<pass>}, {@code blend.<pass>}, {@code flip.<pass>.<buffer>}</li>
 *     <li>Slider lists and profile macro defaults ({@code sliders}, {@code profile.<name>})</li>
 * </ul>
 */
public final class ShaderPackProperties {

    public static final String DEFAULT_COLOR_FORMAT = "RGBA8";
    public static final int DEFAULT_SHADOW_MAP_RESOLUTION = 1024;
    public static final float DEFAULT_SHADOW_DISTANCE = 120.0f;
    public static final float DEFAULT_SUN_PATH_ROTATION = 0.0f;
    public static final int DEFAULT_NOISE_RESOLUTION = 64;

    private static final Pattern DRAWBUFFERS_PATTERN = Pattern.compile(
            "/\\*\\s*DRAWBUFFERS\\s*:\\s*([0-9A-Za-z]+)\\s*\\*/");

    private static final Pattern RENDERTARGETS_PATTERN = Pattern.compile(
            "/\\*\\s*RENDERTARGETS\\s*:\\s*([0-9,\\s]+)\\s*\\*/");

    private static final Pattern COMMENT_LEGACY_FORMAT_PATTERN = Pattern.compile(
            "/\\*\\s*(GCOLOR|GDEPTH|GNORMAL|COMPOSITE|GAUX1|GAUX2|GAUX3|GAUX4)FORMAT\\s*:\\s*([A-Za-z0-9_]+)\\s*\\*/");

    private static final Pattern COMMENT_SHADOWRES_PATTERN = Pattern.compile(
            "/\\*\\s*SHADOWRES\\s*:\\s*([0-9]+)\\s*\\*/");

    private static final Pattern COMMENT_SHADOWHPL_PATTERN = Pattern.compile(
            "/\\*\\s*SHADOWHPL\\s*:\\s*([0-9.+-]+)\\s*\\*/");

    private static final Pattern CONST_FORMAT_PATTERN = Pattern.compile(
            "\\bconst\\s+int\\s+(colortex(?:1[0-5]|[0-9])|gcolor|gdepth|gnormal|composite|gaux[1-4]|shadowcolor[01]|noisetex)Format\\s*=\\s*([A-Za-z0-9_]+)\\s*;");

    private static final Pattern CONST_MIPMAP_PATTERN = Pattern.compile(
            "\\bconst\\s+bool\\s+(colortex(?:1[0-5]|[0-9])|gcolor|gdepth|gnormal|composite|gaux[1-4]|shadowtex[01]|shadow[cC]olor[01])(?:MipmapEnabled|Mipmap)\\s*=\\s*(true|false)\\s*;");

    private static final Pattern CONST_CLEAR_PATTERN = Pattern.compile(
            "\\bconst\\s+bool\\s+(colortex(?:1[0-5]|[0-9])|gcolor|gdepth|gnormal|composite|gaux[1-4]|shadow[cC]olor[01]|shadowtex[01])Clear\\s*=\\s*(true|false)\\s*;");

    private static final Pattern CONST_CLEAR_COLOR_PATTERN = Pattern.compile(
            "\\bconst\\s+vec4\\s+(colortex(?:1[0-5]|[0-9])|gcolor|gdepth|gnormal|composite|gaux[1-4])ClearColor\\s*=\\s*vec4\\s*\\(([^)]+)\\)\\s*;");

    private static final Pattern CONST_INT_DIRECTIVE_PATTERN = Pattern.compile(
            "\\bconst\\s+int\\s+(shadowMapResolution|noiseTextureResolution)\\s*=\\s*([+-]?[0-9]+)\\s*;");

    private static final Pattern CONST_FLOAT_DIRECTIVE_PATTERN = Pattern.compile(
            "\\bconst\\s+float\\s+(shadowDistance|sunPathRotation|shadowIntervalSize|shadowDistanceRenderMul|ambientOcclusionLevel|centerDepthHalflife|eyeBrightnessHalflife|wetnessHalflife|drynessHalflife)\\s*=\\s*([+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?f?)\\s*;");

    private final Map<String, String> rawProperties = new LinkedHashMap<>();
    private final Map<Integer, String> colorTexFormats = new LinkedHashMap<>();
    private final Map<Integer, String> shadowColorFormats = new LinkedHashMap<>();
    private String noiseTexFormat = DEFAULT_COLOR_FORMAT;

    private final Map<Integer, Boolean> colorTexMipmapEnabled = new LinkedHashMap<>();
    private final Map<Integer, Boolean> shadowTexMipmapEnabled = new LinkedHashMap<>();
    private final Map<Integer, Boolean> shadowColorMipmapEnabled = new LinkedHashMap<>();

    private final Map<Integer, Boolean> colorTexClearEnabled = new LinkedHashMap<>();
    private final Map<Integer, Boolean> shadowColorClearEnabled = new LinkedHashMap<>();
    private final Map<Integer, float[]> colorTexClearColors = new LinkedHashMap<>();

    private final Map<String, int[]> passDrawBuffers = new LinkedHashMap<>();
    private final Map<String, Float> passScales = new LinkedHashMap<>();
    private final Map<String, String> passAlphaTests = new LinkedHashMap<>();
    private final Map<String, String> passBlends = new LinkedHashMap<>();
    private final Map<String, Map<String, Boolean>> passFlips = new LinkedHashMap<>();

    private final Set<String> sliders = new LinkedHashSet<>();
    private final Map<String, Map<String, String>> profiles = new LinkedHashMap<>();
    private final Map<String, String> optionDefaults = new LinkedHashMap<>();

    private int shadowMapResolution = DEFAULT_SHADOW_MAP_RESOLUTION;
    private float shadowDistance = DEFAULT_SHADOW_DISTANCE;
    private float sunPathRotation = DEFAULT_SUN_PATH_ROTATION;
    private float shadowIntervalSize = 2.0f;
    private float shadowDistanceRenderMul = -1.0f;
    private int noiseTextureResolution = DEFAULT_NOISE_RESOLUTION;
    private float ambientOcclusionLevel = 1.0f;
    private float centerDepthHalflife = 1.0f;
    private float eyeBrightnessHalflife = 10.0f;
    private float wetnessHalflife = 600.0f;
    private float drynessHalflife = 200.0f;

    public ShaderPackProperties() {
    }

    /**
     * Creates an empty {@link ShaderPackProperties} with standard OptiFine defaults.
     */
    public static ShaderPackProperties empty() {
        return new ShaderPackProperties();
    }

    /**
     * Alias for {@link #empty()}.
     */
    public static ShaderPackProperties defaults() {
        return empty();
    }

    /**
     * Parses a {@code shaders.properties} file string into a new {@link ShaderPackProperties}.
     */
    public static ShaderPackProperties parse(String propertiesText) {
        ShaderPackProperties props = new ShaderPackProperties();
        if (propertiesText != null && !propertiesText.isBlank()) {
            props.parsePropertiesText(propertiesText);
        }
        return props;
    }

    /**
     * Loads {@code shaders/shaders.properties} from a {@link ShaderPackSource} if present.
     */
    public static ShaderPackProperties load(ShaderPackSource source) {
        ShaderPackProperties props = new ShaderPackProperties();
        if (source != null) {
            source.readText("shaders/shaders.properties").ifPresent(props::parsePropertiesText);
        }
        return props;
    }

    /**
     * Parses {@code shaders.properties} content into this instance.
     */
    public void parsePropertiesText(String propertiesText) {
        if (propertiesText == null || propertiesText.isEmpty()) {
            return;
        }
        List<String> logicalLines = joinLogicalLines(propertiesText);
        for (String rawLine : logicalLines) {
            String trimmed = rawLine.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("#") || trimmed.startsWith("//")) {
                continue;
            }
            int eqIdx = findKeyValueSeparator(trimmed);
            if (eqIdx <= 0) {
                continue;
            }
            String key = trimmed.substring(0, eqIdx).trim();
            String value = stripInlineComment(trimmed.substring(eqIdx + 1)).trim();
            if (key.isEmpty()) {
                continue;
            }
            rawProperties.put(key, value);
            applyPropertyKeyValue(key, value);
        }
    }

    private void applyPropertyKeyValue(String key, String value) {
        if ("sliders".equals(key)) {
            for (String token : value.split("\\s+")) {
                if (!token.isEmpty()) {
                    sliders.add(token);
                }
            }
            return;
        }
        if (key.startsWith("profile.")) {
            String profileName = key.substring("profile.".length()).trim();
            if (!profileName.isEmpty()) {
                Map<String, String> map = profiles.computeIfAbsent(profileName, k -> new LinkedHashMap<>());
                for (String token : value.split("\\s+")) {
                    if (token.isEmpty() || token.startsWith("profile.")) {
                        continue;
                    }
                    if (token.startsWith("!")) {
                        map.put(token.substring(1), "false");
                    } else {
                        int eq = token.indexOf('=');
                        int colon = token.indexOf(':');
                        int sep = (eq > 0 && colon > 0) ? Math.min(eq, colon) : Math.max(eq, colon);
                        if (sep > 0) {
                            map.put(token.substring(0, sep), token.substring(sep + 1));
                        } else {
                            map.put(token, "true");
                        }
                    }
                }
            }
            return;
        }
        if (key.startsWith("scale.")) {
            String pass = key.substring("scale.".length()).trim();
            String[] parts = value.split("\\s+");
            if (parts.length >= 1) {
                parseFloatSafe(parts[0]).ifPresent(scale -> passScales.put(pass, scale));
            }
            return;
        }
        if (key.startsWith("alphaTest.")) {
            String pass = key.substring("alphaTest.".length()).trim();
            passAlphaTests.put(pass, value);
            return;
        }
        if (key.startsWith("blend.")) {
            String rest = key.substring("blend.".length()).trim();
            passBlends.put(rest, value);
            return;
        }
        if (key.startsWith("flip.")) {
            String rest = key.substring("flip.".length()).trim();
            int dot = rest.indexOf('.');
            if (dot > 0) {
                String pass = rest.substring(0, dot).trim();
                String buf = canonicalizeBufferName(rest.substring(dot + 1).trim());
                boolean flipVal = "true".equalsIgnoreCase(value) || "1".equals(value);
                passFlips.computeIfAbsent(pass, k -> new LinkedHashMap<>()).put(buf, flipVal);
            }
            return;
        }
        if (key.endsWith("Format")) {
            String prefix = key.substring(0, key.length() - "Format".length());
            assignBufferFormat(prefix, value);
            return;
        }
        if (key.startsWith("format.")) {
            String buf = key.substring("format.".length()).trim();
            assignBufferFormat(buf, value);
        }
    }

    /**
     * Scans GLSL shader source (either raw or preprocessed) for OptiFine comment directives
     * ({@code DRAWBUFFERS}, {@code RENDERTARGETS}, {@code SHADOWRES}, {@code *FORMAT}) and
     * {@code const} directives ({@code colortex*Format}, {@code *MipmapEnabled}, {@code *Clear},
     * {@code shadowMapResolution}, {@code shadowDistance}, {@code sunPathRotation}).
     *
     * @param passName     optional pass name (e.g. {@code "gbuffers_terrain"}, {@code "final"})
     * @param shaderSource shader source or comment text
     */
    public void scanShaderDirectives(String passName, String shaderSource) {
        if (shaderSource == null || shaderSource.isEmpty()) {
            return;
        }

        // 1. Check DRAWBUFFERS / RENDERTARGETS
        Optional<int[]> explicitDrawBuffers = parseExplicitDrawBuffers(shaderSource);
        if (explicitDrawBuffers.isPresent() && passName != null && !passName.isBlank()) {
            passDrawBuffers.put(passName, explicitDrawBuffers.get());
        }

        // 2. Legacy comment directives /* GCOLORFORMAT:RGBA16F */, /* SHADOWRES:1024 */, /* SHADOWHPL:120.0 */
        Matcher legacyFmt = COMMENT_LEGACY_FORMAT_PATTERN.matcher(shaderSource);
        while (legacyFmt.find()) {
            assignBufferFormat(legacyFmt.group(1).toLowerCase(Locale.ROOT), legacyFmt.group(2).trim());
        }
        Matcher shadowResMatcher = COMMENT_SHADOWRES_PATTERN.matcher(shaderSource);
        while (shadowResMatcher.find()) {
            parseIntSafe(shadowResMatcher.group(1)).ifPresent(v -> this.shadowMapResolution = v);
        }
        Matcher shadowHplMatcher = COMMENT_SHADOWHPL_PATTERN.matcher(shaderSource);
        while (shadowHplMatcher.find()) {
            parseFloatSafe(shadowHplMatcher.group(1)).ifPresent(v -> this.shadowDistance = v);
        }

        // 3. const int *Format = ...;
        Matcher fmtMatcher = CONST_FORMAT_PATTERN.matcher(shaderSource);
        while (fmtMatcher.find()) {
            assignBufferFormat(fmtMatcher.group(1), fmtMatcher.group(2));
        }

        // 4. const bool *MipmapEnabled / *Mipmap = ...;
        Matcher mipMatcher = CONST_MIPMAP_PATTERN.matcher(shaderSource);
        while (mipMatcher.find()) {
            String target = mipMatcher.group(1);
            boolean enabled = Boolean.parseBoolean(mipMatcher.group(2));
            assignMipmapEnabled(target, enabled);
        }

        // 5. const bool *Clear = ...;
        Matcher clearMatcher = CONST_CLEAR_PATTERN.matcher(shaderSource);
        while (clearMatcher.find()) {
            String target = clearMatcher.group(1);
            boolean enabled = Boolean.parseBoolean(clearMatcher.group(2));
            assignClearEnabled(target, enabled);
        }

        // 6. const vec4 *ClearColor = vec4(...);
        Matcher clearColorMatcher = CONST_CLEAR_COLOR_PATTERN.matcher(shaderSource);
        while (clearColorMatcher.find()) {
            int idx = resolveColorTexIndex(clearColorMatcher.group(1));
            if (idx >= 0) {
                float[] rgba = parseVec4Components(clearColorMatcher.group(2));
                if (rgba != null) {
                    colorTexClearColors.put(idx, rgba);
                }
            }
        }

        // 7. const int shadowMapResolution / noiseTextureResolution = ...;
        Matcher intMatcher = CONST_INT_DIRECTIVE_PATTERN.matcher(shaderSource);
        while (intMatcher.find()) {
            String name = intMatcher.group(1);
            parseIntSafe(intMatcher.group(2)).ifPresent(val -> {
                if ("shadowMapResolution".equals(name) && val > 0) {
                    this.shadowMapResolution = val;
                } else if ("noiseTextureResolution".equals(name) && val > 0) {
                    this.noiseTextureResolution = val;
                }
            });
        }

        // 8. const float shadowDistance / sunPathRotation / ... = ...;
        Matcher floatMatcher = CONST_FLOAT_DIRECTIVE_PATTERN.matcher(shaderSource);
        while (floatMatcher.find()) {
            String name = floatMatcher.group(1);
            parseFloatSafe(floatMatcher.group(2)).ifPresent(val -> {
                switch (name) {
                    case "shadowDistance" -> this.shadowDistance = val;
                    case "sunPathRotation" -> this.sunPathRotation = val;
                    case "shadowIntervalSize" -> this.shadowIntervalSize = val;
                    case "shadowDistanceRenderMul" -> this.shadowDistanceRenderMul = val;
                    case "ambientOcclusionLevel" -> this.ambientOcclusionLevel = val;
                    case "centerDepthHalflife" -> this.centerDepthHalflife = val;
                    case "eyeBrightnessHalflife" -> this.eyeBrightnessHalflife = val;
                    case "wetnessHalflife" -> this.wetnessHalflife = val;
                    case "drynessHalflife" -> this.drynessHalflife = val;
                    default -> { }
                }
            });
        }
    }

    /**
     * Alias for {@link #scanShaderDirectives(String, String)}.
     */
    public void mergeFromShaderSource(String passName, String shaderSource) {
        scanShaderDirectives(passName, shaderSource);
    }

    /**
     * Extracts the active {@code /* DRAWBUFFERS:... * /} or {@code /* RENDERTARGETS:... * /}
     * array from a shader source string, or returns {@code new int[]{0}} if none is specified.
     */
    public static int[] parseDrawBuffers(String shaderSource) {
        return parseExplicitDrawBuffers(shaderSource).orElseGet(() -> new int[]{0});
    }

    /**
     * Extracts the explicit {@code /* DRAWBUFFERS:... * /} or {@code /* RENDERTARGETS:... * /}
     * array if present in {@code shaderSource}, or {@link Optional#empty()} if no directive exists.
     */
    public static Optional<int[]> parseExplicitDrawBuffers(String shaderSource) {
        if (shaderSource == null || shaderSource.isEmpty()) {
            return Optional.empty();
        }
        int[] result = null;

        Matcher rtMatcher = RENDERTARGETS_PATTERN.matcher(shaderSource);
        while (rtMatcher.find()) {
            int[] parsed = parseRenderTargetsSpec(rtMatcher.group(1));
            if (parsed.length > 0) {
                result = parsed;
            }
        }
        if (result != null) {
            return Optional.of(result);
        }

        Matcher dbMatcher = DRAWBUFFERS_PATTERN.matcher(shaderSource);
        while (dbMatcher.find()) {
            int[] parsed = parseDrawBuffersSpec(dbMatcher.group(1));
            if (parsed.length > 0) {
                result = parsed;
            }
        }
        return Optional.ofNullable(result);
    }

    static int[] parseDrawBuffersSpec(String spec) {
        if (spec == null || spec.isBlank()) {
            return new int[0];
        }
        String trimmed = spec.trim();
        List<Integer> list = new ArrayList<>(trimmed.length());
        for (int i = 0; i < trimmed.length(); i++) {
            char c = trimmed.charAt(i);
            if (c >= '0' && c <= '9') {
                list.add(c - '0');
            } else if (c >= 'A' && c <= 'F') {
                list.add(10 + (c - 'A'));
            } else if (c >= 'a' && c <= 'f') {
                list.add(10 + (c - 'a'));
            }
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    static int[] parseRenderTargetsSpec(String spec) {
        if (spec == null || spec.isBlank()) {
            return new int[0];
        }
        String trimmed = spec.trim();
        if (trimmed.contains(",") || trimmed.contains(" ")) {
            String[] parts = trimmed.split("[,\\s]+");
            List<Integer> list = new ArrayList<>(parts.length);
            for (String p : parts) {
                if (!p.isEmpty()) {
                    parseIntSafe(p).ifPresent(v -> {
                        if (v >= 0 && v <= 15) {
                            list.add(v);
                        }
                    });
                }
            }
            return list.stream().mapToInt(Integer::intValue).toArray();
        }
        // Single token without comma/space: if <= 15, treat as single render target index;
        // otherwise (e.g. /* RENDERTARGETS:012 */), split into single digits.
        Optional<Integer> single = parseIntSafe(trimmed);
        if (single.isPresent() && single.get() >= 0 && single.get() <= 15 && trimmed.length() <= 2 && !trimmed.startsWith("0")) {
            return new int[]{single.get()};
        }
        if ("0".equals(trimmed)) {
            return new int[]{0};
        }
        return parseDrawBuffersSpec(trimmed);
    }

    private void assignBufferFormat(String bufferToken, String formatValue) {
        if (bufferToken == null || formatValue == null || formatValue.isBlank()) {
            return;
        }
        String fmt = formatValue.trim();
        String lower = bufferToken.trim().toLowerCase(Locale.ROOT);
        if ("noisetex".equals(lower)) {
            this.noiseTexFormat = fmt;
            return;
        }
        if ("shadowcolor0".equals(lower) || "shadowcolor".equals(lower)) {
            this.shadowColorFormats.put(0, fmt);
            return;
        }
        if ("shadowcolor1".equals(lower)) {
            this.shadowColorFormats.put(1, fmt);
            return;
        }
        int idx = resolveColorTexIndex(lower);
        if (idx >= 0 && idx <= 15) {
            this.colorTexFormats.put(idx, fmt);
        }
    }

    private void assignMipmapEnabled(String targetToken, boolean enabled) {
        if (targetToken == null) {
            return;
        }
        String lower = targetToken.trim().toLowerCase(Locale.ROOT);
        if ("shadowtex0".equals(lower)) {
            shadowTexMipmapEnabled.put(0, enabled);
        } else if ("shadowtex1".equals(lower)) {
            shadowTexMipmapEnabled.put(1, enabled);
        } else if ("shadowcolor0".equals(lower)) {
            shadowColorMipmapEnabled.put(0, enabled);
        } else if ("shadowcolor1".equals(lower)) {
            shadowColorMipmapEnabled.put(1, enabled);
        } else {
            int idx = resolveColorTexIndex(lower);
            if (idx >= 0 && idx <= 15) {
                colorTexMipmapEnabled.put(idx, enabled);
            }
        }
    }

    private void assignClearEnabled(String targetToken, boolean enabled) {
        if (targetToken == null) {
            return;
        }
        String lower = targetToken.trim().toLowerCase(Locale.ROOT);
        if ("shadowcolor0".equals(lower)) {
            shadowColorClearEnabled.put(0, enabled);
        } else if ("shadowcolor1".equals(lower)) {
            shadowColorClearEnabled.put(1, enabled);
        } else {
            int idx = resolveColorTexIndex(lower);
            if (idx >= 0 && idx <= 15) {
                colorTexClearEnabled.put(idx, enabled);
            }
        }
    }

    public static int resolveColorTexIndex(String name) {
        if (name == null) {
            return -1;
        }
        String lower = name.trim().toLowerCase(Locale.ROOT);
        if (lower.startsWith("colortex")) {
            return parseIntSafe(lower.substring("colortex".length())).orElse(-1);
        }
        return switch (lower) {
            case "gcolor" -> 0;
            case "gdepth" -> 1;
            case "gnormal" -> 2;
            case "composite" -> 3;
            case "gaux1" -> 4;
            case "gaux2" -> 5;
            case "gaux3" -> 6;
            case "gaux4" -> 7;
            default -> -1;
        };
    }

    public static String canonicalizeBufferName(String name) {
        int idx = resolveColorTexIndex(name);
        if (idx >= 0) {
            return "colortex" + idx;
        }
        return name != null ? name.trim() : "";
    }

    private static float[] parseVec4Components(String insideParens) {
        if (insideParens == null) {
            return null;
        }
        String[] parts = insideParens.split(",");
        if (parts.length == 1) {
            Optional<Float> single = parseFloatSafe(parts[0].trim());
            if (single.isPresent()) {
                float v = single.get();
                return new float[]{v, v, v, v};
            }
            return null;
        }
        if (parts.length == 4) {
            float[] out = new float[4];
            for (int i = 0; i < 4; i++) {
                Optional<Float> val = parseFloatSafe(parts[i].trim());
                if (val.isEmpty()) {
                    return null;
                }
                out[i] = val.get();
            }
            return out;
        }
        return null;
    }

    private static List<String> joinLogicalLines(String text) {
        String[] rawLines = text.replace("\r\n", "\n").replace('\r', '\n').split("\n", -1);
        List<String> out = new ArrayList<>(rawLines.length);
        StringBuilder current = new StringBuilder();
        for (String line : rawLines) {
            String trimmedRight = line.stripTrailing();
            if (trimmedRight.endsWith("\\")) {
                current.append(trimmedRight, 0, trimmedRight.length() - 1).append(' ');
            } else {
                if (current.length() > 0) {
                    current.append(line.trim());
                    out.add(current.toString());
                    current.setLength(0);
                } else {
                    out.add(line);
                }
            }
        }
        if (current.length() > 0) {
            out.add(current.toString());
        }
        return out;
    }

    private static int findKeyValueSeparator(String line) {
        int eq = line.indexOf('=');
        int colon = line.indexOf(':');
        if (eq > 0 && colon > 0) {
            return Math.min(eq, colon);
        }
        return Math.max(eq, colon);
    }

    private static String stripInlineComment(String value) {
        int hash = value.indexOf('#');
        if (hash >= 0) {
            return value.substring(0, hash);
        }
        return value;
    }

    private static Optional<Integer> parseIntSafe(String s) {
        if (s == null || s.isBlank()) {
            return Optional.empty();
        }
        try {
            return Optional.of(Integer.parseInt(s.trim()));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    private static Optional<Float> parseFloatSafe(String s) {
        if (s == null || s.isBlank()) {
            return Optional.empty();
        }
        try {
            String clean = s.trim();
            if (clean.endsWith("f") || clean.endsWith("F")) {
                clean = clean.substring(0, clean.length() - 1);
            }
            return Optional.of(Float.parseFloat(clean));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    // --- Public Getters & Mutators ---

    public String getColorTexFormat(int index) {
        return colorTexFormats.getOrDefault(index, DEFAULT_COLOR_FORMAT);
    }

    public String getColorTexFormatOrNull(int index) {
        return colorTexFormats.get(index);
    }

    public void setColorTexFormat(int index, String format) {
        if (index >= 0 && index <= 15 && format != null && !format.isBlank()) {
            colorTexFormats.put(index, format.trim());
        }
    }

    public Map<Integer, String> getColorTexFormats() {
        return Collections.unmodifiableMap(colorTexFormats);
    }

    public String getShadowColorFormat(int index) {
        return shadowColorFormats.getOrDefault(index, DEFAULT_COLOR_FORMAT);
    }

    public String getNoiseTexFormat() {
        return noiseTexFormat;
    }

    public boolean isMipmapEnabled(int colorTexIndex) {
        return colorTexMipmapEnabled.getOrDefault(colorTexIndex, false);
    }

    public boolean isColorTexMipmapEnabled(int colorTexIndex) {
        return isMipmapEnabled(colorTexIndex);
    }

    public boolean isMipmapEnabled(String bufferName) {
        if (bufferName == null) {
            return false;
        }
        String lower = bufferName.trim().toLowerCase(Locale.ROOT);
        if ("shadowtex0".equals(lower)) {
            return shadowTexMipmapEnabled.getOrDefault(0, false);
        }
        if ("shadowtex1".equals(lower)) {
            return shadowTexMipmapEnabled.getOrDefault(1, false);
        }
        if ("shadowcolor0".equals(lower)) {
            return shadowColorMipmapEnabled.getOrDefault(0, false);
        }
        if ("shadowcolor1".equals(lower)) {
            return shadowColorMipmapEnabled.getOrDefault(1, false);
        }
        int idx = resolveColorTexIndex(lower);
        return idx >= 0 && isMipmapEnabled(idx);
    }

    public boolean isShadowTexMipmapEnabled(int index) {
        return shadowTexMipmapEnabled.getOrDefault(index, false);
    }

    public boolean isShadowColorMipmapEnabled(int index) {
        return shadowColorMipmapEnabled.getOrDefault(index, false);
    }

    public boolean isClearEnabled(int colorTexIndex) {
        return colorTexClearEnabled.getOrDefault(colorTexIndex, true);
    }

    public boolean isColorTexClearEnabled(int colorTexIndex) {
        return isClearEnabled(colorTexIndex);
    }

    public boolean isShadowColorClearEnabled(int index) {
        return shadowColorClearEnabled.getOrDefault(index, true);
    }

    public float[] getColorTexClearColor(int index) {
        float[] c = colorTexClearColors.get(index);
        return c != null ? c.clone() : null;
    }

    public int getShadowMapResolution() {
        return shadowMapResolution;
    }

    public void setShadowMapResolution(int shadowMapResolution) {
        if (shadowMapResolution > 0) {
            this.shadowMapResolution = shadowMapResolution;
        }
    }

    public float getShadowDistance() {
        return shadowDistance;
    }

    public void setShadowDistance(float shadowDistance) {
        if (shadowDistance > 0.0f) {
            this.shadowDistance = shadowDistance;
        }
    }

    public float getSunPathRotation() {
        return sunPathRotation;
    }

    public void setSunPathRotation(float sunPathRotation) {
        this.sunPathRotation = sunPathRotation;
    }

    public float getShadowIntervalSize() {
        return shadowIntervalSize;
    }

    public float getShadowDistanceRenderMul() {
        return shadowDistanceRenderMul;
    }

    public int getNoiseTextureResolution() {
        return noiseTextureResolution;
    }

    public float getAmbientOcclusionLevel() {
        return ambientOcclusionLevel;
    }

    public float getCenterDepthHalflife() {
        return centerDepthHalflife;
    }

    public float getEyeBrightnessHalflife() {
        return eyeBrightnessHalflife;
    }

    public float getWetnessHalflife() {
        return wetnessHalflife;
    }

    public float getDrynessHalflife() {
        return drynessHalflife;
    }

    public float getPassScale(String passName) {
        return passScales.getOrDefault(passName, 1.0f);
    }

    public Map<String, Float> getPassScales() {
        return Collections.unmodifiableMap(passScales);
    }

    public String getAlphaTest(String passName) {
        return passAlphaTests.get(passName);
    }

    public Map<String, String> getPassAlphaTests() {
        return Collections.unmodifiableMap(passAlphaTests);
    }

    public String getBlend(String passName) {
        return passBlends.get(passName);
    }

    public String getBlend(String passName, String bufferName) {
        String specific = passBlends.get(passName + "." + canonicalizeBufferName(bufferName));
        if (specific != null) {
            return specific;
        }
        String rawSpecific = passBlends.get(passName + "." + bufferName);
        if (rawSpecific != null) {
            return rawSpecific;
        }
        return passBlends.get(passName);
    }

    public Map<String, String> getPassBlends() {
        return Collections.unmodifiableMap(passBlends);
    }

    public boolean isFlip(String passName, int colorTexIndex) {
        return isFlip(passName, "colortex" + colorTexIndex);
    }

    public boolean isFlip(String passName, String bufferName) {
        Map<String, Boolean> map = passFlips.get(passName);
        if (map == null) {
            return false;
        }
        Boolean val = map.get(canonicalizeBufferName(bufferName));
        if (val != null) {
            return val;
        }
        return map.getOrDefault(bufferName, false);
    }

    public int[] getDrawBuffers(String passName) {
        int[] dbs = passDrawBuffers.get(passName);
        return dbs != null ? dbs.clone() : new int[]{0};
    }

    public void setDrawBuffers(String passName, int[] drawBuffers) {
        if (passName != null && drawBuffers != null && drawBuffers.length > 0) {
            passDrawBuffers.put(passName, drawBuffers.clone());
        }
    }

    public Set<String> getSliders() {
        return Collections.unmodifiableSet(sliders);
    }

    public Map<String, Map<String, String>> getProfiles() {
        return Collections.unmodifiableMap(profiles);
    }

    public Map<String, String> getOptionDefaults() {
        return Collections.unmodifiableMap(optionDefaults);
    }

    public void recordOptionDefault(String optionName, String defaultValue) {
        if (optionName != null && defaultValue != null) {
            optionDefaults.putIfAbsent(optionName, defaultValue);
        }
    }

    public Map<String, String> getRawProperties() {
        return Collections.unmodifiableMap(rawProperties);
    }

    public String getProperty(String key) {
        return rawProperties.get(key);
    }

    @Override
    public String toString() {
        return "ShaderPackProperties{" +
                "colorTexFormats=" + colorTexFormats +
                ", shadowMapResolution=" + shadowMapResolution +
                ", shadowDistance=" + shadowDistance +
                ", sunPathRotation=" + sunPathRotation +
                ", mipmaps=" + colorTexMipmapEnabled +
                '}';
    }
}
