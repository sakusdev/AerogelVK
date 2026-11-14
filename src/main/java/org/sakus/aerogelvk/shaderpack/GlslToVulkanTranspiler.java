package org.sakus.aerogelvk.shaderpack;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Transpiles preprocessed OptiFine / Iris OpenGL GLSL ({@code #version 120}..{@code 460} compatibility)
 * into Vulkan 1.3 GLSL 460 suitable for {@code VkShaderCompiler.compileGlslToSpirv}.
 *
 * <p>Key transformations:
 * <ul>
 *     <li>Replaces legacy {@code #version} / {@code #extension} lines with {@code #version 460}</li>
 *     <li>Emits Set 0 ({@link ShaderUniformLayout} UBO + Valkyrie 64-bit {@code MegaVertexSSBO} /
 *         {@code ChunkMetaSSBO} / {@code VisibleChunkIdSSBO} / {@code AtlasUVSSBO}) and Set 1
 *         ({@code colortex0..19}, {@code depthtex0..2}, {@code shadowtex0..1}, {@code shadowcolor0..1},
 *         {@code vk_gtexture}, {@code lightmap}, {@code noisetex}, {@code normals}, {@code specular})</li>
 *     <li>Rewrites {@code attribute} -> global vertex attributes unpacked from {@code MegaVertexSSBO}</li>
 *     <li>Rewrites {@code varying} -> {@code layout(location = ...)} {@code out}/{@code in} with
 *         deterministic slot allocation (accounting for matrix/array slot widths)</li>
 *     <li>Handles {@code gbuffers_*} / {@code shadow} vertex pulling and {@code deferred*} /
 *         {@code composite*} / {@code final} fullscreen triangle generation</li>
 *     <li>Rewrites {@code gl_FragData[N]} / {@code gl_FragColor} to active {@code DRAWBUFFERS}
 *         outputs and aliases legacy samplers while avoiding GLSL 460 {@code texture} keyword collisions</li>
 * </ul>
 */
public final class GlslToVulkanTranspiler {

    static {
        // Large shaderpacks (e.g. Complementary gbuffers_terrain.fsh ~105 KB) exceed LWJGL's 64 KB MemoryStack default.
        if (System.getProperty("org.lwjgl.system.stackSize") == null) {
            System.setProperty("org.lwjgl.system.stackSize", "512");
        }
    }

    /** Pipeline stage classification determining vertex pulling vs fullscreen synthesis and matrix bindings. */
    public enum PassKind {
        GBUFFERS,
        SHADOW,
        DEFERRED,
        COMPOSITE,
        FINAL
    }

    /** Paired vertex + fragment transpilation result with guaranteed matching varying locations. */
    public static final class TranspiledProgram {
        private final String vertexGlsl;
        private final String fragmentGlsl;
        private final int colorAttachmentCount;

        public TranspiledProgram(String vertexGlsl, String fragmentGlsl, int colorAttachmentCount) {
            this.vertexGlsl = vertexGlsl;
            this.fragmentGlsl = fragmentGlsl;
            this.colorAttachmentCount = colorAttachmentCount;
        }

        public String getVertexGlsl() {
            return vertexGlsl;
        }

        public String getFragmentGlsl() {
            return fragmentGlsl;
        }

        public int getColorAttachmentCount() {
            return colorAttachmentCount;
        }
    }

    private static final Set<String> SAMPLER_TYPES = Set.of(
            "sampler1D", "sampler2D", "sampler3D", "samplerCube",
            "sampler2DRect", "sampler1DShadow", "sampler2DShadow", "samplerCubeShadow",
            "sampler2DArray", "sampler2DArrayShadow",
            "isampler1D", "isampler2D", "isampler3D", "isamplerCube", "isampler2DArray",
            "usampler1D", "usampler2D", "usampler3D", "usamplerCube", "usampler2DArray"
    );

    private static final Set<String> CANONICAL_SAMPLERS = Set.of(
            "colortex0", "colortex1", "colortex2", "colortex3",
            "colortex4", "colortex5", "colortex6", "colortex7",
            "colortex8", "colortex9", "colortex10", "colortex11",
            "colortex12", "colortex13", "colortex14", "colortex15",
            "colortex16", "colortex17", "colortex18", "colortex19",
            "depthtex0", "depthtex1", "depthtex2",
            "shadowtex0", "shadowtex1",
            "shadowcolor0", "shadowcolor1",
            "vk_gtexture", "lightmap", "noisetex", "normals", "specular"
    );

    private static final Map<String, String> SAMPLER_ALIASES;

    static {
        Map<String, String> aliases = new HashMap<>();
        aliases.put("gcolor", "colortex0");
        aliases.put("gdepth", "colortex1");
        aliases.put("gnormal", "colortex2");
        aliases.put("composite", "colortex3");
        aliases.put("gaux1", "colortex4");
        aliases.put("gaux2", "colortex5");
        aliases.put("gaux3", "colortex6");
        aliases.put("gaux4", "colortex7");
        aliases.put("gdepthtex", "depthtex0");
        aliases.put("depthtex", "depthtex0");
        aliases.put("shadow", "shadowtex0");
        aliases.put("watershadow", "shadowtex1");
        aliases.put("gtexture", "vk_gtexture");
        aliases.put("tex", "vk_gtexture");
        aliases.put("texture", "vk_gtexture");
        SAMPLER_ALIASES = Collections.unmodifiableMap(aliases);
    }

    /**
     * Resolves legacy OptiFine sampler aliases according to pass stage:
     * {@code texture}/{@code gtexture}/{@code tex} binds the block/entity atlas
     * ({@code vk_gtexture}) in geometry/shadow passes, and {@code colortex0}
     * ({@code gcolor}) in fullscreen deferred/composite/final passes.
     */
    private static String resolveSamplerAlias(String name, PassKind passKind) {
        if ("texture".equals(name) || "gtexture".equals(name) || "tex".equals(name) || "gcolor".equals(name)) {
            return (passKind == PassKind.GBUFFERS || passKind == PassKind.SHADOW)
                    ? "vk_gtexture"
                    : "colortex0";
        }
        return SAMPLER_ALIASES.get(name);
    }

    private static final Map<String, String> DERIVED_UNIFORMS;

    static {
        Map<String, String> derived = new HashMap<>();
        // Resolution & pixel size (MakeUp, PotatoShaders, VanillaPlus, Mellow, superDuperVanilla)
        derived.put("pixelSizeX", "(1.0 / max(vk_ubo.viewWidth, 1.0))");
        derived.put("pixelSizeY", "(1.0 / max(vk_ubo.viewHeight, 1.0))");
        derived.put("pixelWidth", "(1.0 / max(vk_ubo.viewWidth, 1.0))");
        derived.put("pixelHeight", "(1.0 / max(vk_ubo.viewHeight, 1.0))");
        derived.put("aspectRatioInverse", "(1.0 / max(vk_ubo.aspectRatio, 0.0001))");
        derived.put("fovYInverse", "(1.0 / max(abs(vk_ubo.gbufferProjection[1][1]), 0.0001))");
        derived.put("viewSize", "vec2(max(vk_ubo.viewWidth, 1.0), max(vk_ubo.viewHeight, 1.0))");
        derived.put("resolution", "vec2(max(vk_ubo.viewWidth, 1.0), max(vk_ubo.viewHeight, 1.0))");
        derived.put("pixelSize", "vec2(1.0 / max(vk_ubo.viewWidth, 1.0), 1.0 / max(vk_ubo.viewHeight, 1.0))");
        derived.put("resolutionInv", "vec2(1.0 / max(vk_ubo.viewWidth, 1.0), 1.0 / max(vk_ubo.viewHeight, 1.0))");
        derived.put("atlasSize", "textureSize(vk_gtexture, 0)");

        // Normalized view-space and player-space celestial/up vectors
        derived.put("sunDirView", "vk_safeNorm(vk_ubo.sunPosition)");
        derived.put("sunvecView", "vk_safeNorm(vk_ubo.sunPosition)");
        derived.put("sunPosN", "vk_safeNorm(vk_ubo.sunPosition)");
        derived.put("moonDirView", "vk_safeNorm(vk_ubo.moonPosition)");
        derived.put("moonvecView", "vk_safeNorm(vk_ubo.moonPosition)");
        derived.put("lightDirView", "vk_safeNorm(vk_ubo.shadowLightPosition)");
        derived.put("lightvecView", "vk_safeNorm(vk_ubo.shadowLightPosition)");
        derived.put("slpN", "vk_safeNorm(vk_ubo.shadowLightPosition)");
        derived.put("upDirView", "vk_safeNorm(vk_ubo.upPosition)");
        derived.put("upvecView", "vk_safeNorm(vk_ubo.upPosition)");
        derived.put("sunDir", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.sunPosition, 0.0)).xyz)");
        derived.put("sunvec", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.sunPosition, 0.0)).xyz)");
        derived.put("moonDir", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.moonPosition, 0.0)).xyz)");
        derived.put("moonvec", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.moonPosition, 0.0)).xyz)");
        derived.put("lightDir", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.shadowLightPosition, 0.0)).xyz)");
        derived.put("lightvec", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.shadowLightPosition, 0.0)).xyz)");
        derived.put("slpPlayerN", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.shadowLightPosition, 0.0)).xyz)");
        derived.put("cloudLightDir", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.shadowLightPosition, 0.0)).xyz)");
        derived.put("cloudLightDirView", "vk_safeNorm(vk_ubo.shadowLightPosition)");
        derived.put("upDir", "vec3(0.0, 1.0, 0.0)");
        derived.put("upvec", "vec3(0.0, 1.0, 0.0)");
        derived.put("lightFlip", "clamp(abs((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.sunPosition, 0.0)).y) * 0.25, 0.0, 1.0)");
        derived.put("cloudLightFlip", "clamp(abs((vk_ubo.gbufferModelViewInverse * vec4(vk_ubo.sunPosition, 0.0)).y) * 0.25, 0.0, 1.0)");
        derived.put("playerLookVector", "vk_safeNorm((vk_ubo.gbufferModelViewInverse * vec4(0.0, 0.0, -1.0, 0.0)).xyz)");

        // Time-of-day, lighting, exposure, and celestial cycle uniforms
        derived.put("dayNightMix", "clamp(vk_ubo.sunPosition.y * 0.02 + 0.5, 0.0, 1.0)");
        derived.put("night", "(vk_ubo.sunPosition.y < 0.0 ? 1.0 : 0.0)");
        derived.put("timeAngle", "fract(float(vk_ubo.worldTime) / 24000.0)");
        derived.put("timeBrightness", "max(sin(vk_ubo.sunAngle * 6.2831853), 0.0)");
        derived.put("shadowFade", "clamp(1.0 - (abs(abs(vk_ubo.sunAngle - 0.5) - 0.25) - 0.23) * 100.0, 0.0, 1.0)");
        derived.put("shdFade", "clamp(1.0 - (abs(abs(vk_ubo.sunAngle - 0.5) - 0.25) - 0.23) * 100.0, 0.0, 1.0)");
        derived.put("daytime", "vec4(clamp(1.0 - abs(vk_ubo.sunAngle) * 12.0, 0.0, 1.0), clamp(1.0 - max(abs(vk_ubo.sunAngle - 0.25) - 0.12, 0.0) * 10.0, 0.0, 1.0), clamp(1.0 - abs(vk_ubo.sunAngle - 0.5) * 12.0, 0.0, 1.0), clamp(1.0 - max(abs(vk_ubo.sunAngle - 0.75) - 0.15, 0.0) * 10.0, 0.0, 1.0))");
        derived.put("dayStrength", "max(sin(vk_ubo.sunAngle * 6.2831853), 0.0)");
        derived.put("nightStrength", "max(-sin(vk_ubo.sunAngle * 6.2831853), 0.0)");
        derived.put("sunriseStrength", "clamp(1.0 - abs(vk_ubo.sunAngle) * 16.0, 0.0, 1.0)");
        derived.put("sunsetStrength", "clamp(1.0 - abs(vk_ubo.sunAngle - 0.5) * 16.0, 0.0, 1.0)");
        derived.put("dayCycle", "vk_ubo.sunAngle");
        derived.put("dayCycleAdjust", "vk_ubo.sunAngle");
        derived.put("sunAngleAtHome", "vk_ubo.sunAngle");
        derived.put("twilightPhase", "max(clamp(1.0 - abs(vk_ubo.sunAngle) * 16.0, 0.0, 1.0), clamp(1.0 - abs(vk_ubo.sunAngle - 0.5) * 16.0, 0.0, 1.0))");
        derived.put("celestialCosX", "cos(vk_ubo.sunAngle * 6.2831853)");
        derived.put("celestialSinX", "sin(vk_ubo.sunAngle * 6.2831853)");
        derived.put("dayMoment", "(float(vk_ubo.worldTime) / 24000.0)");
        derived.put("dayMixer", "clamp(-((float(vk_ubo.worldTime) / 24000.0 - 0.25) * (float(vk_ubo.worldTime) / 24000.0 - 0.25)) * 20.0 + 1.25, 0.0, 1.0)");
        derived.put("nightMixer", "clamp(-((float(vk_ubo.worldTime) / 24000.0 - 0.75) * (float(vk_ubo.worldTime) / 24000.0 - 0.75)) * 50.0 + 3.125, 0.0, 1.0)");
        derived.put("volumetricDayMixer", "1.0");
        derived.put("exposureLevel", "1.0");
        derived.put("worldAnimTime", "(float(vk_ubo.worldDay % 48) + float(vk_ubo.worldTime) / 24000.0)");
        derived.put("softLod", "max(0.0, log2(max(1.0, vk_ubo.viewHeight / 128.0)))");

        // Frame counters, eye/sky smoothness, environment flags
        derived.put("framemod2", "float(vk_ubo.frameCounter % 2)");
        derived.put("framemod4", "float(vk_ubo.frameCounter % 4)");
        derived.put("framemod8", "float(vk_ubo.frameCounter % 8)");
        derived.put("framemod600", "float(vk_ubo.frameCounter % 600)");
        derived.put("frameMod", "(vk_ubo.frameCounter % 16)");
        derived.put("frameFract", "(float(vk_ubo.frameCounter % 8) * 0.20225425)");
        derived.put("frameTimeSmooth", "vk_ubo.frameTime");
        derived.put("vertexFrameTime", "vk_ubo.frameTimeCounter");
        derived.put("fragmentFrameTime", "vk_ubo.frameTimeCounter");
        derived.put("eyeBrightnessM", "(float(vk_ubo.eyeBrightness.y) / 240.0)");
        derived.put("eyeBrightnessM2", "(vk_ubo.eyeBrightness.y > 239 ? 1.0 : 0.0)");
        derived.put("eyeSkylight", "(float(vk_ubo.eyeBrightness.y) / 240.0)");
        derived.put("isOutdoorsSmooth", "(vk_ubo.eyeBrightness.y > 50 ? 1.0 : 0.0)");
        derived.put("isEyeInCave", "(vk_ubo.isEyeInWater == 0 ? (1.0 - float(vk_ubo.eyeBrightness.y) / 240.0) : 0.0)");
        derived.put("eyeAltitude", "vk_ubo.cameraPosition.y");
        derived.put("eyePosition", "vk_ubo.cameraPosition");
        derived.put("cameraPositionFract", "fract(vk_ubo.cameraPosition)");
        derived.put("cameraPositionInt", "ivec3(floor(vk_ubo.cameraPosition))");
        derived.put("previousCameraPositionFract", "fract(vk_ubo.previousCameraPosition)");
        derived.put("previousCameraPositionInt", "ivec3(floor(vk_ubo.previousCameraPosition))");
        derived.put("relativeEyePosition", "vec3(0.0)");
        derived.put("camPosDelta", "(vk_ubo.cameraPosition - vk_ubo.previousCameraPosition)");
        derived.put("rainFactor", "vk_ubo.rainStrength");
        derived.put("inDry", "(1.0 - vk_ubo.rainStrength)");
        derived.put("inRainy", "vk_ubo.rainStrength");
        derived.put("darknessLightFactor", "clamp(vk_ubo.darknessFactor, 0.0, 1.0)");
        derived.put("maxBlindnessDarkness", "max(vk_ubo.blindness, vk_ubo.darknessFactor)");
        derived.put("effectFactor", "max(vk_ubo.blindness, vk_ubo.darknessFactor)");
        derived.put("blindFactor", "(clamp(vk_ubo.blindness * 2.0 - 1.0, 0.0, 1.0) * clamp(vk_ubo.blindness * 2.0 - 1.0, 0.0, 1.0))");
        derived.put("fogAmount", "(max(-sin(vk_ubo.sunAngle * 6.2831853), 0.0) + 1.0 + vk_ubo.rainStrength)");
        derived.put("starter", "1.0");
        derived.put("fogMode", "9729");
        derived.put("cloudHeight", "192.0");
        derived.put("bedrockLevel", "-64");
        derived.put("heightLimit", "320");
        derived.put("isRightHanded", "true");
        derived.put("volumetricCloudFar", "(vk_ubo.far * 8.0)");
        derived.put("borderFar", "vk_ubo.far");
        derived.put("currentColorSpace", "0");
        DERIVED_UNIFORMS = Collections.unmodifiableMap(derived);
    }

    private static String resolveDerivedUniformForToken(String name, PassKind passKind, Token prev, Token next) {
        String derived = resolveDerivedUniform(name, passKind);
        if (derived != null && prev != null && "[".equals(prev.text)
                && next != null && "]".equals(next.text) && derived.startsWith("float(")) {
            return "int(" + derived.substring("float(".length());
        }
        return derived;
    }

    private static String resolveDerivedUniform(String name, PassKind passKind) {
        return switch (name) {
            case "modelViewMatrix" -> switch (passKind) {
                case SHADOW -> "vk_ubo.shadowModelView";
                case GBUFFERS -> "vk_ubo.gbufferModelView";
                default -> "mat4(1.0)";
            };
            case "modelViewMatrixInverse" -> switch (passKind) {
                case SHADOW -> "vk_ubo.shadowModelViewInverse";
                case GBUFFERS -> "vk_ubo.gbufferModelViewInverse";
                default -> "mat4(1.0)";
            };
            case "projectionMatrix" -> switch (passKind) {
                case SHADOW -> "vk_ubo.shadowProjection";
                case GBUFFERS -> "vk_ubo.gbufferProjection";
                default -> "VK_FULLSCREEN_MVP";
            };
            case "projectionMatrixInverse" -> switch (passKind) {
                case SHADOW -> "vk_ubo.shadowProjectionInverse";
                case GBUFFERS -> "vk_ubo.gbufferProjectionInverse";
                default -> "VK_FULLSCREEN_MVP_INV";
            };
            case "normalMatrix" -> "vk_NormalMatrix";
            case "renderStage" -> (passKind == PassKind.GBUFFERS ? "1" : "0");
            default -> DERIVED_UNIFORMS.get(name);
        };
    }

    private static final Set<String> INTERP_QUALIFIERS = Set.of(
            "flat", "smooth", "noperspective", "centroid", "sample"
    );

    private static final Set<String> PRECISION_QUALIFIERS = Set.of(
            "highp", "mediump", "lowp"
    );

    private static final Set<String> GLSL_KEYWORDS = Set.of(
            "attribute", "const", "uniform", "varying", "buffer", "shared", "coherent",
            "volatile", "restrict", "readonly", "writeonly", "layout", "centroid", "flat",
            "smooth", "noperspective", "patch", "sample", "break", "continue", "do", "for",
            "while", "switch", "case", "default", "if", "else", "subroutine", "in", "out",
            "inout", "true", "false", "invariant", "precise", "discard", "return", "struct",
            "void", "precision", "highp", "mediump", "lowp"
    );

    private static final Set<String> KNOWN_GLSL_TYPES = Set.of(
            "void", "bool", "int", "uint", "float", "double",
            "vec2", "vec3", "vec4", "bvec2", "bvec3", "bvec4",
            "ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4",
            "dvec2", "dvec3", "dvec4",
            "mat2", "mat3", "mat4", "mat2x2", "mat2x3", "mat2x4",
            "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4"
    );

    private GlslToVulkanTranspiler() {
    }

    /**
     * Transpiles a preprocessed vertex shader source into Vulkan 1.3 GLSL 460.
     *
     * @param preprocessedSource preprocessed {@code .vsh} GLSL source
     * @param passKind           pass classification ({@link PassKind})
     * @return Vulkan 1.3 GLSL 460 vertex shader source
     */
    public static String transpileVertex(String preprocessedSource, PassKind passKind) {
        if (preprocessedSource == null) {
            throw new IllegalArgumentException("preprocessedSource must not be null");
        }
        PassKind kind = passKind != null ? passKind : PassKind.GBUFFERS;
        ParsedShader parsed = parseShader(preprocessedSource, true);
        Map<String, Integer> varyingLocations = assignVaryingLocations(parsed.varyings, null);
        return emitShader(parsed, true, kind, new int[0], varyingLocations, Collections.emptyList(), Collections.emptyMap());
    }

    /**
     * Transpiles a preprocessed fragment shader source into Vulkan 1.3 GLSL 460.
     *
     * @param preprocessedSource preprocessed {@code .fsh} GLSL source
     * @param passKind           pass classification ({@link PassKind})
     * @param drawbuffers        active {@code DRAWBUFFERS} indices (defaults to {@code [0]} if null/empty for non-shadow passes)
     * @return Vulkan 1.3 GLSL 460 fragment shader source
     */
    public static String transpileFragment(String preprocessedSource, PassKind passKind, int[] drawbuffers) {
        if (preprocessedSource == null) {
            throw new IllegalArgumentException("preprocessedSource must not be null");
        }
        PassKind kind = passKind != null ? passKind : PassKind.GBUFFERS;
        int[] effectiveDb = normalizeDrawbuffers(kind, drawbuffers);
        ParsedShader parsed = parseShader(preprocessedSource, false);
        Map<String, Integer> varyingLocations = assignVaryingLocations(parsed.varyings, null);
        return emitShader(parsed, false, kind, effectiveDb, varyingLocations, Collections.emptyList(), Collections.emptyMap());
    }

    /**
     * Transpiles a paired vertex + fragment program, guaranteeing identical {@code layout(location = ...)}
     * assignments for all varyings across both stages and matching output attachment counts.
     */
    public static TranspiledProgram transpileProgram(
            String preprocessedVertex,
            String preprocessedFragment,
            PassKind passKind,
            int[] drawbuffers
    ) {
        return transpileProgram(preprocessedVertex, preprocessedFragment, passKind, drawbuffers, Collections.emptyMap());
    }

    public static TranspiledProgram transpileProgram(
            String preprocessedVertex,
            String preprocessedFragment,
            PassKind passKind,
            int[] drawbuffers,
            Map<String, ShaderPackProperties.CustomUniform> customUniforms
    ) {
        if (preprocessedVertex == null) {
            throw new IllegalArgumentException("preprocessedVertex must not be null");
        }
        if (preprocessedFragment == null) {
            throw new IllegalArgumentException("preprocessedFragment must not be null");
        }
        PassKind kind = passKind != null ? passKind : PassKind.GBUFFERS;
        int[] effectiveDb = normalizeDrawbuffers(kind, drawbuffers);
        Map<String, ShaderPackProperties.CustomUniform> effectiveCustomUniforms =
                customUniforms != null ? customUniforms : Collections.emptyMap();

        ParsedShader vertParsed = parseShader(preprocessedVertex, true);
        ParsedShader fragParsed = parseShader(preprocessedFragment, false);
        promoteCustomUniformsOverDerived(vertParsed, effectiveCustomUniforms);
        promoteCustomUniformsOverDerived(fragParsed, effectiveCustomUniforms);

        Map<String, Integer> varyingLocations = assignVaryingLocations(vertParsed.varyings, fragParsed.varyings);

        List<VarDecl> missingVertOutputs = new ArrayList<>();
        for (VarDecl fragVar : fragParsed.varyings.values()) {
            if (!vertParsed.varyings.containsKey(fragVar.name)) {
                missingVertOutputs.add(fragVar);
            }
        }

        String vertOut = emitShader(vertParsed, true, kind, effectiveDb, varyingLocations, missingVertOutputs, effectiveCustomUniforms);
        String fragOut = emitShader(fragParsed, false, kind, effectiveDb, varyingLocations, Collections.emptyList(), effectiveCustomUniforms);
        int attachmentCount = computeAttachmentCount(fragParsed, effectiveDb);
        return new TranspiledProgram(vertOut, fragOut, attachmentCount);
    }

    private static int[] normalizeDrawbuffers(PassKind kind, int[] drawbuffers) {
        if (drawbuffers == null) {
            return kind == PassKind.SHADOW ? new int[0] : new int[]{0};
        }
        if (drawbuffers.length == 0) {
            return kind == PassKind.SHADOW ? new int[0] : new int[]{0};
        }
        return drawbuffers.clone();
    }

    private static int computeAttachmentCount(ParsedShader fragParsed, int[] drawbuffers) {
        if (!fragParsed.usesLegacyFragOutputs && !fragParsed.customFragOuts.isEmpty()) {
            int total = 0;
            for (VarDecl out : fragParsed.customFragOuts.values()) {
                total += locationSlotCount(out.type, out.arraySuffix);
            }
            return Math.min(8, Math.max(1, total));
        }
        return Math.min(8, drawbuffers.length);
    }

    // ------------------------------------------------------------------------
    // Lexing & Comment/Directive Stripping
    // ------------------------------------------------------------------------

    private enum TokenKind {
        WORD,
        NUMBER,
        PUNCT,
        PREPROC,
        NEWLINE
    }

    private static final class Token {
        final TokenKind kind;
        final String text;

        Token(TokenKind kind, String text) {
            this.kind = kind;
            this.text = text;
        }
    }

    private static String stripCommentsAndLegacyDirectives(String raw) {
        String normalized = raw.replace("\r\n", "\n").replace('\r', '\n');
        StringBuilder noComments = new StringBuilder(normalized.length());
        int n = normalized.length();
        int i = 0;
        while (i < n) {
            char c = normalized.charAt(i);
            if (c == '/' && i + 1 < n) {
                char next = normalized.charAt(i + 1);
                if (next == '/') {
                    i += 2;
                    while (i < n && normalized.charAt(i) != '\n') {
                        i++;
                    }
                    continue;
                } else if (next == '*') {
                    i += 2;
                    while (i < n) {
                        char bc = normalized.charAt(i);
                        if (bc == '\n') {
                            noComments.append('\n');
                            i++;
                        } else if (bc == '*' && i + 1 < n && normalized.charAt(i + 1) == '/') {
                            i += 2;
                            break;
                        } else {
                            i++;
                        }
                    }
                    continue;
                }
            }
            noComments.append(c);
            i++;
        }

        // Strip legacy #version and #extension lines
        StringBuilder filtered = new StringBuilder(noComments.length());
        String[] lines = noComments.toString().split("\n", -1);
        for (int lineIdx = 0; lineIdx < lines.length; lineIdx++) {
            String line = lines[lineIdx];
            String trimmed = line.trim();
            if (trimmed.startsWith("#")) {
                String afterHash = trimmed.substring(1).trim();
                if (afterHash.startsWith("version") || afterHash.startsWith("extension")) {
                    if (lineIdx < lines.length - 1) {
                        filtered.append('\n');
                    }
                    continue;
                }
            }
            filtered.append(line);
            if (lineIdx < lines.length - 1) {
                filtered.append('\n');
            }
        }
        return filtered.toString();
    }

    private static List<Token> tokenize(String src) {
        List<Token> tokens = new ArrayList<>();
        int n = src.length();
        int i = 0;
        while (i < n) {
            char c = src.charAt(i);
            if (c == '\n') {
                tokens.add(new Token(TokenKind.NEWLINE, "\n"));
                i++;
                continue;
            }
            if (Character.isWhitespace(c)) {
                i++;
                continue;
            }
            if (c == '#') {
                int start = i;
                while (i < n && src.charAt(i) != '\n') {
                    i++;
                }
                tokens.add(new Token(TokenKind.PREPROC, src.substring(start, i)));
                continue;
            }
            if (isIdentStart(c)) {
                int start = i;
                i++;
                while (i < n && isIdentPart(src.charAt(i))) {
                    i++;
                }
                tokens.add(new Token(TokenKind.WORD, src.substring(start, i)));
                continue;
            }
            if (Character.isDigit(c) || (c == '.' && i + 1 < n && Character.isDigit(src.charAt(i + 1)))) {
                int start = i;
                if (c == '0' && i + 1 < n && (src.charAt(i + 1) == 'x' || src.charAt(i + 1) == 'X')) {
                    i += 2;
                    while (i < n && isHexDigit(src.charAt(i))) {
                        i++;
                    }
                    if (i < n && (src.charAt(i) == 'u' || src.charAt(i) == 'U')) {
                        i++;
                    }
                } else {
                    while (i < n && Character.isDigit(src.charAt(i))) {
                        i++;
                    }
                    if (i < n && src.charAt(i) == '.') {
                        i++;
                        while (i < n && Character.isDigit(src.charAt(i))) {
                            i++;
                        }
                    }
                    if (i < n && (src.charAt(i) == 'e' || src.charAt(i) == 'E')) {
                        i++;
                        if (i < n && (src.charAt(i) == '+' || src.charAt(i) == '-')) {
                            i++;
                        }
                        while (i < n && Character.isDigit(src.charAt(i))) {
                            i++;
                        }
                    }
                    if (i < n && (src.charAt(i) == 'f' || src.charAt(i) == 'F'
                            || src.charAt(i) == 'u' || src.charAt(i) == 'U'
                            || src.charAt(i) == 'l' || src.charAt(i) == 'L')) {
                        i++;
                    }
                }
                tokens.add(new Token(TokenKind.NUMBER, src.substring(start, i)));
                continue;
            }
            // Multi-char operators
            if (i + 1 < n) {
                String two = src.substring(i, i + 2);
                if (two.equals("++") || two.equals("--") || two.equals("==") || two.equals("!=")
                        || two.equals("<=") || two.equals(">=") || two.equals("&&") || two.equals("||")
                        || two.equals("^^") || two.equals("<<") || two.equals(">>") || two.equals("+=")
                        || two.equals("-=") || two.equals("*=") || two.equals("/=") || two.equals("%=")
                        || two.equals("&=") || two.equals("|=") || two.equals("^=")) {
                    tokens.add(new Token(TokenKind.PUNCT, two));
                    i += 2;
                    continue;
                }
            }
            tokens.add(new Token(TokenKind.PUNCT, String.valueOf(c)));
            i++;
        }
        return tokens;
    }

    private static boolean isIdentStart(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }

    private static boolean isIdentPart(char c) {
        return isIdentStart(c) || (c >= '0' && c <= '9');
    }

    private static boolean isHexDigit(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }

    // ------------------------------------------------------------------------
    // Top-Level Declaration Parsing
    // ------------------------------------------------------------------------

    private static final class VarDecl {
        final String interpQualifier;
        final String type;
        final String name;
        final String arraySuffix;
        final Integer explicitLocation;

        VarDecl(String interpQualifier, String type, String name, String arraySuffix, Integer explicitLocation) {
            this.interpQualifier = interpQualifier;
            this.type = type;
            this.name = name;
            this.arraySuffix = arraySuffix;
            this.explicitLocation = explicitLocation;
        }
    }

    private static final class ParsedShader {
        final List<Token> bodyTokens;
        final Map<String, VarDecl> unknownUniforms = new LinkedHashMap<>();
        final Map<String, String> unknownSamplers = new LinkedHashMap<>();
        final Set<String> shadowCompareSamplers = new HashSet<>();
        final Set<String> declaredDerivedUniforms = new HashSet<>();
        final Map<String, VarDecl> declaredDerivedUniformDecls = new LinkedHashMap<>();
        final Set<String> resolvedCustomUniformNames = new HashSet<>();
        final Map<String, VarDecl> attributes = new LinkedHashMap<>();
        final Map<String, VarDecl> varyings = new LinkedHashMap<>();
        final Map<String, VarDecl> customFragOuts = new LinkedHashMap<>();
        final Set<String> structNames = new HashSet<>();
        boolean usesLegacyFragOutputs = false;
        boolean hasUserRound = false;
        boolean hasUserFma = false;

        ParsedShader(List<Token> bodyTokens) {
            this.bodyTokens = bodyTokens;
        }
    }

    private static ParsedShader parseShader(String preprocessedSource, boolean isVertex) {
        String cleaned = stripCommentsAndLegacyDirectives(preprocessedSource);
        List<Token> tokens = tokenize(cleaned);
        List<Token> bodyTokens = new ArrayList<>(tokens.size());
        ParsedShader parsed = new ParsedShader(bodyTokens);

        int n = tokens.size();
        int braceDepth = 0;
        int parenDepth = 0;
        int i = 0;

        while (i < n) {
            Token tok = tokens.get(i);

            if (tok.kind == TokenKind.NEWLINE || tok.kind == TokenKind.PREPROC) {
                bodyTokens.add(tok);
                i++;
                continue;
            }

            if (braceDepth == 0 && parenDepth == 0) {
                // Collect top-level statement up to ';' or '{'
                if (tok.kind == TokenKind.WORD && tok.text.equals("struct")) {
                    // Record struct name and pass struct declaration through unchanged
                    int nextIdx = nextCodeToken(tokens, i + 1);
                    if (nextIdx < n && tokens.get(nextIdx).kind == TokenKind.WORD) {
                        parsed.structNames.add(tokens.get(nextIdx).text);
                    }
                    while (i < n) {
                        Token t = tokens.get(i);
                        bodyTokens.add(t);
                        if (t.kind == TokenKind.PUNCT && t.text.equals("{")) {
                            braceDepth++;
                        } else if (t.kind == TokenKind.PUNCT && t.text.equals("}")) {
                            if (braceDepth > 0) braceDepth--;
                        } else if (t.kind == TokenKind.PUNCT && t.text.equals(";") && braceDepth == 0) {
                            i++;
                            break;
                        }
                        i++;
                    }
                    continue;
                }

                // Scan ahead to find ';' or '{' at braceDepth == 0, parenDepth == 0
                int scan = i;
                int scanParen = 0;
                boolean hitBrace = false;
                boolean hitSemi = false;
                while (scan < n) {
                    Token st = tokens.get(scan);
                    if (st.kind == TokenKind.PUNCT) {
                        if (st.text.equals("(")) scanParen++;
                        else if (st.text.equals(")") && scanParen > 0) scanParen--;
                        else if (scanParen == 0 && st.text.equals("{")) {
                            hitBrace = true;
                            break;
                        } else if (scanParen == 0 && st.text.equals(";")) {
                            hitSemi = true;
                            break;
                        }
                    }
                    scan++;
                }

                if (hitSemi && !hitBrace) {
                    List<Token> stmt = tokens.subList(i, scan);
                    if (tryConsumeStorageDeclaration(stmt, isVertex, parsed)) {
                        i = scan + 1;
                        continue;
                    }
                }
            }

            if (tok.kind == TokenKind.PUNCT) {
                if (tok.text.equals("{")) braceDepth++;
                else if (tok.text.equals("}") && braceDepth > 0) braceDepth--;
                else if (tok.text.equals("(")) parenDepth++;
                else if (tok.text.equals(")") && parenDepth > 0) parenDepth--;
            }

            if (!isVertex && tok.kind == TokenKind.WORD
                    && (tok.text.equals("gl_FragData") || tok.text.equals("gl_FragColor"))) {
                parsed.usesLegacyFragOutputs = true;
            }
            if (braceDepth == 0 && tok.kind == TokenKind.WORD && (tok.text.equals("round") || tok.text.equals("fma"))) {
                Token prev = prevCodeToken(tokens, i);
                Token next = nextCodeTokenObj(tokens, i);
                if (prev != null && prev.kind == TokenKind.WORD
                        && (KNOWN_GLSL_TYPES.contains(prev.text) || parsed.structNames.contains(prev.text))
                        && next != null && next.kind == TokenKind.PUNCT && next.text.equals("(")) {
                    if (tok.text.equals("round")) {
                        parsed.hasUserRound = true;
                    } else {
                        parsed.hasUserFma = true;
                    }
                }
            }

            bodyTokens.add(tok);
            i++;
        }

        return parsed;
    }

    private static boolean tryConsumeStorageDeclaration(List<Token> stmt, boolean isVertex, ParsedShader parsed) {
        List<Token> code = new ArrayList<>();
        for (Token t : stmt) {
            if (t.kind != TokenKind.NEWLINE && t.kind != TokenKind.PREPROC) {
                code.add(t);
            }
        }
        if (code.isEmpty()) {
            return false;
        }

        int idx = 0;
        Integer explicitLocation = null;
        String storageKind = null; // uniform, attribute, varying, in, out
        String interp = null;

        // Parse leading layout(...), storage qualifiers, interpolation qualifiers, precision
        while (idx < code.size()) {
            Token t = code.get(idx);
            if (t.kind == TokenKind.WORD && t.text.equals("layout")) {
                int lp = idx + 1;
                if (lp < code.size() && code.get(lp).text.equals("(")) {
                    int rp = findMatchingParen(code, lp);
                    if (rp > lp) {
                        explicitLocation = parseLayoutLocation(code.subList(lp + 1, rp));
                        idx = rp + 1;
                        continue;
                    }
                }
            }
            if (t.kind == TokenKind.WORD) {
                String w = t.text;
                if (w.equals("uniform") || w.equals("attribute") || w.equals("varying")
                        || w.equals("in") || w.equals("out")) {
                    storageKind = w;
                    idx++;
                    continue;
                }
                if (INTERP_QUALIFIERS.contains(w)) {
                    if (w.equals("flat") || w.equals("noperspective")) {
                        interp = w;
                    }
                    idx++;
                    continue;
                }
                if (PRECISION_QUALIFIERS.contains(w) || w.equals("invariant") || w.equals("precise")) {
                    idx++;
                    continue;
                }
            }
            break;
        }

        if (storageKind == null || idx >= code.size()) {
            return false;
        }

        // Next token must be the type name
        Token typeTok = code.get(idx);
        if (typeTok.kind != TokenKind.WORD) {
            return false;
        }
        String typeName = typeTok.text;
        idx++;

        // Parse comma-separated declarators: name [array] [= init]
        List<VarDecl> decls = new ArrayList<>();
        while (idx < code.size()) {
            Token nameTok = code.get(idx);
            if (nameTok.kind != TokenKind.WORD) {
                return false;
            }
            String varName = nameTok.text;
            idx++;

            StringBuilder arraySuffix = new StringBuilder();
            while (idx < code.size() && code.get(idx).text.equals("[")) {
                int close = findMatchingBracket(code, idx);
                if (close < 0) return false;
                for (int k = idx; k <= close; k++) {
                    arraySuffix.append(code.get(k).text);
                }
                idx = close + 1;
            }

            // Skip optional '= initializer' at top level
            if (idx < code.size() && code.get(idx).text.equals("=")) {
                idx++;
                int depth = 0;
                while (idx < code.size()) {
                    String pt = code.get(idx).text;
                    if (pt.equals("(") || pt.equals("[") || pt.equals("{")) depth++;
                    else if ((pt.equals(")") || pt.equals("]") || pt.equals("}")) && depth > 0) depth--;
                    else if (pt.equals(",") && depth == 0) break;
                    idx++;
                }
            }

            decls.add(new VarDecl(interp, typeName, varName, arraySuffix.toString(), explicitLocation));

            if (idx < code.size() && code.get(idx).text.equals(",")) {
                idx++;
                continue;
            }
            break;
        }

        if (decls.isEmpty()) {
            return false;
        }

        switch (storageKind) {
            case "uniform" -> {
                for (VarDecl d : decls) {
                    if (SAMPLER_TYPES.contains(d.type)) {
                        if (d.type.contains("Shadow")) {
                            parsed.shadowCompareSamplers.add(d.name);
                            String alias = SAMPLER_ALIASES.get(d.name);
                            if (alias != null) {
                                parsed.shadowCompareSamplers.add(alias);
                            }
                        }
                        if (!CANONICAL_SAMPLERS.contains(d.name) && !SAMPLER_ALIASES.containsKey(d.name)) {
                            String fallback = d.name.toLowerCase(Locale.ROOT).contains("depth")
                                    ? "depthtex0" : "colortex0";
                            parsed.unknownSamplers.putIfAbsent(d.name, fallback);
                        }
                    } else if (resolveDerivedUniform(d.name, PassKind.GBUFFERS) != null) {
                        parsed.declaredDerivedUniforms.add(d.name);
                        parsed.declaredDerivedUniformDecls.putIfAbsent(d.name, d);
                    } else if (!ShaderUniformLayout.UBO_FIELD_NAMES.contains(d.name)) {
                        parsed.unknownUniforms.putIfAbsent(d.name, d);
                    }
                }
                return true;
            }
            case "attribute" -> {
                for (VarDecl d : decls) {
                    parsed.attributes.putIfAbsent(d.name, d);
                }
                return true;
            }
            case "in" -> {
                if (isVertex) {
                    for (VarDecl d : decls) {
                        parsed.attributes.putIfAbsent(d.name, d);
                    }
                } else {
                    for (VarDecl d : decls) {
                        parsed.varyings.putIfAbsent(d.name, ensureIntegerFlat(d));
                    }
                }
                return true;
            }
            case "varying" -> {
                for (VarDecl d : decls) {
                    parsed.varyings.putIfAbsent(d.name, ensureIntegerFlat(d));
                }
                return true;
            }
            case "out" -> {
                if (isVertex) {
                    for (VarDecl d : decls) {
                        parsed.varyings.putIfAbsent(d.name, ensureIntegerFlat(d));
                    }
                } else {
                    for (VarDecl d : decls) {
                        parsed.customFragOuts.putIfAbsent(d.name, d);
                    }
                }
                return true;
            }
            default -> {
                return false;
            }
        }
    }

    private static VarDecl ensureIntegerFlat(VarDecl d) {
        if (d.interpQualifier == null && isIntegerType(d.type)) {
            return new VarDecl("flat", d.type, d.name, d.arraySuffix, d.explicitLocation);
        }
        return d;
    }

    private static boolean isIntegerType(String type) {
        return type.equals("int") || type.equals("uint") || type.equals("bool")
                || type.startsWith("ivec") || type.startsWith("uvec") || type.startsWith("bvec");
    }

    private static Integer parseLayoutLocation(List<Token> layoutArgs) {
        for (int i = 0; i + 2 < layoutArgs.size(); i++) {
            if (layoutArgs.get(i).text.equals("location") && layoutArgs.get(i + 1).text.equals("=")) {
                try {
                    return Integer.parseInt(layoutArgs.get(i + 2).text);
                } catch (NumberFormatException ignored) {
                    return null;
                }
            }
        }
        return null;
    }

    private static Map<String, Integer> assignVaryingLocations(
            Map<String, VarDecl> primary,
            Map<String, VarDecl> secondary
    ) {
        Map<String, Integer> locations = new LinkedHashMap<>();
        int nextLoc = ShaderDescriptorBindings.LOC_USER_VARYING_BASE;

        if (primary != null) {
            for (VarDecl d : primary.values()) {
                if (!locations.containsKey(d.name)) {
                    locations.put(d.name, nextLoc);
                    nextLoc += locationSlotCount(d.type, d.arraySuffix);
                }
            }
        }
        if (secondary != null) {
            for (VarDecl d : secondary.values()) {
                if (!locations.containsKey(d.name)) {
                    locations.put(d.name, nextLoc);
                    nextLoc += locationSlotCount(d.type, d.arraySuffix);
                }
            }
        }
        return locations;
    }

    private static int locationSlotCount(String type, String arraySuffix) {
        int baseSlots;
        if (type.startsWith("mat4") || type.startsWith("dmat4")) {
            baseSlots = 4;
        } else if (type.startsWith("mat3") || type.startsWith("dmat3")) {
            baseSlots = 3;
        } else if (type.startsWith("mat2") || type.startsWith("dmat2")) {
            baseSlots = 2;
        } else {
            baseSlots = 1;
        }

        int arrayLen = 1;
        if (arraySuffix != null && !arraySuffix.isEmpty()) {
            int l = arraySuffix.indexOf('[');
            int r = arraySuffix.indexOf(']');
            if (l >= 0 && r > l + 1) {
                try {
                    arrayLen = Math.max(1, Integer.parseInt(arraySuffix.substring(l + 1, r).trim()));
                } catch (NumberFormatException ignored) {
                    arrayLen = 1;
                }
            }
        }
        return baseSlots * arrayLen;
    }

    // ------------------------------------------------------------------------
    // GLSL 460 Emission & Token Rewriting
    // ------------------------------------------------------------------------

    private static String emitShader(
            ParsedShader parsed,
            boolean isVertex,
            PassKind passKind,
            int[] drawbuffers,
            Map<String, Integer> varyingLocations,
            List<VarDecl> extraVertStubs,
            Map<String, ShaderPackProperties.CustomUniform> customUniforms
    ) {
        List<ResolvedCustomUniform> resolvedCustoms = resolveCustomUniforms(parsed, customUniforms, passKind);
        Set<String> resolvedCustomNames = new HashSet<>();
        for (ResolvedCustomUniform rc : resolvedCustoms) {
            resolvedCustomNames.add(rc.name());
        }

        StringBuilder out = new StringBuilder(4096);
        out.append("#version 460\n\n");

        // 1. Set 0 UBO & Terrain SSBOs (Vertex/Fragment)
        out.append(ShaderUniformLayout.GLSL_UBO_BLOCK).append('\n');

        if (isVertex && (passKind == PassKind.GBUFFERS || passKind == PassKind.SHADOW)) {
            emitGeometrySsboBindings(out);
        } else if (!isVertex && (passKind == PassKind.GBUFFERS || passKind == PassKind.SHADOW)) {
            emitAtlasSsboBinding(out);
        }

        // 2. Set 1 Samplers
        emitSet1Samplers(out);

        // 3. Built-in legacy synthetic inputs/outputs (must precede helpers referencing vk_SpriteMeta)
        emitLegacyBuiltins(out, isVertex);

        // 4. Fog struct & compatibility texture helpers
        emitCompatibilityHelpers(out, isVertex, passKind);

        // 5. Unknown uniforms -> custom uniform globals or constant zero defaults
        boolean emittedAnyUnknown = false;
        for (VarDecl u : parsed.unknownUniforms.values()) {
            if (!resolvedCustomNames.contains(u.name)) {
                out.append("const ").append(u.type).append(' ').append(u.name).append(u.arraySuffix)
                        .append(" = ").append(zeroInitExpr(u.type, u.arraySuffix)).append(";\n");
                emittedAnyUnknown = true;
            }
        }
        parsed.resolvedCustomUniformNames.clear();
        parsed.resolvedCustomUniformNames.addAll(resolvedCustomNames);
        for (ResolvedCustomUniform rc : resolvedCustoms) {
            out.append(rc.type()).append(" _vk_cu_").append(rc.name()).append("() { return ")
                    .append(rc.glslExpr()).append("; }\n");
            emittedAnyUnknown = true;
        }
        if (emittedAnyUnknown) {
            out.append('\n');
        }

        // 6. Vertex attributes -> globals unpacked in main()
        if (isVertex && !parsed.attributes.isEmpty()) {
            for (VarDecl a : parsed.attributes.values()) {
                out.append(a.type).append(' ').append(a.name).append(a.arraySuffix).append(";\n");
            }
            out.append('\n');
        }

        // 7. Varyings -> layout(location = ...) out / in
        String ioKeyword = isVertex ? "out" : "in";
        for (VarDecl v : parsed.varyings.values()) {
            int loc = varyingLocations.getOrDefault(v.name, ShaderDescriptorBindings.LOC_USER_VARYING_BASE);
            out.append("layout(location = ").append(loc).append(") ");
            if (v.interpQualifier != null) {
                out.append(v.interpQualifier).append(' ');
            }
            out.append(ioKeyword).append(' ').append(v.type).append(' ').append(v.name).append(v.arraySuffix).append(";\n");
        }
        if (isVertex) {
            for (VarDecl stub : extraVertStubs) {
                int loc = varyingLocations.getOrDefault(stub.name, ShaderDescriptorBindings.LOC_USER_VARYING_BASE);
                out.append("layout(location = ").append(loc).append(") ");
                if (stub.interpQualifier != null) {
                    out.append(stub.interpQualifier).append(' ');
                }
                out.append("out ").append(stub.type).append(' ').append(stub.name).append(stub.arraySuffix).append(";\n");
            }
        }
        out.append('\n');

        // 8. Fragment outputs
        boolean useFragDataWrapper = !isVertex && (parsed.usesLegacyFragOutputs || parsed.customFragOuts.isEmpty());
        int attachmentCount = Math.min(8, drawbuffers.length);
        if (!isVertex) {
            if (useFragDataWrapper) {
                for (int i = 0; i < attachmentCount; i++) {
                    out.append("layout(location = ").append(i).append(") out vec4 vk_FragOut").append(i).append(";\n");
                }
                out.append("vec4 vk_FragData[8];\n\n");
            } else {
                int nextOutLoc = 0;
                for (VarDecl fragOut : parsed.customFragOuts.values()) {
                    int loc = fragOut.explicitLocation != null ? fragOut.explicitLocation : nextOutLoc;
                    nextOutLoc = Math.max(nextOutLoc, loc + locationSlotCount(fragOut.type, fragOut.arraySuffix));
                    out.append("layout(location = ").append(loc).append(") out ")
                            .append(fragOut.type).append(' ').append(fragOut.name).append(fragOut.arraySuffix).append(";\n");
                }
                out.append('\n');
            }
        }

        // 9. Rewritten user code (with void main() -> void vk_user_main())
        Set<String> nonShadowRewritten = new HashSet<>(parsed.structNames);
        String rewrittenBody = rewriteBody(parsed, isVertex, passKind, nonShadowRewritten);
        out.append(rewrittenBody);
        if (!rewrittenBody.endsWith("\n")) {
            out.append('\n');
        }
        out.append('\n');

        // 10. Wrapper void main()
        if (isVertex) {
            emitVertexMainWrapper(out, passKind, parsed.attributes, extraVertStubs);
        } else {
            emitFragmentMainWrapper(out, useFragDataWrapper, attachmentCount);
        }

        return out.toString();
    }

    private static void emitGeometrySsboBindings(StringBuilder out) {
        out.append("layout(std430, set = ").append(ShaderDescriptorBindings.SET_GLOBAL)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_MEGA_VERTEX_SSBO)
                .append(") readonly buffer VkMegaVertexSSBO {\n")
                .append("    uvec2 vk_vertices[];\n")
                .append("};\n\n");

        out.append("struct VkChunkDrawMeta {\n")
                .append("    ivec3 chunkPos;\n")
                .append("    uint vertexOffset;\n")
                .append("    uint quadCount;\n")
                .append("    uint aabbPacked;\n")
                .append("    uint flags;\n")
                .append("    uint padding;\n")
                .append("};\n\n");

        out.append("layout(std430, set = ").append(ShaderDescriptorBindings.SET_GLOBAL)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_CHUNK_META_SSBO)
                .append(") readonly buffer VkChunkMetaSSBO {\n")
                .append("    VkChunkDrawMeta vk_chunks[];\n")
                .append("};\n\n");

        out.append("layout(std430, set = ").append(ShaderDescriptorBindings.SET_GLOBAL)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_VISIBLE_CHUNK_ID_SSBO)
                .append(") readonly buffer VkVisibleChunkIdSSBO {\n")
                .append("    uint vk_instanceToChunkIdx[];\n")
                .append("};\n\n");

        emitAtlasSsboBinding(out);

        out.append("const vec2 VK_CORNER_UVS[4] = vec2[4](\n")
                .append("    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)\n")
                .append(");\n\n");

        out.append("const vec3 VK_FACE_NORMALS[6] = vec3[6](\n")
                .append("    vec3( 1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0),\n")
                .append("    vec3( 0.0,  1.0,  0.0), vec3( 0.0, -1.0,  0.0),\n")
                .append("    vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0, -1.0)\n")
                .append(");\n\n");
    }

    private static void emitAtlasSsboBinding(StringBuilder out) {
        out.append("struct VkAtlasUVEntry {\n")
                .append("    vec2 uvMin;\n")
                .append("    vec2 uvMax;\n")
                .append("    vec4 tint;\n")
                .append("};\n\n");

        out.append("layout(std430, set = ").append(ShaderDescriptorBindings.SET_GLOBAL)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_ATLAS_UV_SSBO)
                .append(") readonly buffer VkAtlasUVSSBO {\n")
                .append("    VkAtlasUVEntry vk_uvTable[];\n")
                .append("};\n\n");
    }

    private static void emitSet1Samplers(StringBuilder out) {
        for (int i = 0; i < ShaderDescriptorBindings.COLORTEX_COUNT; i++) {
            out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                    .append(", binding = ").append(ShaderDescriptorBindings.colortexBinding(i))
                    .append(") uniform sampler2D colortex").append(i).append(";\n");
        }
        for (int i = 0; i < ShaderDescriptorBindings.DEPTHTEX_COUNT; i++) {
            out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                    .append(", binding = ").append(ShaderDescriptorBindings.depthtexBinding(i))
                    .append(") uniform sampler2D depthtex").append(i).append(";\n");
        }
        for (int i = 0; i < ShaderDescriptorBindings.SHADOWTEX_COUNT; i++) {
            out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                    .append(", binding = ").append(ShaderDescriptorBindings.shadowtexBinding(i))
                    .append(") uniform sampler2D shadowtex").append(i).append(";\n");
        }
        for (int i = 0; i < ShaderDescriptorBindings.SHADOWCOLOR_COUNT; i++) {
            out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                    .append(", binding = ").append(ShaderDescriptorBindings.shadowcolorBinding(i))
                    .append(") uniform sampler2D shadowcolor").append(i).append(";\n");
        }
        out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_GTEXTURE)
                .append(") uniform sampler2D vk_gtexture;\n");
        out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_LIGHTMAP)
                .append(") uniform sampler2D lightmap;\n");
        out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_NOISETEX)
                .append(") uniform sampler2D noisetex;\n");
        out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_NORMALS)
                .append(") uniform sampler2D normals;\n");
        out.append("layout(set = ").append(ShaderDescriptorBindings.SET_TEXTURES)
                .append(", binding = ").append(ShaderDescriptorBindings.BINDING_SPECULAR)
                .append(") uniform sampler2D specular;\n\n");
    }

    private static void emitCompatibilityHelpers(StringBuilder out, boolean isVertex, PassKind passKind) {
        out.append("struct VkFogParameters {\n")
                .append("    vec4 color;\n")
                .append("    float density;\n")
                .append("    float start;\n")
                .append("    float end;\n")
                .append("    float scale;\n")
                .append("};\n")
                .append("VkFogParameters vk_Fog;\n")
                .append("float vk_FogFragCoord = 0.0;\n")
                .append("mat3 vk_NormalMatrix;\n\n");

        // OptiFine fullscreen pass projection maps [0, 1] quad/triangle vertices to [-1, 1] NDC
        out.append("const mat4 VK_FULLSCREEN_MVP = mat4(\n")
                .append("    vec4(2.0, 0.0, 0.0, 0.0),\n")
                .append("    vec4(0.0, 2.0, 0.0, 0.0),\n")
                .append("    vec4(0.0, 0.0, 1.0, 0.0),\n")
                .append("    vec4(-1.0, -1.0, 0.0, 1.0)\n")
                .append(");\n\n");
        out.append("const mat4 VK_FULLSCREEN_MVP_INV = mat4(\n")
                .append("    vec4(0.5, 0.0, 0.0, 0.0),\n")
                .append("    vec4(0.0, 0.5, 0.0, 0.0),\n")
                .append("    vec4(0.0, 0.0, 1.0, 0.0),\n")
                .append("    vec4(0.5, 0.5, 0.0, 1.0)\n")
                .append(");\n\n");
        out.append("vec3 vk_safeNorm(vec3 v) { float l = dot(v, v); return l > 1e-8 ? v * inversesqrt(l) : vec3(0.0, 1.0, 0.0); }\n")
                .append("vec2 _vk_normalize(vec2 v) { float l = dot(v, v); return l > 1e-20 ? v * inversesqrt(l) : vec2(0.0); }\n")
                .append("vec3 _vk_normalize(vec3 v) { float l = dot(v, v); return l > 1e-20 ? v * inversesqrt(l) : vec3(0.0); }\n")
                .append("vec4 _vk_normalize(vec4 v) { float l = dot(v, v); return l > 1e-20 ? v * inversesqrt(l) : vec4(0.0); }\n\n");

        // Lightmap texture matrix maps [0..240] -> [8/256..248/256], matching Minecraft OpenGL
        out.append("const mat4 VK_LIGHTMAP_MATRIX = mat4(\n")
                .append("    vec4(1.0 / 256.0, 0.0, 0.0, 0.0),\n")
                .append("    vec4(0.0, 1.0 / 256.0, 0.0, 0.0),\n")
                .append("    vec4(0.0, 0.0, 1.0, 0.0),\n")
                .append("    vec4(8.0 / 256.0, 8.0 / 256.0, 0.0, 1.0)\n")
                .append(");\n\n");

        if (passKind == PassKind.GBUFFERS || passKind == PassKind.SHADOW) {
            out.append("const mat4 vk_TextureMatrix[8] = mat4[8](\n")
                    .append("    mat4(1.0), VK_LIGHTMAP_MATRIX, mat4(1.0), mat4(1.0),\n")
                    .append("    mat4(1.0), mat4(1.0), mat4(1.0), mat4(1.0)\n")
                    .append(");\n\n");
        } else {
            out.append("const mat4 vk_TextureMatrix[8] = mat4[8](\n")
                    .append("    mat4(1.0), mat4(1.0), mat4(1.0), mat4(1.0),\n")
                    .append("    mat4(1.0), mat4(1.0), mat4(1.0), mat4(1.0)\n")
                    .append(");\n\n");
        }

        // Legacy texture sampling functions -> GLSL 460 equivalents
        out.append("vec4 vk_texture2D(sampler2D s, vec2 c) { return texture(s, c); }\n")
                .append(isVertex
                        ? "vec4 vk_texture2D(sampler2D s, vec2 c, float bias) { return textureLod(s, c, bias); }\n"
                        : "vec4 vk_texture2D(sampler2D s, vec2 c, float bias) { return texture(s, c, bias); }\n")
                .append("vec4 vk_texture2DLod(sampler2D s, vec2 c, float lod) { return textureLod(s, c, lod); }\n")
                .append("vec4 vk_texture2DGrad(sampler2D s, vec2 c, vec2 ddx, vec2 ddy) { return textureGrad(s, c, ddx, ddy); }\n")
                .append("vec4 vk_texture2DProj(sampler2D s, vec3 c) { return textureProj(s, c); }\n")
                .append("vec4 vk_texture2DProj(sampler2D s, vec4 c) { return textureProj(s, c); }\n")
                .append("vec4 vk_texture2DProjLod(sampler2D s, vec3 c, float lod) { return textureProjLod(s, c, lod); }\n")
                .append("vec4 vk_texture2DProjLod(sampler2D s, vec4 c, float lod) { return textureProjLod(s, c, lod); }\n")
                .append("vec4 vk_texture3D(sampler3D s, vec3 c) { return texture(s, c); }\n")
                .append("vec4 vk_texture3D(sampler2D s, vec3 c) { return texture(s, c.xy); }\n")
                .append("vec4 vk_texture3DLod(sampler3D s, vec3 c, float lod) { return textureLod(s, c, lod); }\n")
                .append("vec4 vk_texture3DLod(sampler2D s, vec3 c, float lod) { return textureLod(s, c.xy, lod); }\n")
                .append("vec4 vk_textureCube(samplerCube s, vec3 c) { return texture(s, c); }\n")
                .append("vec4 vk_textureCubeLod(samplerCube s, vec3 c, float lod) { return textureLod(s, c, lod); }\n")
                .append("vec4 vk_tex(sampler2D s, vec2 c) { return texture(s, c); }\n")
                .append(isVertex
                        ? "vec4 vk_tex(sampler2D s, vec2 c, float bias) { return textureLod(s, c, bias); }\n"
                        : "vec4 vk_tex(sampler2D s, vec2 c, float bias) { return texture(s, c, bias); }\n")
                .append("vec4 vk_tex(sampler2D s, vec3 c) { return texture(s, c.xy); }\n")
                .append(isVertex
                        ? "vec4 vk_tex(sampler2D s, vec3 c, float bias) { return textureLod(s, c.xy, bias); }\n"
                        : "vec4 vk_tex(sampler2D s, vec3 c, float bias) { return texture(s, c.xy, bias); }\n")
                .append("vec4 vk_tex(sampler3D s, vec3 c) { return texture(s, c); }\n")
                .append("vec4 vk_tex(samplerCube s, vec3 c) { return texture(s, c); }\n")
                .append("vec4 vk_tex(sampler2DArray s, vec3 c) { return texture(s, c); }\n")
                .append("vec4 vk_tex(sampler1D s, float c) { return texture(s, c); }\n")
                .append("ivec4 vk_tex(isampler2D s, vec2 c) { return texture(s, c); }\n")
                .append("uvec4 vk_tex(usampler2D s, vec2 c) { return texture(s, c); }\n")
                .append("vec4 vk_textureLod(sampler2D s, vec2 c, float lod) { return textureLod(s, c, lod); }\n")
                .append("vec4 vk_textureLod(sampler2D s, vec3 c, float lod) { return textureLod(s, c.xy, lod); }\n")
                .append("vec4 vk_textureLod(sampler3D s, vec3 c, float lod) { return textureLod(s, c, lod); }\n")
                .append("vec4 vk_textureLod(samplerCube s, vec3 c, float lod) { return textureLod(s, c, lod); }\n")
                .append("vec4 vk_textureLod(sampler2DArray s, vec3 c, float lod) { return textureLod(s, c, lod); }\n")
                .append("vec4 vk_textureLod(sampler1D s, float c, float lod) { return textureLod(s, c, lod); }\n")
                .append("vec4 vk_texelFetch(sampler2D s, ivec2 p, int lod) { return texelFetch(s, p, lod); }\n")
                .append("vec4 vk_texelFetch(sampler2D s, ivec3 p, int lod) { return texelFetch(s, p.xy, lod); }\n")
                .append("ivec4 vk_texelFetch(isampler2D s, ivec2 p, int lod) { return texelFetch(s, p, lod); }\n")
                .append("uvec4 vk_texelFetch(usampler2D s, ivec2 p, int lod) { return texelFetch(s, p, lod); }\n")
                .append("vec4 vk_textureGather(sampler2D s, vec2 p) { return textureGather(s, p, 0); }\n")
                .append("vec4 vk_textureGather(sampler2D s, vec2 p, int comp) { return comp == 1 ? textureGather(s, p, 1) : (comp == 2 ? textureGather(s, p, 2) : (comp == 3 ? textureGather(s, p, 3) : textureGather(s, p, 0))); }\n")
                .append("vec4 vk_textureGather(sampler2D s, vec2 p, float refZ) { return step(vec4(refZ), textureGather(s, p, 0)); }\n")
                .append("vec4 vk_shadow2D(sampler2D s, vec3 c) { return vec4(c.z <= texture(s, c.xy).r ? 1.0 : 0.0); }\n")
                .append(isVertex
                        ? "vec4 vk_shadow2D(sampler2D s, vec3 c, float bias) { return vec4(c.z <= textureLod(s, c.xy, bias).r ? 1.0 : 0.0); }\n"
                        : "vec4 vk_shadow2D(sampler2D s, vec3 c, float bias) { return vec4(c.z <= texture(s, c.xy, bias).r ? 1.0 : 0.0); }\n")
                .append("vec4 vk_shadow2DLod(sampler2D s, vec3 c, float lod) { return vec4(c.z <= textureLod(s, c.xy, lod).r ? 1.0 : 0.0); }\n")
                .append("vec4 vk_shadow2DProj(sampler2D s, vec4 c) { vec3 p = c.xyz / max(abs(c.w), 1e-6); return vec4(p.z <= texture(s, p.xy).r ? 1.0 : 0.0); }\n")
                .append("vec4 vk_shadow2DProjLod(sampler2D s, vec4 c, float lod) { vec3 p = c.xyz / max(abs(c.w), 1e-6); return vec4(p.z <= textureLod(s, p.xy, lod).r ? 1.0 : 0.0); }\n")
                .append("float vk_shadowTex(sampler2D s, vec3 c) { return c.z <= texture(s, c.xy).r ? 1.0 : 0.0; }\n")
                .append("float vk_shadowTexLod(sampler2D s, vec3 c, float lod) { return c.z <= textureLod(s, c.xy, lod).r ? 1.0 : 0.0; }\n")
                .append("float vk_noise1(float x) { return fract(sin(x * 12.9898) * 43758.5453) * 2.0 - 1.0; }\n")
                .append("vec2 vk_noise2(vec2 x) { return fract(sin(vec2(dot(x, vec2(127.1, 311.7)), dot(x, vec2(269.5, 183.3)))) * 43758.5453) * 2.0 - 1.0; }\n")
                .append("vec3 vk_noise3(vec3 x) { return fract(sin(vec3(dot(x, vec3(127.1, 311.7, 74.7)), dot(x, vec3(269.5, 183.3, 246.1)), dot(x, vec3(113.5, 271.9, 124.6)))) * 43758.5453) * 2.0 - 1.0; }\n")
                .append("vec4 vk_noise4(vec4 x) { return vec4(vk_noise3(x.xyz), vk_noise1(x.w)); }\n\n");

        if (!isVertex && (passKind == PassKind.GBUFFERS || passKind == PassKind.SHADOW)) {
            // Greedy-quad UV wrapping + half-texel clamp helper for uBlockAtlas (vk_gtexture)
            out.append("vec2 vk_remapAtlasUV(vec2 uv, out vec2 ddxOut, out vec2 ddyOut) {\n")
                    .append("    if (vk_SpriteMeta.w == 0u) {\n")
                    .append("        ddxOut = dFdx(uv);\n")
                    .append("        ddyOut = dFdy(uv);\n")
                    .append("        return uv;\n")
                    .append("    }\n")
                    .append("    VkAtlasUVEntry e = vk_uvTable[vk_SpriteMeta.x];\n")
                    .append("    vec2 span = e.uvMax - e.uvMin;\n")
                    .append("    if (span.x <= 0.0 || span.y <= 0.0) {\n")
                    .append("        ddxOut = dFdx(uv);\n")
                    .append("        ddyOut = dFdy(uv);\n")
                    .append("        return uv;\n")
                    .append("    }\n")
                    .append("    vec2 quadSize = vec2(float(vk_SpriteMeta.y), float(vk_SpriteMeta.z));\n")
                    .append("    vec2 norm = (uv - e.uvMin) / span;\n")
                    .append("    vec2 tiled = (vk_SpriteMeta.y > 1u || vk_SpriteMeta.z > 1u)\n")
                    .append("        ? fract(norm * quadSize)\n")
                    .append("        : norm;\n")
                    .append("    vec2 atlasSize = vec2(textureSize(vk_gtexture, 0));\n")
                    .append("    vec2 halfTexel = min(0.5 / max(atlasSize, vec2(1.0)), span * 0.49);\n")
                    .append("    vec2 remapped = clamp(mix(e.uvMin, e.uvMax, tiled), e.uvMin + halfTexel, e.uvMax - halfTexel);\n")
                    .append("    ddxOut = dFdx(norm * quadSize) * span;\n")
                    .append("    ddyOut = dFdy(norm * quadSize) * span;\n")
                    .append("    return remapped;\n")
                    .append("}\n")
                    .append("vec4 vk_sampleAtlas(vec2 uv) {\n")
                    .append("    vec2 ddx, ddy;\n")
                    .append("    vec2 r = vk_remapAtlasUV(uv, ddx, ddy);\n")
                    .append("    return textureGrad(vk_gtexture, r, ddx, ddy);\n")
                    .append("}\n")
                    .append("vec4 vk_sampleAtlasLod(vec2 uv, float lod) {\n")
                    .append("    vec2 ddx, ddy;\n")
                    .append("    vec2 r = vk_remapAtlasUV(uv, ddx, ddy);\n")
                    .append("    return textureLod(vk_gtexture, r, lod);\n")
                    .append("}\n\n");
        }
    }

    private static void emitLegacyBuiltins(StringBuilder out, boolean isVertex) {
        if (isVertex) {
            out.append("vec4 vk_Vertex;\n")
                    .append("vec3 vk_Normal;\n")
                    .append("vec4 vk_Color;\n")
                    .append("vec4 vk_MultiTexCoord0;\n")
                    .append("vec4 vk_MultiTexCoord1;\n")
                    .append("vec4 vk_MultiTexCoord2 = vec4(0.0);\n")
                    .append("vec4 vk_MultiTexCoord3 = vec4(0.0);\n")
                    .append("vec4 vk_mcEntity = vec4(0.0);\n\n");

            out.append("layout(location = ").append(ShaderDescriptorBindings.LOC_FRONT_COLOR)
                    .append(") out vec4 vk_FrontColor;\n");
            out.append("layout(location = ").append(ShaderDescriptorBindings.LOC_TEXCOORD_BASE)
                    .append(") out vec4 vk_TexCoord[4];\n");
            out.append("layout(location = ").append(ShaderDescriptorBindings.LOC_SPRITE_META)
                    .append(") flat out uvec4 vk_SpriteMeta;\n\n");
        } else {
            out.append("layout(location = ").append(ShaderDescriptorBindings.LOC_FRONT_COLOR)
                    .append(") in vec4 vk_FrontColor;\n");
            out.append("layout(location = ").append(ShaderDescriptorBindings.LOC_TEXCOORD_BASE)
                    .append(") in vec4 vk_TexCoord[4];\n");
            out.append("layout(location = ").append(ShaderDescriptorBindings.LOC_SPRITE_META)
                    .append(") flat in uvec4 vk_SpriteMeta;\n\n");
        }
    }

    // ------------------------------------------------------------------------
    // Body Token Rewriter with Lexical Scope Tracking
    // ------------------------------------------------------------------------

    private static String rewriteBody(
            ParsedShader parsed,
            boolean isVertex,
            PassKind passKind,
            Set<String> structNames
    ) {
        List<Token> tokens = parsed.bodyTokens;
        int n = tokens.size();
        StringBuilder sb = new StringBuilder(n * 4);

        List<Set<String>> localScopes = new ArrayList<>();
        Set<String> globalShadowedNames = new HashSet<>();
        Set<String> pendingLocalVars = new HashSet<>();
        Set<String> pendingGlobalVars = new HashSet<>();
        Set<Integer> declaredNameTokenIndices = new HashSet<>();
        Set<String> pendingFnParams = new HashSet<>();
        Set<String> prototypeParams = new HashSet<>();
        Set<String> shadowCompareParams = new HashSet<>();
        int prototypeParenEnd = -1;
        int braceDepth = 0;
        boolean inStructDef = false;
        int structBraceDepth = -1;
        boolean seenEqualsInTopStmt = false;

        for (int i = 0; i < n; i++) {
            Token tok = tokens.get(i);
            if (i > prototypeParenEnd) {
                prototypeParams.clear();
                prototypeParenEnd = -1;
            }

            if (tok.kind == TokenKind.NEWLINE) {
                sb.append('\n');
                continue;
            }
            if (tok.kind == TokenKind.PREPROC) {
                sb.append(tok.text);
                continue;
            }

            if (tok.kind == TokenKind.PUNCT) {
                if (tok.text.equals("{")) {
                    if (!pendingLocalVars.isEmpty() && !localScopes.isEmpty()) {
                        localScopes.get(localScopes.size() - 1).addAll(pendingLocalVars);
                        pendingLocalVars.clear();
                    }
                    braceDepth++;
                    Set<String> scope = new HashSet<>(pendingFnParams);
                    pendingFnParams.clear();
                    localScopes.add(scope);
                } else if (tok.text.equals("}")) {
                    if (inStructDef && braceDepth == structBraceDepth) {
                        inStructDef = false;
                        structBraceDepth = -1;
                    }
                    if (!pendingLocalVars.isEmpty() && !localScopes.isEmpty()) {
                        localScopes.get(localScopes.size() - 1).addAll(pendingLocalVars);
                        pendingLocalVars.clear();
                    }
                    if (braceDepth > 0) braceDepth--;
                    if (braceDepth == 0) {
                        seenEqualsInTopStmt = false;
                        shadowCompareParams.clear();
                    }
                    if (!localScopes.isEmpty()) localScopes.remove(localScopes.size() - 1);
                } else if (braceDepth == 0 && tok.text.equals("=")) {
                    seenEqualsInTopStmt = true;
                } else if (tok.text.equals(";")) {
                    if (braceDepth == 0) {
                        seenEqualsInTopStmt = false;
                        if (!pendingGlobalVars.isEmpty()) {
                            globalShadowedNames.addAll(pendingGlobalVars);
                            pendingGlobalVars.clear();
                        }
                    } else if (!pendingLocalVars.isEmpty() && !localScopes.isEmpty()) {
                        localScopes.get(localScopes.size() - 1).addAll(pendingLocalVars);
                        pendingLocalVars.clear();
                    }
                } else if (tok.text.equals("(") && braceDepth == 0 && !seenEqualsInTopStmt) {
                    // Only treat (...) as function parameter list if NOT inside a global '=' initializer
                    Token fnNameTok = prevCodeToken(tokens, i);
                    Token retTypeTok = fnNameTok != null ? prevCodeTokenBefore(tokens, i, fnNameTok) : null;
                    if (fnNameTok != null && fnNameTok.kind == TokenKind.WORD
                            && retTypeTok != null && retTypeTok.kind == TokenKind.WORD) {
                        int rp = findMatchingParen(tokens, i);
                        if (rp > i) {
                            int afterRp = nextCodeToken(tokens, rp + 1);
                            if (afterRp < n && tokens.get(afterRp).text.equals("{")) {
                                pendingFnParams.clear();
                                shadowCompareParams.clear();
                                collectParamNames(tokens.subList(i + 1, rp), pendingFnParams, shadowCompareParams);
                                prototypeParams.clear();
                                prototypeParams.addAll(pendingFnParams);
                                prototypeParenEnd = rp;
                            } else if (afterRp < n && tokens.get(afterRp).text.equals(";")) {
                                prototypeParams.clear();
                                collectParamNames(tokens.subList(i + 1, rp), prototypeParams, null);
                                prototypeParenEnd = rp;
                            }
                        }
                    }
                }
                sb.append(tok.text);
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }

            if (tok.kind == TokenKind.NUMBER) {
                sb.append(tok.text);
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }

            // TokenKind.WORD
            String w = tok.text;

            if (w.equals("struct") && braceDepth == 0) {
                inStructDef = true;
                structBraceDepth = braceDepth + 1;
                sb.append(w);
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }

            // Detect variable declarations to shield local/global names from UBO rewrite after their initializer
            if (!inStructDef && isLocalDeclarationStart(tokens, i, structNames)) {
                if (braceDepth > 0) {
                    collectLocalDeclaredNames(tokens, i, pendingLocalVars, declaredNameTokenIndices);
                } else if (!seenEqualsInTopStmt && !isFunctionDeclAhead(tokens, i)) {
                    collectLocalDeclaredNames(tokens, i, pendingGlobalVars, declaredNameTokenIndices);
                }
            }

            // Rewrite void main(...) -> void vk_user_main(...)
            if (braceDepth == 0 && w.equals("main")) {
                int nextIdx = nextCodeToken(tokens, i + 1);
                if (nextIdx < n && tokens.get(nextIdx).text.equals("(")) {
                    sb.append("vk_user_main");
                    appendSpaceIfNeeded(tokens, i, sb);
                    continue;
                }
            }

            // Never rewrite field names after '.'
            Token prev = prevCodeToken(tokens, i);
            boolean afterDot = prev != null && prev.kind == TokenKind.PUNCT && prev.text.equals(".");
            if (afterDot) {
                sb.append(w);
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }

            Token next = nextCodeTokenObj(tokens, i);
            boolean followedByParen = next != null && next.kind == TokenKind.PUNCT && next.text.equals("(");

            // 0. GLSL 460 reserved keywords & shadow/3D sampler type normalization
            String normalizedSampler = normalizeSamplerType(w);
            if (normalizedSampler != null) {
                sb.append(normalizedSampler);
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }
            if (w.equals("sample")) {
                String nextWord = (next != null && next.kind == TokenKind.WORD) ? next.text : "";
                if (!nextWord.equals("in") && !nextWord.equals("out") && !nextWord.equals("varying")) {
                    sb.append("vk_sample_var");
                    appendSpaceIfNeeded(tokens, i, sb);
                    continue;
                }
            }
            if (w.equals("sampler")) {
                sb.append("vk_sampler_var");
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }
            if (w.equals("input")) {
                sb.append("vk_input_var");
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }
            if (w.equals("output")) {
                sb.append("vk_output_var");
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }
            if (parsed.hasUserRound && w.equals("round") && followedByParen) {
                sb.append("vk_round");
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }
            if (parsed.hasUserFma && w.equals("fma") && followedByParen) {
                sb.append("vk_fma");
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }
            if (w.equals("normalize") && followedByParen && (prev == null || !".".equals(prev.text))) {
                sb.append("_vk_normalize");
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }

            // 1. ftransform()
            if (w.equals("ftransform") && followedByParen) {
                int lp = nextCodeToken(tokens, i + 1);
                int rp = findMatchingParen(tokens, lp);
                if (rp > lp) {
                    sb.append(ftransformExpr(passKind));
                    i = rp;
                    appendSpaceIfNeeded(tokens, i, sb);
                    continue;
                }
            }

            // 2. texture keyword vs texture() function collision
            if (w.equals("texture")) {
                if (followedByParen) {
                    int lp = nextCodeToken(tokens, i + 1);
                    String firstArg = firstArgIdentifier(tokens, lp + 1);
                    if (firstArg != null && (parsed.shadowCompareSamplers.contains(firstArg)
                            || shadowCompareParams.contains(firstArg))) {
                        sb.append("vk_shadowTex");
                    } else if (!isVertex && (passKind == PassKind.GBUFFERS || passKind == PassKind.SHADOW)
                            && isBlockAtlasArg(firstArg) && countCallArgs(tokens, lp) == 2) {
                        int rp = findMatchingParen(tokens, lp);
                        if (rp > lp) {
                            int comma = findFirstTopLevelComma(tokens, lp + 1, rp);
                            sb.append("vk_sampleAtlas(");
                            sb.append(rewriteTokenSlice(parsed, tokens, comma + 1, rp, isVertex, passKind,
                                    localScopes, prototypeParams, globalShadowedNames, shadowCompareParams, inStructDef));
                            sb.append(')');
                            i = rp;
                            appendSpaceIfNeeded(tokens, i, sb);
                            continue;
                        }
                        sb.append("vk_tex");
                    } else {
                        sb.append("vk_tex");
                    }
                } else {
                    sb.append(resolveSamplerAlias("texture", passKind));
                }
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }

            if (w.equals("textureLod") && followedByParen) {
                int lp = nextCodeToken(tokens, i + 1);
                String firstArg = firstArgIdentifier(tokens, lp + 1);
                if (firstArg != null && (parsed.shadowCompareSamplers.contains(firstArg)
                        || shadowCompareParams.contains(firstArg))) {
                    sb.append("vk_shadowTexLod");
                } else {
                    sb.append("vk_textureLod");
                }
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }

            // 3. Legacy sampling builtins
            if (followedByParen) {
                if (!isVertex && (passKind == PassKind.GBUFFERS || passKind == PassKind.SHADOW)
                        && (w.equals("texture2D") || w.equals("texture2DLod"))) {
                    int lp = nextCodeToken(tokens, i + 1);
                    String firstArg = firstArgIdentifier(tokens, lp + 1);
                    int rp = findMatchingParen(tokens, lp);
                    if (isBlockAtlasArg(firstArg) && rp > lp) {
                        int argCount = countCallArgs(tokens, lp);
                        if (w.equals("texture2D") && argCount == 2) {
                            int comma = findFirstTopLevelComma(tokens, lp + 1, rp);
                            sb.append("vk_sampleAtlas(");
                            sb.append(rewriteTokenSlice(parsed, tokens, comma + 1, rp, isVertex, passKind,
                                    localScopes, prototypeParams, globalShadowedNames, shadowCompareParams, inStructDef));
                            sb.append(')');
                            i = rp;
                            appendSpaceIfNeeded(tokens, i, sb);
                            continue;
                        } else if (w.equals("texture2DLod") && argCount == 3) {
                            int comma = findFirstTopLevelComma(tokens, lp + 1, rp);
                            sb.append("vk_sampleAtlasLod(");
                            sb.append(rewriteTokenSlice(parsed, tokens, comma + 1, rp, isVertex, passKind,
                                    localScopes, prototypeParams, globalShadowedNames, shadowCompareParams, inStructDef));
                            sb.append(')');
                            i = rp;
                            appendSpaceIfNeeded(tokens, i, sb);
                            continue;
                        }
                    }
                }
                String mappedFn = mapLegacySampleFunction(w);
                if (mappedFn != null) {
                    sb.append(mappedFn);
                    appendSpaceIfNeeded(tokens, i, sb);
                    continue;
                }
            }

            boolean precededByDeclType = isPrecededByTypeOrQualifier(prev, structNames);

            // 4. Sampler aliases and unknown samplers
            if (!precededByDeclType && !declaredNameTokenIndices.contains(i)
                    && !isShadowed(w, localScopes) && !prototypeParams.contains(w)
                    && !globalShadowedNames.contains(w)) {
                String samplerAlias = resolveSamplerAlias(w, passKind);
                if (samplerAlias != null && !followedByParen) {
                    sb.append(samplerAlias);
                    appendSpaceIfNeeded(tokens, i, sb);
                    continue;
                }
                String unknownSamplerFallback = parsed.unknownSamplers.get(w);
                if (unknownSamplerFallback != null && !followedByParen) {
                    sb.append(unknownSamplerFallback);
                    appendSpaceIfNeeded(tokens, i, sb);
                    continue;
                }
            }

            // 5. Legacy gl_* built-ins
            String glReplacement = mapLegacyBuiltin(w, isVertex, passKind);
            if (glReplacement != null) {
                sb.append(glReplacement);
                appendSpaceIfNeeded(tokens, i, sb);
                continue;
            }

            // 6. ShaderPackUBO fields and derived uniforms (only when not shadowed by local/param/global)
            if (!inStructDef && !followedByParen && !precededByDeclType && !declaredNameTokenIndices.contains(i)
                    && !isShadowed(w, localScopes) && !prototypeParams.contains(w)
                    && !globalShadowedNames.contains(w)) {
                if (ShaderUniformLayout.UBO_FIELD_NAMES.contains(w)) {
                    sb.append("vk_ubo.").append(w);
                    appendSpaceIfNeeded(tokens, i, sb);
                    continue;
                }
                if (parsed.resolvedCustomUniformNames.contains(w)) {
                    sb.append("_vk_cu_").append(w).append("()");
                    appendSpaceIfNeeded(tokens, i, sb);
                    continue;
                }
                if (parsed.declaredDerivedUniforms.contains(w)) {
                    String derived = resolveDerivedUniformForToken(w, passKind, prev, nextCodeTokenObj(tokens, i));
                    if (derived != null) {
                        sb.append(derived);
                        appendSpaceIfNeeded(tokens, i, sb);
                        continue;
                    }
                }
            }

            sb.append(w);
            appendSpaceIfNeeded(tokens, i, sb);
        }

        return sb.toString();
    }

    private static String mapLegacySampleFunction(String fn) {
        return switch (fn) {
            case "texture2D" -> "vk_texture2D";
            case "texture2DLod" -> "vk_texture2DLod";
            case "texture2DGrad", "texture2DGradARB" -> "vk_texture2DGrad";
            case "texture2DProj" -> "vk_texture2DProj";
            case "texture2DProjLod" -> "vk_texture2DProjLod";
            case "texture3D" -> "vk_texture3D";
            case "texture3DLod" -> "vk_texture3DLod";
            case "textureCube" -> "vk_textureCube";
            case "textureCubeLod" -> "vk_textureCubeLod";
            case "shadow2D" -> "vk_shadow2D";
            case "shadow2DLod" -> "vk_shadow2DLod";
            case "shadow2DProj" -> "vk_shadow2DProj";
            case "shadow2DProjLod" -> "vk_shadow2DProjLod";
            case "texelFetch" -> "vk_texelFetch";
            case "textureGather" -> "vk_textureGather";
            case "noise1" -> "vk_noise1";
            case "noise2" -> "vk_noise2";
            case "noise3" -> "vk_noise3";
            case "noise4" -> "vk_noise4";
            default -> null;
        };
    }

    private static String mapLegacyBuiltin(String name, boolean isVertex, PassKind passKind) {
        return switch (name) {
            case "gl_Vertex" -> "vk_Vertex";
            case "gl_Normal" -> "vk_Normal";
            case "gl_Color" -> isVertex ? "vk_Color" : "vk_FrontColor";
            case "gl_SecondaryColor" -> "vec4(0.0)";
            case "gl_MultiTexCoord0" -> "vk_MultiTexCoord0";
            case "gl_MultiTexCoord1" -> "vk_MultiTexCoord1";
            case "gl_MultiTexCoord2" -> "vk_MultiTexCoord2";
            case "gl_MultiTexCoord3" -> "vk_MultiTexCoord3";
            case "gl_MultiTexCoord4", "gl_MultiTexCoord5", "gl_MultiTexCoord6", "gl_MultiTexCoord7" -> "vec4(0.0)";
            case "gl_FrontColor", "gl_BackColor" -> "vk_FrontColor";
            case "gl_TexCoord" -> "vk_TexCoord";
            case "gl_FragData" -> "vk_FragData";
            case "gl_FragColor" -> "vk_FragData[0]";
            case "gl_ModelViewMatrix" -> switch (passKind) {
                case SHADOW -> "vk_ubo.shadowModelView";
                case GBUFFERS -> "vk_ubo.gbufferModelView";
                default -> "mat4(1.0)";
            };
            case "gl_ModelViewMatrixInverse" -> switch (passKind) {
                case SHADOW -> "vk_ubo.shadowModelViewInverse";
                case GBUFFERS -> "vk_ubo.gbufferModelViewInverse";
                default -> "mat4(1.0)";
            };
            case "gl_ProjectionMatrix" -> switch (passKind) {
                case SHADOW -> "vk_ubo.shadowProjection";
                case GBUFFERS -> "vk_ubo.gbufferProjection";
                default -> "VK_FULLSCREEN_MVP";
            };
            case "gl_ProjectionMatrixInverse" -> switch (passKind) {
                case SHADOW -> "vk_ubo.shadowProjectionInverse";
                case GBUFFERS -> "vk_ubo.gbufferProjectionInverse";
                default -> "VK_FULLSCREEN_MVP_INV";
            };
            case "gl_ModelViewProjectionMatrix" -> switch (passKind) {
                case SHADOW -> "(vk_ubo.shadowProjection * vk_ubo.shadowModelView)";
                case GBUFFERS -> "(vk_ubo.gbufferProjection * vk_ubo.gbufferModelView)";
                default -> "VK_FULLSCREEN_MVP";
            };
            case "gl_ModelViewProjectionMatrixInverse" -> switch (passKind) {
                case SHADOW -> "(vk_ubo.shadowModelViewInverse * vk_ubo.shadowProjectionInverse)";
                case GBUFFERS -> "(vk_ubo.gbufferModelViewInverse * vk_ubo.gbufferProjectionInverse)";
                default -> "VK_FULLSCREEN_MVP_INV";
            };
            case "gl_NormalMatrix" -> "vk_NormalMatrix";
            case "gl_TextureMatrix" -> "vk_TextureMatrix";
            case "gl_Fog" -> "vk_Fog";
            case "gl_FogFragCoord" -> "vk_FogFragCoord";
            case "gl_MaxDrawBuffers" -> "8";
            case "gl_VertexID" -> "gl_VertexIndex";
            case "gl_InstanceID" -> "gl_InstanceIndex";
            default -> null;
        };
    }

    private static String ftransformExpr(PassKind passKind) {
        return switch (passKind) {
            case SHADOW -> "(vk_ubo.shadowProjection * vk_ubo.shadowModelView * vk_Vertex)";
            case GBUFFERS -> "(vk_ubo.gbufferProjection * vk_ubo.gbufferModelView * vk_Vertex)";
            default -> "(VK_FULLSCREEN_MVP * vk_Vertex)";
        };
    }

    // ------------------------------------------------------------------------
    // Wrapper main() Emitters
    // ------------------------------------------------------------------------

    private static void emitVertexMainWrapper(
            StringBuilder out,
            PassKind passKind,
            Map<String, VarDecl> attributes,
            List<VarDecl> extraVertStubs
    ) {
        out.append("void main() {\n");
        emitFogAndNormalInit(out, passKind);

        for (int i = 0; i < 4; i++) {
            out.append("    vk_TexCoord[").append(i).append("] = vec4(0.0);\n");
        }

        if (passKind == PassKind.GBUFFERS || passKind == PassKind.SHADOW) {
            out.append("    uvec2 _vp_packed = vk_vertices[gl_VertexIndex];\n")
                    .append("    uint _vp_low = _vp_packed.x;\n")
                    .append("    uint _vp_high = _vp_packed.y;\n")
                    .append("    uint _vp_px = _vp_low & 0x3Fu;\n")
                    .append("    uint _vp_py = (_vp_low >> 6) & 0x3Fu;\n")
                    .append("    uint _vp_pz = (_vp_low >> 12) & 0x3Fu;\n")
                    .append("    uint _vp_normEnum = (_vp_low >> 18) & 0x7u;\n")
                    .append("    uint _vp_ao = (_vp_low >> 21) & 0x3u;\n")
                    .append("    uint _vp_bl = (_vp_low >> 23) & 0xFu;\n")
                    .append("    uint _vp_sl = (_vp_low >> 27) & 0xFu;\n")
                    .append("    uint _vp_wind = (_vp_low >> 31) & 0x1u;\n")
                    .append("    uint _vp_spriteId = _vp_high & 0xFFFFu;\n")
                    .append("    uint _vp_quadW = (_vp_high >> 16) & 0x1Fu;\n")
                    .append("    uint _vp_quadH = (_vp_high >> 21) & 0x1Fu;\n")
                    .append("    uint _vp_corner = (_vp_high >> 26) & 0x3u;\n")
                    .append("    uint _vp_matMask = (_vp_high >> 28) & 0xFu;\n")
                    .append("    uint _vp_chunkIdx = vk_instanceToChunkIdx[gl_InstanceIndex];\n")
                    .append("    VkChunkDrawMeta _vp_meta = vk_chunks[_vp_chunkIdx];\n")
                    .append("    float _vp_posScale = ((_vp_meta.flags & 4u) != 0u) ? 0.5 : 1.0;\n")
                    .append("    vec3 _vp_worldPos = vec3(_vp_meta.chunkPos) * 16.0 + vec3(float(_vp_px), float(_vp_py), float(_vp_pz)) * _vp_posScale;\n")
                    .append("    vk_Vertex = vec4(_vp_worldPos - vk_ubo.cameraPosition, 1.0);\n")
                    .append("    vk_Normal = (_vp_normEnum < 6u) ? VK_FACE_NORMALS[_vp_normEnum] : vec3(0.0, 1.0, 0.0);\n")
                    .append("    VkAtlasUVEntry _vp_atlas = vk_uvTable[_vp_spriteId];\n")
                    .append("    vec2 _vp_cornerUV = VK_CORNER_UVS[_vp_corner];\n")
                    .append("    vec2 _vp_uv = mix(_vp_atlas.uvMin, _vp_atlas.uvMax, _vp_cornerUV);\n")
                    .append("    vk_MultiTexCoord0 = vec4(_vp_uv, 0.0, 1.0);\n")
                    .append("    vk_MultiTexCoord1 = vec4(float(_vp_bl) * 16.0, float(_vp_sl) * 16.0, 0.0, 1.0);\n")
                    .append("    float _vp_dirShade = (_vp_normEnum == 2u) ? 1.0 : ((_vp_normEnum == 3u) ? 0.5 : ((_vp_normEnum <= 1u) ? 0.6 : 0.8));\n")
                    .append("    float _vp_aoShade = 0.55 + 0.15 * float(_vp_ao);\n")
                    .append("    vk_Color = vec4(_vp_atlas.tint.rgb * (_vp_dirShade * _vp_aoShade), 1.0);\n")
                    .append("    vk_FrontColor = vk_Color;\n")
                    .append("    vk_SpriteMeta = uvec4(_vp_spriteId, max(_vp_quadW, 1u), max(_vp_quadH, 1u), 1u);\n")
                    .append("    vk_mcEntity = vec4(float(_vp_matMask), (_vp_wind == 1u) ? 1.0 : 0.0, 0.0, 0.0);\n");
        } else {
            out.append("    vec2 _vp_fsPos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);\n")
                    .append("    vk_Vertex = vec4(_vp_fsPos, 0.0, 1.0);\n")
                    .append("    vk_Normal = vec3(0.0, 0.0, 1.0);\n")
                    .append("    vk_Color = vec4(1.0);\n")
                    .append("    vk_FrontColor = vec4(1.0);\n")
                    .append("    vk_MultiTexCoord0 = vec4(_vp_fsPos, 0.0, 1.0);\n")
                    .append("    vk_MultiTexCoord1 = vec4(0.0, 0.0, 0.0, 1.0);\n")
                    .append("    vk_SpriteMeta = uvec4(0u);\n");
        }

        for (VarDecl attr : attributes.values()) {
            out.append("    ").append(attr.name).append(" = ").append(attributeInitExpr(attr)).append(";\n");
        }
        for (VarDecl stub : extraVertStubs) {
            out.append("    ").append(stub.name).append(" = ")
                    .append(zeroInitExpr(stub.type, stub.arraySuffix)).append(";\n");
        }

        out.append("    vk_user_main();\n");
        if (passKind == PassKind.SHADOW) {
            out.append("    gl_Position.z = gl_Position.z * 0.5 + 0.5 * gl_Position.w;\n");
        }
        out.append("}\n");
    }

    private static void emitFragmentMainWrapper(
            StringBuilder out,
            boolean useFragDataWrapper,
            int attachmentCount
    ) {
        out.append("void main() {\n");
        emitFogAndNormalInit(out, PassKind.GBUFFERS);
        if (useFragDataWrapper) {
            out.append("    for (int _vk_i = 0; _vk_i < 8; ++_vk_i) {\n")
                    .append("        vk_FragData[_vk_i] = vec4(0.0);\n")
                    .append("    }\n")
                    .append("    vk_user_main();\n");
            for (int i = 0; i < attachmentCount; i++) {
                out.append("    vk_FragOut").append(i).append(" = vk_FragData[").append(i).append("];\n");
            }
        } else {
            out.append("    vk_user_main();\n");
        }
        out.append("}\n");
    }

    private static void emitFogAndNormalInit(StringBuilder out, PassKind passKind) {
        out.append("    vk_Fog.color = vec4(vk_ubo.fogColor, 1.0);\n")
                .append("    vk_Fog.density = vk_ubo.fogDensity;\n")
                .append("    vk_Fog.start = vk_ubo.fogStart;\n")
                .append("    vk_Fog.end = vk_ubo.fogEnd;\n")
                .append("    vk_Fog.scale = 1.0 / max(vk_ubo.fogEnd - vk_ubo.fogStart, 0.0001);\n");
        switch (passKind) {
            case SHADOW -> out.append("    vk_NormalMatrix = transpose(inverse(mat3(vk_ubo.shadowModelView)));\n");
            case GBUFFERS -> out.append("    vk_NormalMatrix = transpose(inverse(mat3(vk_ubo.gbufferModelView)));\n");
            default -> out.append("    vk_NormalMatrix = mat3(1.0);\n");
        }
    }

    private record ResolvedCustomUniform(String name, String type, String glslExpr) {}

    private static void promoteCustomUniformsOverDerived(
            ParsedShader parsed,
            Map<String, ShaderPackProperties.CustomUniform> customUniforms
    ) {
        if (customUniforms == null || customUniforms.isEmpty()) {
            return;
        }
        for (String name : customUniforms.keySet()) {
            if (parsed.declaredDerivedUniforms.remove(name)) {
                VarDecl decl = parsed.declaredDerivedUniformDecls.get(name);
                if (decl != null) {
                    parsed.unknownUniforms.putIfAbsent(name, decl);
                }
            }
        }
    }

    private static List<ResolvedCustomUniform> resolveCustomUniforms(
            ParsedShader parsed,
            Map<String, ShaderPackProperties.CustomUniform> customUniforms,
            PassKind passKind
    ) {
        if (customUniforms == null || customUniforms.isEmpty() || parsed.unknownUniforms.isEmpty()) {
            return Collections.emptyList();
        }
        Set<String> needed = new LinkedHashSet<>();
        Set<String> visiting = new HashSet<>();
        for (VarDecl u : parsed.unknownUniforms.values()) {
            if (customUniforms.containsKey(u.name)) {
                collectCustomUniformDeps(u.name, customUniforms, needed, visiting);
            }
        }
        if (needed.isEmpty()) {
            return Collections.emptyList();
        }
        Set<String> activeNames = new HashSet<>();
        List<ResolvedCustomUniform> result = new ArrayList<>();
        for (String name : needed) {
            ShaderPackProperties.CustomUniform def = customUniforms.get(name);
            if (def == null) {
                continue;
            }
            VarDecl unknownDecl = parsed.unknownUniforms.get(def.name());
            // Skip internal variable.* if its name collides with an existing shader symbol
            if (unknownDecl == null && (parsed.declaredDerivedUniforms.contains(def.name())
                    || ShaderUniformLayout.UBO_FIELD_NAMES.contains(def.name())
                    || parsed.attributes.containsKey(def.name())
                    || parsed.varyings.containsKey(def.name())
                    || parsed.structNames.contains(def.name()))) {
                continue;
            }
            String targetType = unknownDecl != null ? unknownDecl.type : def.type();
            if (unknownDecl != null && !unknownDecl.arraySuffix.isEmpty()) {
                continue;
            }
            String translated = translateCustomUniformExpr(def.expression(), passKind, activeNames);
            if (translated == null || translated.isBlank()) {
                continue;
            }
            if ("int".equals(targetType)) {
                translated = "int(" + translated + ")";
            } else if ("float".equals(targetType) && "int".equals(def.type())) {
                translated = "float(" + translated + ")";
            } else if ("bool".equals(targetType)) {
                translated = "bool(" + translated + ")";
            }
            activeNames.add(def.name());
            result.add(new ResolvedCustomUniform(def.name(), targetType, translated));
        }
        return result;
    }

    private static void collectCustomUniformDeps(
            String name,
            Map<String, ShaderPackProperties.CustomUniform> customUniforms,
            Set<String> needed,
            Set<String> visiting
    ) {
        if (needed.contains(name) || !visiting.add(name)) {
            return;
        }
        ShaderPackProperties.CustomUniform def = customUniforms.get(name);
        if (def != null) {
            List<Token> tokens = tokenize(def.expression());
            for (Token t : tokens) {
                if (t.kind == TokenKind.WORD && customUniforms.containsKey(t.text) && !t.text.equals(name)) {
                    collectCustomUniformDeps(t.text, customUniforms, needed, visiting);
                }
            }
            needed.add(name);
        }
        visiting.remove(name);
    }

    private static final Set<String> CUSTOM_EXPR_GLSL_BUILTINS = Set.of(
            "float", "int", "bool", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "mat3", "mat4",
            "abs", "sign", "floor", "ceil", "fract", "mod", "min", "max", "clamp", "mix", "step", "smoothstep",
            "sin", "cos", "tan", "asin", "acos", "atan", "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
            "length", "distance", "dot", "cross", "normalize", "radians", "degrees", "round", "trunc",
            "true", "false"
    );

    private static String translateCustomUniformExpr(
            String expr,
            PassKind passKind,
            Set<String> activeCustomNames
    ) {
        List<Token> tokens = tokenize(expr);
        return translateCustomTokens(tokens, 0, tokens.size(), passKind, activeCustomNames);
    }

    private static String translateCustomTokens(
            List<Token> tokens,
            int start,
            int end,
            PassKind passKind,
            Set<String> activeCustomNames
    ) {
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < end; i++) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.NEWLINE || t.kind == TokenKind.PREPROC) {
                continue;
            }
            if (t.kind == TokenKind.WORD) {
                // Check function calls: if(...), fmod(...), frac(...), smooth(...), random(...), in(...), between(...), equals(...), torad(...), todeg(...), lerp(...)
                int nextIdx = nextCodeToken(tokens, i + 1);
                if (nextIdx >= 0 && nextIdx < end && "(".equals(tokens.get(nextIdx).text)) {
                    int closeIdx = findMatchingParen(tokens, nextIdx, end);
                    if (closeIdx > nextIdx) {
                        List<int[]> args = splitCallArgs(tokens, nextIdx + 1, closeIdx);
                        String fn = t.text;
                        if ("if".equals(fn) && args.size() == 3) {
                            String c = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            String a = translateCustomTokens(tokens, args.get(1)[0], args.get(1)[1], passKind, activeCustomNames);
                            String b = translateCustomTokens(tokens, args.get(2)[0], args.get(2)[1], passKind, activeCustomNames);
                            if (c == null || a == null || b == null) return null;
                            sb.append("(bool(").append(c).append(") ? (").append(a).append(") : (").append(b).append("))");
                            i = closeIdx;
                            continue;
                        } else if ("fmod".equals(fn) && args.size() == 2) {
                            String a = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            String b = translateCustomTokens(tokens, args.get(1)[0], args.get(1)[1], passKind, activeCustomNames);
                            if (a == null || b == null) return null;
                            sb.append("mod(float(").append(a).append("), float(").append(b).append("))");
                            i = closeIdx;
                            continue;
                        } else if ("frac".equals(fn) && args.size() == 1) {
                            String a = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            if (a == null) return null;
                            sb.append("fract(float(").append(a).append("))");
                            i = closeIdx;
                            continue;
                        } else if ("smooth".equals(fn) && (args.size() == 2 || args.size() == 4)) {
                            int valIdx = args.size() == 4 ? 1 : 0;
                            String v = translateCustomTokens(tokens, args.get(valIdx)[0], args.get(valIdx)[1], passKind, activeCustomNames);
                            if (v == null) return null;
                            sb.append("(").append(v).append(")");
                            i = closeIdx;
                            continue;
                        } else if ("random".equals(fn)) {
                            sb.append("fract(sin(float(vk_ubo.frameCounter) * 12.9898) * 43758.5453)");
                            i = closeIdx;
                            continue;
                        } else if ("in".equals(fn) && args.size() >= 2) {
                            String x = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            if (x == null) return null;
                            sb.append("(");
                            for (int ai = 1; ai < args.size(); ai++) {
                                if (ai > 1) sb.append(" || ");
                                String v = translateCustomTokens(tokens, args.get(ai)[0], args.get(ai)[1], passKind, activeCustomNames);
                                if (v == null) return null;
                                sb.append("((").append(x).append(") == (").append(v).append("))");
                            }
                            sb.append(")");
                            i = closeIdx;
                            continue;
                        } else if ("between".equals(fn) && args.size() == 3) {
                            String x = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            String lo = translateCustomTokens(tokens, args.get(1)[0], args.get(1)[1], passKind, activeCustomNames);
                            String hi = translateCustomTokens(tokens, args.get(2)[0], args.get(2)[1], passKind, activeCustomNames);
                            if (x == null || lo == null || hi == null) return null;
                            sb.append("((").append(x).append(") >= (").append(lo).append(") && (").append(x).append(") <= (").append(hi).append("))");
                            i = closeIdx;
                            continue;
                        } else if ("equals".equals(fn) && args.size() == 3) {
                            String a = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            String b = translateCustomTokens(tokens, args.get(1)[0], args.get(1)[1], passKind, activeCustomNames);
                            String eps = translateCustomTokens(tokens, args.get(2)[0], args.get(2)[1], passKind, activeCustomNames);
                            if (a == null || b == null || eps == null) return null;
                            sb.append("(abs((").append(a).append(") - (").append(b).append(")) <= (").append(eps).append("))");
                            i = closeIdx;
                            continue;
                        } else if ("torad".equals(fn) && args.size() == 1) {
                            String a = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            if (a == null) return null;
                            sb.append("radians(float(").append(a).append("))");
                            i = closeIdx;
                            continue;
                        } else if ("todeg".equals(fn) && args.size() == 1) {
                            String a = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            if (a == null) return null;
                            sb.append("degrees(float(").append(a).append("))");
                            i = closeIdx;
                            continue;
                        } else if ("lerp".equals(fn) && args.size() == 3) {
                            String a = translateCustomTokens(tokens, args.get(0)[0], args.get(0)[1], passKind, activeCustomNames);
                            String b = translateCustomTokens(tokens, args.get(1)[0], args.get(1)[1], passKind, activeCustomNames);
                            String tArg = translateCustomTokens(tokens, args.get(2)[0], args.get(2)[1], passKind, activeCustomNames);
                            if (a == null || b == null || tArg == null) return null;
                            sb.append("mix(").append(a).append(", ").append(b).append(", ").append(tArg).append(")");
                            i = closeIdx;
                            continue;
                        }
                    }
                }

                // Check member access after dot (like .x, .y, .z, .w or matrix .1.1)
                Token prev = prevCodeToken(tokens, i);
                if (prev != null && ".".equals(prev.text)) {
                    sb.append(t.text);
                    continue;
                }

                // Resolve identifier
                String w = t.text;
                String mapped = null;
                if (activeCustomNames.contains(w)) {
                    mapped = "_vk_cu_" + w + "()";
                } else if (CUSTOM_EXPR_GLSL_BUILTINS.contains(w)) {
                    mapped = w;
                } else if (ShaderUniformLayout.UBO_FIELD_NAMES.contains(w)) {
                    mapped = "vk_ubo." + w;
                } else {
                    String derived = resolveDerivedUniform(w, passKind);
                    if (derived != null) {
                        mapped = derived;
                    } else {
                        mapped = switch (w) {
                            case "pi" -> "3.14159265359";
                            case "dhRenderDistance" -> "vk_ubo.far";
                            case "biome", "biome_precipitation", "biome_category", "worldDay" -> "0";
                            case "eyeAltitude" -> "vk_ubo.cameraPosition.y";
                            case "temperature", "rainfall" -> "0.5";
                            case "lightningBoltPosition" -> "vec4(0.0)";
                            default -> null;
                        };
                    }
                }
                if (mapped == null) {
                    return null;
                }
                sb.append(mapped);

                // Handle OptiFine matrix .col.row indexing e.g. gbufferProjection.1.1
                int dot1 = nextCodeToken(tokens, i + 1);
                if (dot1 >= 0 && dot1 < end && tokens.get(dot1).kind == TokenKind.NUMBER
                        && tokens.get(dot1).text.startsWith(".")) {
                    int dot2 = nextCodeToken(tokens, dot1 + 1);
                    if (dot2 >= 0 && dot2 < end && tokens.get(dot2).kind == TokenKind.NUMBER
                            && tokens.get(dot2).text.startsWith(".")) {
                        sb.append('[').append(tokens.get(dot1).text.substring(1)).append("][")
                                .append(tokens.get(dot2).text.substring(1)).append(']');
                        i = dot2;
                        continue;
                    }
                } else if (dot1 >= 0 && dot1 < end && ".".equals(tokens.get(dot1).text)) {
                    int colIdx = nextCodeToken(tokens, dot1 + 1);
                    if (colIdx >= 0 && colIdx < end && tokens.get(colIdx).kind == TokenKind.NUMBER) {
                        int dot2 = nextCodeToken(tokens, colIdx + 1);
                        if (dot2 >= 0 && dot2 < end && ".".equals(tokens.get(dot2).text)) {
                            int rowIdx = nextCodeToken(tokens, dot2 + 1);
                            if (rowIdx >= 0 && rowIdx < end && tokens.get(rowIdx).kind == TokenKind.NUMBER) {
                                sb.append('[').append(tokens.get(colIdx).text).append("][")
                                        .append(tokens.get(rowIdx).text).append(']');
                                i = rowIdx;
                                continue;
                            }
                        }
                    }
                }
            } else {
                sb.append(t.text);
            }
        }
        return sb.toString();
    }

    private static int findMatchingParen(List<Token> tokens, int openIdx, int end) {
        int depth = 0;
        for (int i = openIdx; i < end; i++) {
            String s = tokens.get(i).text;
            if ("(".equals(s)) depth++;
            else if (")".equals(s)) {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

    private static List<int[]> splitCallArgs(List<Token> tokens, int start, int end) {
        List<int[]> args = new ArrayList<>();
        int depth = 0;
        int curStart = start;
        for (int i = start; i < end; i++) {
            String s = tokens.get(i).text;
            if ("(".equals(s) || "[".equals(s)) depth++;
            else if (")".equals(s) || "]".equals(s)) depth--;
            else if (",".equals(s) && depth == 0) {
                args.add(new int[]{curStart, i});
                curStart = i + 1;
            }
        }
        if (curStart < end) {
            boolean hasCode = false;
            for (int i = curStart; i < end; i++) {
                if (tokens.get(i).kind != TokenKind.NEWLINE && tokens.get(i).kind != TokenKind.PREPROC) {
                    hasCode = true;
                    break;
                }
            }
            if (hasCode) {
                args.add(new int[]{curStart, end});
            }
        }
        return args;
    }

    private static String attributeInitExpr(VarDecl attr) {
        if (!attr.arraySuffix.isEmpty()) {
            return zeroInitExpr(attr.type, attr.arraySuffix);
        }
        if (attr.name.equals("mc_Entity")) {
            return switch (attr.type) {
                case "vec4" -> "vk_mcEntity";
                case "vec3" -> "vk_mcEntity.xyz";
                case "vec2" -> "vk_mcEntity.xy";
                case "float" -> "vk_mcEntity.x";
                case "int" -> "int(vk_mcEntity.x)";
                case "ivec4" -> "ivec4(vk_mcEntity)";
                default -> zeroInitExpr(attr.type, attr.arraySuffix);
            };
        }
        if (attr.name.equals("mc_midTexCoord")) {
            return switch (attr.type) {
                case "vec2" -> "vk_MultiTexCoord0.xy";
                case "vec4" -> "vec4(vk_MultiTexCoord0.xy, 0.0, 1.0)";
                default -> zeroInitExpr(attr.type, attr.arraySuffix);
            };
        }
        if (attr.name.equals("at_tangent")) {
            return switch (attr.type) {
                case "vec4" -> "(abs(vk_Normal.x) > 0.9 ? vec4(0.0, 0.0, 1.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0))";
                case "vec3" -> "(abs(vk_Normal.x) > 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0))";
                default -> zeroInitExpr(attr.type, attr.arraySuffix);
            };
        }
        return zeroInitExpr(attr.type, attr.arraySuffix);
    }

    private static String zeroInitExpr(String type, String arraySuffix) {
        if (arraySuffix == null || arraySuffix.isEmpty()) {
            return type + "(0)";
        }
        int l = arraySuffix.indexOf('[');
        int r = arraySuffix.indexOf(']');
        if (l >= 0 && r > l + 1) {
            try {
                int len = Integer.parseInt(arraySuffix.substring(l + 1, r).trim());
                if (len > 0 && len <= 64) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(type).append('[').append(len).append("](");
                    for (int i = 0; i < len; i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(type).append("(0)");
                    }
                    sb.append(')');
                    return sb.toString();
                }
            } catch (NumberFormatException ignored) {
                // Fall through
            }
        }
        return type + "(0)";
    }

    // ------------------------------------------------------------------------
    // Helpers for Scope / Token Navigation
    // ------------------------------------------------------------------------

    private static boolean isShadowed(String name, List<Set<String>> localScopes) {
        for (int i = localScopes.size() - 1; i >= 0; i--) {
            if (localScopes.get(i).contains(name)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isLocalDeclarationStart(List<Token> tokens, int idx, Set<String> structNames) {
        Token t = tokens.get(idx);
        if (t.kind != TokenKind.WORD) return false;
        String w = t.text;
        if (w.equals("const") || PRECISION_QUALIFIERS.contains(w)) {
            return true;
        }
        if (!KNOWN_GLSL_TYPES.contains(w) && !structNames.contains(w)) {
            return false;
        }
        Token prev = prevCodeToken(tokens, idx);
        if (prev != null && prev.kind == TokenKind.PUNCT) {
            String p = prev.text;
            if (!p.equals(";") && !p.equals("{") && !p.equals("}") && !p.equals("(") && !p.equals(":")) {
                // Avoid misclassifying expressions like "a = vec3(1.0)" or "return float(x)"
                return false;
            }
        } else if (prev != null && prev.kind == TokenKind.WORD) {
            if (!prev.text.equals("const") && !PRECISION_QUALIFIERS.contains(prev.text) && !prev.text.equals("else")) {
                return false;
            }
        }
        // Must be followed by another WORD (variable name), NOT '(' (constructor call like vec3(...))
        Token next = nextCodeTokenObj(tokens, idx);
        return next != null && next.kind == TokenKind.WORD && !GLSL_KEYWORDS.contains(next.text);
    }

    private static void collectLocalDeclaredNames(
            List<Token> tokens,
            int idx,
            Set<String> outNames,
            Set<Integer> outTokenIndices
    ) {
        int i = idx;
        int n = tokens.size();
        // Skip leading const/precision/type
        while (i < n) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.WORD && (t.text.equals("const") || PRECISION_QUALIFIERS.contains(t.text))) {
                i++;
                continue;
            }
            break;
        }
        if (i < n && tokens.get(i).kind == TokenKind.WORD) {
            i++; // skip type
        }
        int parenDepth = 0;
        boolean expectName = true;
        while (i < n) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.PUNCT) {
                if (t.text.equals("(") || t.text.equals("[") || t.text.equals("{")) {
                    parenDepth++;
                } else if ((t.text.equals(")") || t.text.equals("]") || t.text.equals("}")) && parenDepth > 0) {
                    parenDepth--;
                } else if (parenDepth == 0 && (t.text.equals(";") || t.text.equals(")"))) {
                    break;
                } else if (parenDepth == 0 && t.text.equals(",")) {
                    expectName = true;
                } else if (parenDepth == 0 && t.text.equals("=")) {
                    expectName = false;
                }
            } else if (t.kind == TokenKind.WORD && parenDepth == 0 && expectName) {
                outNames.add(t.text);
                if (outTokenIndices != null) {
                    outTokenIndices.add(i);
                }
                expectName = false;
            }
            i++;
        }
    }

    private static void collectParamNames(
            List<Token> params,
            Set<String> outNames,
            Set<String> shadowCompareParams
    ) {
        int i = 0;
        int n = params.size();
        List<Token> currentParam = new ArrayList<>();
        int bracketDepth = 0;
        while (i <= n) {
            Token t = (i < n) ? params.get(i) : null;
            if (t == null || (t.kind == TokenKind.PUNCT && t.text.equals(",") && bracketDepth == 0)) {
                boolean isShadowSampler = false;
                for (Token ct : currentParam) {
                    if (ct.kind == TokenKind.WORD && ct.text.contains("Shadow")) {
                        isShadowSampler = true;
                        break;
                    }
                }
                // Parameter name is the last WORD before any '[' in currentParam
                for (int k = currentParam.size() - 1; k >= 0; k--) {
                    Token ct = currentParam.get(k);
                    if (ct.kind == TokenKind.PUNCT && ct.text.equals("]")) {
                        while (k >= 0 && !currentParam.get(k).text.equals("[")) k--;
                        continue;
                    }
                    if (ct.kind == TokenKind.WORD && !KNOWN_GLSL_TYPES.contains(ct.text)
                            && !SAMPLER_TYPES.contains(ct.text)
                            && !GLSL_KEYWORDS.contains(ct.text)) {
                        outNames.add(ct.text);
                        if (isShadowSampler && shadowCompareParams != null) {
                            shadowCompareParams.add(ct.text);
                        }
                        break;
                    }
                }
                currentParam.clear();
            } else if (t.kind != TokenKind.NEWLINE && t.kind != TokenKind.PREPROC) {
                if (t.text.equals("[")) bracketDepth++;
                else if (t.text.equals("]") && bracketDepth > 0) bracketDepth--;
                currentParam.add(t);
            }
            i++;
        }
    }

    private static boolean isPrecededByTypeOrQualifier(Token prev, Set<String> structNames) {
        if (prev == null || prev.kind != TokenKind.WORD) {
            return false;
        }
        String pw = prev.text;
        return KNOWN_GLSL_TYPES.contains(pw)
                || SAMPLER_TYPES.contains(pw)
                || structNames.contains(pw)
                || pw.equals("const")
                || pw.equals("in")
                || pw.equals("out")
                || pw.equals("inout")
                || PRECISION_QUALIFIERS.contains(pw)
                || INTERP_QUALIFIERS.contains(pw);
    }

    private static boolean isFunctionDeclAhead(List<Token> tokens, int declStartIdx) {
        int i = declStartIdx;
        int n = tokens.size();
        while (i < n) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.WORD && (t.text.equals("const") || PRECISION_QUALIFIERS.contains(t.text))) {
                i++;
                continue;
            }
            break;
        }
        if (i < n && tokens.get(i).kind == TokenKind.WORD) {
            i++; // type
        }
        int nameIdx = nextCodeToken(tokens, i);
        if (nameIdx < n) {
            Token afterName = nextCodeTokenObj(tokens, nameIdx);
            return afterName != null && afterName.kind == TokenKind.PUNCT && afterName.text.equals("(");
        }
        return false;
    }

    private static Token prevCodeTokenBefore(List<Token> tokens, int openParenIdx, Token targetToken) {
        boolean foundTarget = false;
        for (int i = openParenIdx - 1; i >= 0; i--) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.NEWLINE || t.kind == TokenKind.PREPROC) continue;
            if (!foundTarget) {
                if (t == targetToken) foundTarget = true;
                continue;
            }
            return t;
        }
        return null;
    }

    private static boolean isBlockAtlasArg(String ident) {
        return "texture".equals(ident) || "tex".equals(ident) || "gtexture".equals(ident)
                || "gcolor".equals(ident) || "vk_gtexture".equals(ident);
    }

    private static int countCallArgs(List<Token> tokens, int openParenIdx) {
        int depth = 0;
        int commas = 0;
        boolean hasContent = false;
        for (int i = openParenIdx + 1; i < tokens.size(); i++) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.NEWLINE || t.kind == TokenKind.PREPROC) continue;
            if (t.kind == TokenKind.PUNCT) {
                if (t.text.equals("(") || t.text.equals("[") || t.text.equals("{")) {
                    depth++;
                } else if (t.text.equals(")") || t.text.equals("]") || t.text.equals("}")) {
                    if (depth == 0 && t.text.equals(")")) {
                        return hasContent ? commas + 1 : 0;
                    }
                    if (depth > 0) depth--;
                } else if (t.text.equals(",") && depth == 0) {
                    commas++;
                }
            }
            hasContent = true;
        }
        return 0;
    }

    private static int findFirstTopLevelComma(List<Token> tokens, int start, int end) {
        int depth = 0;
        for (int i = start; i < end; i++) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.PUNCT) {
                if (t.text.equals("(") || t.text.equals("[") || t.text.equals("{")) depth++;
                else if ((t.text.equals(")") || t.text.equals("]") || t.text.equals("}")) && depth > 0) depth--;
                else if (t.text.equals(",") && depth == 0) return i;
            }
        }
        return -1;
    }

    private static String normalizeSamplerType(String word) {
        return switch (word) {
            case "sampler2DShadow", "sampler3D" -> "sampler2D";
            case "sampler1DShadow" -> "sampler1D";
            case "samplerCubeShadow" -> "samplerCube";
            case "sampler2DArrayShadow" -> "sampler2DArray";
            default -> null;
        };
    }

    private static String rewriteTokenSlice(
            ParsedShader parsed,
            List<Token> tokens,
            int start,
            int end,
            boolean isVertex,
            PassKind passKind,
            List<Set<String>> localScopes,
            Set<String> prototypeParams,
            Set<String> globalShadowedNames,
            Set<String> shadowCompareParams,
            boolean inStructDef
    ) {
        StringBuilder sub = new StringBuilder();
        for (int i = start; i < end; i++) {
            Token tok = tokens.get(i);
            if (tok.kind == TokenKind.NEWLINE) {
                sub.append('\n');
                continue;
            }
            if (tok.kind != TokenKind.WORD) {
                sub.append(tok.text);
                if (i + 1 < end) appendSpaceIfNeeded(tokens, i, sub);
                continue;
            }
            String w = tok.text;
            Token prev = prevCodeToken(tokens, i);
            boolean afterDot = prev != null && prev.kind == TokenKind.PUNCT && prev.text.equals(".");
            Token next = nextCodeTokenObj(tokens, i);
            boolean followedByParen = next != null && next.kind == TokenKind.PUNCT && next.text.equals("(");
            String normalizedSampler = afterDot ? null : normalizeSamplerType(w);
            if (afterDot) {
                sub.append(w);
            } else if (normalizedSampler != null) {
                sub.append(normalizedSampler);
            } else if (w.equals("sample")) {
                sub.append("vk_sample_var");
            } else if (w.equals("sampler")) {
                sub.append("vk_sampler_var");
            } else if (w.equals("input")) {
                sub.append("vk_input_var");
            } else if (w.equals("output")) {
                sub.append("vk_output_var");
            } else if (parsed.hasUserRound && w.equals("round") && followedByParen) {
                sub.append("vk_round");
            } else if (parsed.hasUserFma && w.equals("fma") && followedByParen) {
                sub.append("vk_fma");
            } else if (w.equals("normalize") && followedByParen && (prev == null || !".".equals(prev.text))) {
                sub.append("_vk_normalize");
            } else if (w.equals("texture") && followedByParen) {
                int lp = nextCodeToken(tokens, i + 1);
                String firstArg = firstArgIdentifier(tokens, lp + 1);
                if (firstArg != null && (parsed.shadowCompareSamplers.contains(firstArg)
                        || shadowCompareParams.contains(firstArg))) {
                    sub.append("vk_shadowTex");
                } else {
                    sub.append("vk_tex");
                }
            } else if (w.equals("textureLod") && followedByParen) {
                int lp = nextCodeToken(tokens, i + 1);
                String firstArg = firstArgIdentifier(tokens, lp + 1);
                if (firstArg != null && (parsed.shadowCompareSamplers.contains(firstArg)
                        || shadowCompareParams.contains(firstArg))) {
                    sub.append("vk_shadowTexLod");
                } else {
                    sub.append("vk_textureLod");
                }
            } else if (w.equals("texture") && !followedByParen) {
                sub.append(resolveSamplerAlias("texture", passKind));
            } else if (followedByParen && mapLegacySampleFunction(w) != null) {
                sub.append(mapLegacySampleFunction(w));
            } else if (!isShadowed(w, localScopes) && !prototypeParams.contains(w) && !globalShadowedNames.contains(w)
                    && resolveSamplerAlias(w, passKind) != null && !followedByParen) {
                sub.append(resolveSamplerAlias(w, passKind));
            } else if (!isShadowed(w, localScopes) && !prototypeParams.contains(w) && !globalShadowedNames.contains(w)
                    && parsed.unknownSamplers.containsKey(w) && !followedByParen) {
                sub.append(parsed.unknownSamplers.get(w));
            } else if (mapLegacyBuiltin(w, isVertex, passKind) != null) {
                sub.append(mapLegacyBuiltin(w, isVertex, passKind));
            } else if (!inStructDef && !followedByParen && !isPrecededByTypeOrQualifier(prev, parsed.structNames)
                    && !isShadowed(w, localScopes) && !prototypeParams.contains(w) && !globalShadowedNames.contains(w)) {
                if (ShaderUniformLayout.UBO_FIELD_NAMES.contains(w)) {
                    sub.append("vk_ubo.").append(w);
                } else if (parsed.resolvedCustomUniformNames.contains(w)) {
                    sub.append("_vk_cu_").append(w).append("()");
                } else if (parsed.declaredDerivedUniforms.contains(w)) {
                    String derived = resolveDerivedUniformForToken(w, passKind, prev, nextCodeTokenObj(tokens, i));
                    if (derived != null) {
                        sub.append(derived);
                    } else {
                        sub.append(w);
                    }
                } else {
                    sub.append(w);
                }
            } else {
                sub.append(w);
            }
            if (i + 1 < end) appendSpaceIfNeeded(tokens, i, sub);
        }
        return sub.toString();
    }

    private static String firstArgIdentifier(List<Token> tokens, int start) {
        int i = nextCodeToken(tokens, start);
        if (i >= tokens.size()) return null;
        Token first = tokens.get(i);
        if (first.kind != TokenKind.WORD) return null;
        Token after = nextCodeTokenObj(tokens, i);
        if (after != null && after.kind == TokenKind.PUNCT && (after.text.equals(",") || after.text.equals(")"))) {
            return first.text;
        }
        return null;
    }

    private static void appendSpaceIfNeeded(List<Token> tokens, int idx, StringBuilder sb) {
        if (idx + 1 >= tokens.size()) return;
        Token cur = tokens.get(idx);
        Token next = tokens.get(idx + 1);
        if (next.kind == TokenKind.NEWLINE) return;
        if (cur.kind == TokenKind.WORD && next.kind == TokenKind.WORD) {
            sb.append(' ');
        } else if (cur.kind == TokenKind.WORD && next.kind == TokenKind.NUMBER) {
            sb.append(' ');
        } else if (cur.kind == TokenKind.NUMBER && next.kind == TokenKind.WORD) {
            sb.append(' ');
        } else if (cur.kind == TokenKind.PUNCT && (cur.text.equals(",") || cur.text.equals(";"))) {
            sb.append(' ');
        } else if (cur.kind == TokenKind.WORD && cur.text.equals("return") && !next.text.equals(";")) {
            sb.append(' ');
        }
    }

    private static int nextCodeToken(List<Token> tokens, int start) {
        int i = start;
        while (i < tokens.size()) {
            TokenKind k = tokens.get(i).kind;
            if (k != TokenKind.NEWLINE && k != TokenKind.PREPROC) return i;
            i++;
        }
        return tokens.size();
    }

    private static Token nextCodeTokenObj(List<Token> tokens, int idx) {
        int n = nextCodeToken(tokens, idx + 1);
        return n < tokens.size() ? tokens.get(n) : null;
    }

    private static Token prevCodeToken(List<Token> tokens, int idx) {
        for (int i = idx - 1; i >= 0; i--) {
            TokenKind k = tokens.get(i).kind;
            if (k != TokenKind.NEWLINE && k != TokenKind.PREPROC) {
                return tokens.get(i);
            }
        }
        return null;
    }

    private static int findMatchingParen(List<Token> tokens, int openIdx) {
        int depth = 0;
        for (int i = openIdx; i < tokens.size(); i++) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.PUNCT) {
                if (t.text.equals("(")) depth++;
                else if (t.text.equals(")")) {
                    depth--;
                    if (depth == 0) return i;
                }
            }
        }
        return -1;
    }

    private static int findMatchingBracket(List<Token> tokens, int openIdx) {
        int depth = 0;
        for (int i = openIdx; i < tokens.size(); i++) {
            Token t = tokens.get(i);
            if (t.kind == TokenKind.PUNCT) {
                if (t.text.equals("[")) depth++;
                else if (t.text.equals("]")) {
                    depth--;
                    if (depth == 0) return i;
                }
            }
        }
        return -1;
    }
}
