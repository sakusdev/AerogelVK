package org.sakus.aerogelvk.shaderpack;

import org.sakus.aerogelvk.shaderpack.GlslToVulkanTranspiler.PassKind;
import org.sakus.aerogelvk.shaderpack.GlslToVulkanTranspiler.TranspiledProgram;
import org.sakus.aerogelvk.shaderpack.ShaderPreprocessor.PreprocessResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Discovers OptiFine / Iris shaderpacks in {@code gameDir/shaderpacks/} and {@code gameDir/shaders/},
 * resolves program pairs ({@code .vsh} + {@code .fsh}) using standard OptiFine fallback chains,
 * preprocesses their sources, and collects {@link ShaderPackProperties} metadata.
 *
 * <p>This class and its data types ({@link ShaderPackPrograms}, {@link PassProgram}) are pure Java
 * with zero Vulkan or LWJGL dependencies so {@code GlslToVulkanTranspiler} and
 * {@code VkShaderPackPipeline} can consume them directly.</p>
 */
public final class ShaderPackManager {

    /**
     * Standard OptiFine program fallback chains.
     */
    public static final Map<String, List<String>> FALLBACK_CHAINS;

    static {
        if (System.getProperty("org.lwjgl.system.stackSize") == null) {
            System.setProperty("org.lwjgl.system.stackSize", "512");
        }
        Map<String, List<String>> chains = new LinkedHashMap<>();
        chains.put("shadow", List.of("shadow", "watershadow"));
        chains.put("gbuffers_basic", List.of("gbuffers_basic"));
        chains.put("gbuffers_textured", List.of("gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_textured_lit", List.of("gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_skybasic", List.of("gbuffers_skybasic", "gbuffers_basic"));
        chains.put("gbuffers_skytextured", List.of("gbuffers_skytextured", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_clouds", List.of("gbuffers_clouds", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_terrain", List.of("gbuffers_terrain", "gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_damagedblock", List.of("gbuffers_damagedblock", "gbuffers_terrain", "gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_block", List.of("gbuffers_block", "gbuffers_terrain", "gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_beaconbeam", List.of("gbuffers_beaconbeam", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_entities", List.of("gbuffers_entities", "gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_armor_glint", List.of("gbuffers_armor_glint", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_spidereyes", List.of("gbuffers_spidereyes", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_hand", List.of("gbuffers_hand", "gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_weather", List.of("gbuffers_weather", "gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_water", List.of("gbuffers_water", "gbuffers_terrain", "gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));
        chains.put("gbuffers_hand_water", List.of("gbuffers_hand_water", "gbuffers_water", "gbuffers_terrain", "gbuffers_textured_lit", "gbuffers_textured", "gbuffers_basic"));

        for (int i = 0; i <= 15; i++) {
            String defName = (i == 0) ? "deferred" : ("deferred" + i);
            chains.put(defName, List.of(defName));
        }
        for (int i = 0; i <= 15; i++) {
            String compName = (i == 0) ? "composite" : ("composite" + i);
            chains.put(compName, List.of(compName));
        }
        chains.put("final", List.of("final"));
        FALLBACK_CHAINS = Collections.unmodifiableMap(chains);
    }

    /**
     * Ordered list of core passes resolved for every shaderpack.
     */
    public static final List<String> CORE_PASS_ORDER;

    static {
        List<String> order = new ArrayList<>();
        for (int i = 0; i <= 15; i++) {
            order.add(i == 0 ? "prepare" : ("prepare" + i));
        }
        order.add("shadow");
        order.add("gbuffers_basic");
        order.add("gbuffers_textured");
        order.add("gbuffers_skybasic");
        order.add("gbuffers_clouds");
        order.add("gbuffers_terrain");
        order.add("gbuffers_entities");
        order.add("gbuffers_water");
        for (int i = 0; i <= 15; i++) {
            order.add(i == 0 ? "deferred" : ("deferred" + i));
        }
        for (int i = 0; i <= 15; i++) {
            order.add(i == 0 ? "composite" : ("composite" + i));
        }
        order.add("final");
        CORE_PASS_ORDER = Collections.unmodifiableList(order);
    }

    /**
     * Represents a resolved, preprocessed shader pass program pair ({@code .vsh} + {@code .fsh}).
     */
    public static final class PassProgram {
        private final String name;
        private final String resolvedFrom;
        private final PassKind passKind;
        private final String vertexPath;
        private final String fragmentPath;
        private final String rawVertexSource;
        private final String rawFragmentSource;
        private final String preprocessedVertexSource;
        private final String preprocessedFragmentSource;
        private final int[] drawBuffers;
        private final boolean explicitDrawBuffers;
        private final float scale;
        private final String alphaTest;
        private final String blend;
        private final Map<String, ShaderPackProperties.CustomUniform> customUniforms;

        public PassProgram(
                String name,
                String resolvedFrom,
                PassKind passKind,
                String vertexPath,
                String fragmentPath,
                String rawVertexSource,
                String rawFragmentSource,
                String preprocessedVertexSource,
                String preprocessedFragmentSource,
                int[] drawBuffers,
                boolean explicitDrawBuffers,
                float scale,
                String alphaTest,
                String blend) {
            this(name, resolvedFrom, passKind, vertexPath, fragmentPath, rawVertexSource, rawFragmentSource,
                    preprocessedVertexSource, preprocessedFragmentSource, drawBuffers, explicitDrawBuffers,
                    scale, alphaTest, blend, Collections.emptyMap());
        }

        public PassProgram(
                String name,
                String resolvedFrom,
                PassKind passKind,
                String vertexPath,
                String fragmentPath,
                String rawVertexSource,
                String rawFragmentSource,
                String preprocessedVertexSource,
                String preprocessedFragmentSource,
                int[] drawBuffers,
                boolean explicitDrawBuffers,
                float scale,
                String alphaTest,
                String blend,
                Map<String, ShaderPackProperties.CustomUniform> customUniforms) {
            this.name = name;
            this.resolvedFrom = resolvedFrom;
            this.passKind = passKind;
            this.vertexPath = vertexPath;
            this.fragmentPath = fragmentPath;
            this.rawVertexSource = rawVertexSource != null ? rawVertexSource : "";
            this.rawFragmentSource = rawFragmentSource != null ? rawFragmentSource : "";
            this.preprocessedVertexSource = preprocessedVertexSource != null ? preprocessedVertexSource : "";
            this.preprocessedFragmentSource = preprocessedFragmentSource != null ? preprocessedFragmentSource : "";
            this.drawBuffers = drawBuffers != null && drawBuffers.length > 0 ? drawBuffers.clone() : new int[]{0};
            this.explicitDrawBuffers = explicitDrawBuffers;
            this.scale = scale;
            this.alphaTest = alphaTest;
            this.blend = blend;
            this.customUniforms = customUniforms != null ? Map.copyOf(customUniforms) : Collections.emptyMap();
        }

        public String getName() {
            return name;
        }

        public String getResolvedFrom() {
            return resolvedFrom;
        }

        public String getResolvedName() {
            return resolvedFrom;
        }

        public PassKind getPassKind() {
            return passKind;
        }

        public PassKind getKind() {
            return passKind;
        }

        public String getVertexPath() {
            return vertexPath;
        }

        public String getFragmentPath() {
            return fragmentPath;
        }

        public String getRawVertexSource() {
            return rawVertexSource;
        }

        public String getRawFragmentSource() {
            return rawFragmentSource;
        }

        public String getPreprocessedVertexSource() {
            return preprocessedVertexSource;
        }

        public String getPreprocessedFragmentSource() {
            return preprocessedFragmentSource;
        }

        public String getVertexSource() {
            return preprocessedVertexSource;
        }

        public String getFragmentSource() {
            return preprocessedFragmentSource;
        }

        public int[] getDrawBuffers() {
            return drawBuffers.clone();
        }

        public boolean hasExplicitDrawBuffers() {
            return explicitDrawBuffers;
        }

        public float getScale() {
            return scale;
        }

        public String getAlphaTest() {
            return alphaTest;
        }

        public String getBlend() {
            return blend;
        }

        /**
         * Transpiles this preprocessed pass pair into Vulkan 1.3 GLSL 460 using {@link GlslToVulkanTranspiler}.
         */
        public TranspiledProgram transpile() {
            int[] dbsForTranspile = (passKind == PassKind.SHADOW && !explicitDrawBuffers)
                    ? new int[]{0}
                    : drawBuffers;
            return GlslToVulkanTranspiler.transpileProgram(
                    preprocessedVertexSource,
                    preprocessedFragmentSource,
                    passKind,
                    dbsForTranspile,
                    customUniforms);
        }

        @Override
        public String toString() {
            return "PassProgram{name='" + name + "', resolvedFrom='" + resolvedFrom
                    + "', kind=" + passKind + ", drawBuffers=" + java.util.Arrays.toString(drawBuffers) + "}";
        }
    }

    /**
     * Holds all resolved and preprocessed programs + properties for a compiled shaderpack.
     */
    public static final class ShaderPackPrograms {
        private final String packName;
        private final Path packPath;
        private final ShaderPackProperties properties;
        private final Map<String, PassProgram> programs;
        private final PassProgram shadow;
        private final PassProgram gbuffersTerrain;
        private final PassProgram gbuffersWater;
        private final PassProgram gbuffersSkyBasic;
        private final PassProgram gbuffersClouds;
        private final PassProgram gbuffersEntities;
        private final PassProgram gbuffersTextured;
        private final PassProgram gbuffersBasic;
        private final List<PassProgram> preparePasses;
        private final List<PassProgram> deferredPasses;
        private final List<PassProgram> compositePasses;
        private final PassProgram finalPass;

        public ShaderPackPrograms(
                String packName,
                Path packPath,
                ShaderPackProperties properties,
                Map<String, PassProgram> programs) {
            this.packName = packName != null ? packName : "unnamed";
            this.packPath = packPath;
            this.properties = properties != null ? properties : ShaderPackProperties.empty();
            this.programs = Collections.unmodifiableMap(new LinkedHashMap<>(programs));

            this.shadow = this.programs.get("shadow");
            this.gbuffersTerrain = this.programs.get("gbuffers_terrain");
            this.gbuffersWater = this.programs.get("gbuffers_water");
            this.gbuffersSkyBasic = this.programs.get("gbuffers_skybasic");
            this.gbuffersClouds = this.programs.get("gbuffers_clouds");
            this.gbuffersEntities = this.programs.get("gbuffers_entities");
            this.gbuffersTextured = this.programs.get("gbuffers_textured");
            this.gbuffersBasic = this.programs.get("gbuffers_basic");

            List<PassProgram> prepList = new ArrayList<>();
            for (int i = 0; i <= 15; i++) {
                String k = (i == 0) ? "prepare" : ("prepare" + i);
                PassProgram p = this.programs.get(k);
                if (p != null) {
                    prepList.add(p);
                }
            }
            this.preparePasses = Collections.unmodifiableList(prepList);

            List<PassProgram> defList = new ArrayList<>();
            for (int i = 0; i <= 15; i++) {
                String k = (i == 0) ? "deferred" : ("deferred" + i);
                PassProgram p = this.programs.get(k);
                if (p != null) {
                    defList.add(p);
                }
            }
            this.deferredPasses = Collections.unmodifiableList(defList);

            List<PassProgram> compList = new ArrayList<>();
            for (int i = 0; i <= 15; i++) {
                String k = (i == 0) ? "composite" : ("composite" + i);
                PassProgram p = this.programs.get(k);
                if (p != null) {
                    compList.add(p);
                }
            }
            this.compositePasses = Collections.unmodifiableList(compList);
            this.finalPass = this.programs.get("final");
        }

        public String getPackName() {
            return packName;
        }

        public Path getPackPath() {
            return packPath;
        }

        public ShaderPackProperties getProperties() {
            return properties;
        }

        public Map<String, PassProgram> getPrograms() {
            return programs;
        }

        public PassProgram getProgram(String name) {
            return programs.get(name);
        }

        public PassProgram getPass(String name) {
            return programs.get(name);
        }

        public Optional<PassProgram> findProgram(String name) {
            return Optional.ofNullable(programs.get(name));
        }

        public boolean hasProgram(String name) {
            return programs.containsKey(name);
        }

        public boolean hasPass(String name) {
            return programs.containsKey(name);
        }

        public PassProgram getShadow() {
            return shadow;
        }

        public PassProgram getGbuffersTerrain() {
            return gbuffersTerrain;
        }

        public PassProgram getGbuffersWater() {
            return gbuffersWater;
        }

        public PassProgram getGbuffersSkyBasic() {
            return gbuffersSkyBasic;
        }

        public PassProgram getGbuffersClouds() {
            return gbuffersClouds;
        }

        public PassProgram getGbuffersEntities() {
            return gbuffersEntities;
        }

        public PassProgram getGbuffersTextured() {
            return gbuffersTextured;
        }

        public PassProgram getGbuffersBasic() {
            return gbuffersBasic;
        }

        public List<PassProgram> getPreparePasses() {
            return preparePasses;
        }

        public List<PassProgram> getDeferredPasses() {
            return deferredPasses;
        }

        public List<PassProgram> getCompositePasses() {
            return compositePasses;
        }

        public PassProgram getFinalPass() {
            return finalPass;
        }

        public boolean isValid() {
            return !programs.isEmpty() && (gbuffersTerrain != null || finalPass != null);
        }
    }

    private ShaderPackPrograms activePrograms;

    public ShaderPackManager() {
    }

    public ShaderPackManager(ShaderPackPrograms activePrograms) {
        this.activePrograms = activePrograms;
    }

    public boolean isShaderPackActive() {
        return activePrograms != null && activePrograms.isValid();
    }

    public ShaderPackPrograms getActivePrograms() {
        return activePrograms;
    }

    public void setActivePrograms(ShaderPackPrograms activePrograms) {
        this.activePrograms = activePrograms;
    }

    /**
     * Scans both {@code gameDir/shaderpacks/} and {@code gameDir/shaders/} for valid
     * shaderpack {@code .zip} archives or directories containing {@code shaders/}.
     *
     * @param gameDir Minecraft game directory
     * @return ordered list of discovered shaderpack paths
     */
    public static List<Path> discoverShaderPacks(Path gameDir) {
        if (gameDir == null) {
            return List.of();
        }
        Set<Path> discovered = new LinkedHashSet<>();

        // 1. Scan gameDir/shaderpacks/
        Path shaderpacksDir = gameDir.resolve("shaderpacks");
        scanContainerDirectory(shaderpacksDir, discovered);

        // 2. Scan gameDir/shaders/
        Path shadersDir = gameDir.resolve("shaders");
        if (Files.isDirectory(shadersDir)) {
            scanContainerDirectory(shadersDir, discovered);
            // Also check if gameDir/shaders/ itself directly contains shader files (.vsh/.fsh/world0)
            if (ShaderPackSource.hasShaderContent(shadersDir)) {
                discovered.add(shadersDir.toAbsolutePath().normalize());
            }
        }

        return new ArrayList<>(discovered);
    }

    /**
     * Alias for {@link #discoverShaderPacks(Path)}.
     */
    public static List<Path> scanShaderPacks(Path gameDir) {
        return discoverShaderPacks(gameDir);
    }

    /**
     * Returns the display names of all discovered shaderpacks in {@code gameDir/shaderpacks/}
     * and {@code gameDir/shaders/}.
     */
    public static List<String> listShaderPackNames(Path gameDir) {
        List<Path> paths = discoverShaderPacks(gameDir);
        List<String> names = new ArrayList<>(paths.size());
        for (Path p : paths) {
            String fn = p.getFileName() != null ? p.getFileName().toString() : p.toString();
            names.add(fn);
        }
        return names;
    }

    /**
     * Resolves a shaderpack name (or {@code "auto"}) against {@code gameDir/shaderpacks/}
     * and {@code gameDir/shaders/}.
     *
     * @param gameDir  Minecraft game directory
     * @param packName pack name (with or without {@code .zip}), {@code "auto"}, or {@code "off"}
     * @return resolved {@link Path} if found
     */
    public static Optional<Path> findShaderPack(Path gameDir, String packName) {
        if (gameDir == null || packName == null) {
            return Optional.empty();
        }
        String trimmed = packName.trim();
        if (trimmed.isEmpty()
                || "off".equalsIgnoreCase(trimmed)
                || "none".equalsIgnoreCase(trimmed)
                || "false".equalsIgnoreCase(trimmed)
                || "(none)".equalsIgnoreCase(trimmed)) {
            return Optional.empty();
        }

        List<Path> available = discoverShaderPacks(gameDir);
        if ("auto".equalsIgnoreCase(trimmed) || "default".equalsIgnoreCase(trimmed)) {
            return available.isEmpty() ? Optional.empty() : Optional.of(available.get(0));
        }

        String withoutZip = trimmed.toLowerCase(Locale.ROOT).endsWith(".zip")
                ? trimmed.substring(0, trimmed.length() - 4)
                : trimmed;

        for (Path candidate : available) {
            String fn = candidate.getFileName() != null ? candidate.getFileName().toString() : "";
            String fnNoZip = fn.toLowerCase(Locale.ROOT).endsWith(".zip")
                    ? fn.substring(0, fn.length() - 4)
                    : fn;
            if (fn.equalsIgnoreCase(trimmed) || fnNoZip.equalsIgnoreCase(withoutZip)) {
                return Optional.of(candidate);
            }
        }

        // Also check direct path resolution under shaderpacks/, shaders/, or gameDir
        for (Path base : List.of(gameDir.resolve("shaderpacks"), gameDir.resolve("shaders"), gameDir)) {
            Path direct = base.resolve(trimmed);
            if (Files.exists(direct) && isCandidateShaderPack(direct)) {
                return Optional.of(direct.toAbsolutePath().normalize());
            }
            Path withZip = base.resolve(withoutZip + ".zip");
            if (Files.exists(withZip) && isCandidateShaderPack(withZip)) {
                return Optional.of(withZip.toAbsolutePath().normalize());
            }
        }

        return Optional.empty();
    }

    /**
     * Alias for {@link #findShaderPack(Path, String)}.
     */
    public static Optional<Path> resolveShaderPackPath(Path gameDir, String packName) {
        return findShaderPack(gameDir, packName);
    }

    private static void scanContainerDirectory(Path containerDir, Set<Path> out) {
        if (containerDir == null || !Files.isDirectory(containerDir)) {
            return;
        }
        try (Stream<Path> stream = Files.list(containerDir)) {
            List<Path> children = stream
                    .sorted(Comparator.comparing(p -> p.getFileName() != null ? p.getFileName().toString().toLowerCase(Locale.ROOT) : ""))
                    .toList();
            for (Path child : children) {
                if (isCandidateShaderPack(child)) {
                    out.add(child.toAbsolutePath().normalize());
                }
            }
        } catch (IOException ignored) {
        }
    }

    private static boolean isCandidateShaderPack(Path path) {
        if (path == null || !Files.exists(path)) {
            return false;
        }
        if (Files.isRegularFile(path)) {
            String fn = path.getFileName() != null ? path.getFileName().toString().toLowerCase(Locale.ROOT) : "";
            if (!fn.endsWith(".zip")) {
                return false;
            }
            try (ShaderPackSource src = ShaderPackSource.fromZip(path)) {
                return src.isValidShaderPack();
            } catch (IOException e) {
                return false;
            }
        }
        if (Files.isDirectory(path)) {
            Path shadersSub = path.resolve("shaders");
            if (Files.isDirectory(shadersSub) && ShaderPackSource.hasShaderContent(shadersSub)) {
                return true;
            }
            if (ShaderPackSource.hasShaderContent(path)) {
                return true;
            }
            try (Stream<Path> children = Files.list(path)) {
                for (Path sub : children.filter(Files::isDirectory).toList()) {
                    if (Files.isDirectory(sub.resolve("shaders")) && ShaderPackSource.hasShaderContent(sub.resolve("shaders"))) {
                        return true;
                    }
                }
            } catch (IOException ignored) {
            }
        }
        return false;
    }

    /**
     * Loads and preprocesses all shader programs in a shaderpack at {@code packPath}
     * for the Overworld ({@code world0}).
     */
    public static ShaderPackPrograms compileShaderPack(Path packPath) throws IOException {
        return compileShaderPack(packPath, ShaderPackSource.WORLD_OVERWORLD, null);
    }

    /**
     * Alias for {@link #compileShaderPack(Path)}.
     */
    public static ShaderPackPrograms loadShaderPack(Path packPath) throws IOException {
        return compileShaderPack(packPath);
    }

    /**
     * Finds and compiles a named shaderpack from {@code gameDir}.
     */
    public static Optional<ShaderPackPrograms> loadShaderPack(Path gameDir, String packName) throws IOException {
        Optional<Path> path = findShaderPack(gameDir, packName);
        if (path.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(compileShaderPack(path.get()));
    }

    /**
     * Loads and preprocesses all shader programs in a shaderpack at {@code packPath}
     * for the specified {@code worldFolder} and optional custom macro definitions.
     */
    public static ShaderPackPrograms compileShaderPack(
            Path packPath,
            String worldFolder,
            Map<String, String> customDefines) throws IOException {
        try (ShaderPackSource source = ShaderPackSource.open(packPath)) {
            return compileShaderPack(source, worldFolder, customDefines);
        }
    }

    /**
     * Compiles a {@link ShaderPackSource} for the Overworld ({@code world0}).
     */
    public static ShaderPackPrograms compileShaderPack(ShaderPackSource source) {
        return compileShaderPack(source, ShaderPackSource.WORLD_OVERWORLD, null);
    }

    /**
     * Compiles a {@link ShaderPackSource} for the specified dimension folder and custom macro definitions.
     */
    public static ShaderPackPrograms compileShaderPack(
            ShaderPackSource source,
            String worldFolder,
            Map<String, String> customDefines) {
        if (source == null) {
            throw new IllegalArgumentException("ShaderPackSource must not be null");
        }
        String wf = (worldFolder == null || worldFolder.isBlank())
                ? ShaderPackSource.WORLD_OVERWORLD
                : worldFolder.trim();

        ShaderPackProperties properties = ShaderPackProperties.load(source);
        Map<String, PassProgram> resolvedPrograms = new LinkedHashMap<>();
        Map<String, PreprocessedPairCache> pairCache = new LinkedHashMap<>();

        for (String passName : CORE_PASS_ORDER) {
            Optional<ResolvedPairPaths> pairPaths = resolveProgramPairPaths(source, wf, passName);
            if (pairPaths.isEmpty()) {
                continue;
            }
            ResolvedPairPaths resolved = pairPaths.get();
            PassKind kind = classifyPassKind(passName);

            PreprocessedPairCache cached = pairCache.computeIfAbsent(
                    resolved.resolvedFromName,
                    k -> preprocessPair(source, resolved, customDefines, properties));

            // Use explicit drawBuffers from the fragment stage (or properties if overridden)
            int[] drawBuffers = cached.fragResult.hasExplicitDrawBuffers()
                    ? cached.fragResult.getDrawBuffers()
                    : properties.getDrawBuffers(passName);
            boolean explicitDbs = cached.fragResult.hasExplicitDrawBuffers();
            properties.setDrawBuffers(passName, drawBuffers);

            PassProgram passProgram = new PassProgram(
                    passName,
                    resolved.resolvedFromName,
                    kind,
                    resolved.vertexPath,
                    resolved.fragmentPath,
                    cached.rawVertex,
                    cached.rawFragment,
                    cached.vertResult.getSource(),
                    cached.fragResult.getSource(),
                    drawBuffers,
                    explicitDbs,
                    properties.getPassScale(passName),
                    properties.getAlphaTest(passName),
                    properties.getBlend(passName),
                    properties.getCustomUniforms());

            resolvedPrograms.put(passName, passProgram);
        }

        return new ShaderPackPrograms(source.getName(), source.getSourcePath(), properties, resolvedPrograms);
    }

    private record ResolvedPairPaths(String resolvedFromName, String vertexPath, String fragmentPath) {
    }

    private record PreprocessedPairCache(
            String rawVertex,
            String rawFragment,
            PreprocessResult vertResult,
            PreprocessResult fragResult) {
    }

    private static PreprocessedPairCache preprocessPair(
            ShaderPackSource source,
            ResolvedPairPaths resolved,
            Map<String, String> customDefines,
            ShaderPackProperties properties) {
        String rawVsh = source.readText(resolved.vertexPath).orElse("");
        String rawFsh = source.readText(resolved.fragmentPath).orElse("");

        PreprocessResult vertRes = ShaderPreprocessor.preprocess(
                source, resolved.vertexPath, customDefines, properties);
        PreprocessResult fragRes = ShaderPreprocessor.preprocess(
                source, resolved.fragmentPath, customDefines, properties);

        return new PreprocessedPairCache(rawVsh, rawFsh, vertRes, fragRes);
    }

    /**
     * Resolves a program's {@code .vsh} and {@code .fsh} paths in {@code source} using
     * OptiFine's fallback chain for {@code passName}.
     */
    public static Optional<String[]> resolveProgramPair(ShaderPackSource source, String passName) {
        return resolveProgramPair(source, ShaderPackSource.WORLD_OVERWORLD, passName);
    }

    /**
     * Resolves a program's {@code .vsh} and {@code .fsh} paths in {@code source} for {@code worldFolder}
     * using OptiFine's fallback chain for {@code passName}.
     */
    public static Optional<String[]> resolveProgramPair(ShaderPackSource source, String worldFolder, String passName) {
        return resolveProgramPairPaths(source, worldFolder, passName)
                .map(r -> new String[]{r.vertexPath, r.fragmentPath, r.resolvedFromName});
    }

    private static Optional<ResolvedPairPaths> resolveProgramPairPaths(
            ShaderPackSource source,
            String worldFolder,
            String passName) {
        if (source == null || passName == null || passName.isBlank()) {
            return Optional.empty();
        }
        List<String> candidates = FALLBACK_CHAINS.getOrDefault(passName, List.of(passName));
        for (String candidate : candidates) {
            Optional<String> vsh = source.resolveShaderPath(worldFolder, candidate + ".vsh");
            Optional<String> fsh = source.resolveShaderPath(worldFolder, candidate + ".fsh");
            if (vsh.isPresent() && fsh.isPresent()) {
                return Optional.of(new ResolvedPairPaths(candidate, vsh.get(), fsh.get()));
            }
        }
        return Optional.empty();
    }

    /**
     * Classifies a pass name into its {@link PassKind}.
     */
    public static PassKind classifyPassKind(String passName) {
        if (passName == null) {
            return PassKind.GBUFFERS;
        }
        String lower = passName.trim().toLowerCase(Locale.ROOT);
        if (lower.startsWith("shadow") || lower.equals("watershadow")) {
            return PassKind.SHADOW;
        }
        if (lower.startsWith("deferred") || lower.startsWith("prepare")) {
            return PassKind.DEFERRED;
        }
        if (lower.startsWith("composite")) {
            return PassKind.COMPOSITE;
        }
        if (lower.equals("final")) {
            return PassKind.FINAL;
        }
        return PassKind.GBUFFERS;
    }
}
