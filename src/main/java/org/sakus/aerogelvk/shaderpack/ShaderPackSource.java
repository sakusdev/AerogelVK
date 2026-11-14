package org.sakus.aerogelvk.shaderpack;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Clean abstraction for reading OptiFine / Iris shaderpack files from either a
 * directory or a {@code .zip} archive (or an in-memory map for testing).
 *
 * <p>Supports standard shaderpack layouts:
 * <ul>
 *     <li>{@code shaders/...} at archive/directory root</li>
 *     <li>{@code <PackFolder>/shaders/...} inside an archive or wrapper folder</li>
 *     <li>Direct {@code shaders/} directory containing {@code .vsh}/{@code .fsh}/{@code world0/}</li>
 *     <li>Dimension sub-folders: {@code shaders/world0/...}, {@code shaders/world-1/...}, {@code shaders/world1/...}</li>
 * </ul>
 */
public final class ShaderPackSource implements AutoCloseable {

    public static final String WORLD_OVERWORLD = "world0";
    public static final String WORLD_NETHER = "world-1";
    public static final String WORLD_END = "world1";

    private final String name;
    private final Path sourcePath;
    private final boolean zipArchive;
    private final Map<String, byte[]> entries;

    private ShaderPackSource(String name, Path sourcePath, boolean zipArchive, Map<String, byte[]> entries) {
        this.name = name != null ? name : "unnamed";
        this.sourcePath = sourcePath;
        this.zipArchive = zipArchive;
        this.entries = Collections.unmodifiableMap(entries);
    }

    /**
     * Convenience constructor that opens either a directory or a {@code .zip} file.
     *
     * @param path directory or {@code .zip} archive path
     * @throws IOException if reading the pack fails
     */
    public ShaderPackSource(Path path) throws IOException {
        this(openInternal(path));
    }

    private ShaderPackSource(ShaderPackSource delegate) {
        this.name = delegate.name;
        this.sourcePath = delegate.sourcePath;
        this.zipArchive = delegate.zipArchive;
        this.entries = delegate.entries;
    }

    /**
     * Opens a shaderpack from either a directory or a {@code .zip} file.
     *
     * @param path directory or {@code .zip} file path
     * @return loaded {@link ShaderPackSource}
     * @throws IOException if the path does not exist or cannot be read
     */
    public static ShaderPackSource open(Path path) throws IOException {
        return openInternal(path);
    }

    /**
     * Opens a shaderpack from a directory on disk.
     */
    public static ShaderPackSource fromDirectory(Path directory) throws IOException {
        if (directory == null || !Files.isDirectory(directory)) {
            throw new IOException("ShaderPack directory does not exist: " + directory);
        }
        return loadFromDirectory(directory);
    }

    /**
     * Opens a shaderpack from a {@code .zip} archive on disk.
     */
    public static ShaderPackSource fromZip(Path zipPath) throws IOException {
        if (zipPath == null || !Files.isRegularFile(zipPath)) {
            throw new IOException("ShaderPack zip archive does not exist: " + zipPath);
        }
        return loadFromZip(zipPath);
    }

    /**
     * Creates an in-memory {@link ShaderPackSource} from a map of paths to UTF-8 contents.
     */
    public static ShaderPackSource fromMemory(String name, Map<String, String> textFiles) {
        Map<String, byte[]> raw = new LinkedHashMap<>();
        if (textFiles != null) {
            for (Map.Entry<String, String> entry : textFiles.entrySet()) {
                if (entry.getKey() != null && entry.getValue() != null) {
                    String norm = normalizeEntryKey(entry.getKey());
                    if (!norm.startsWith("shaders/")) {
                        norm = "shaders/" + norm;
                    }
                    raw.put(norm, entry.getValue().getBytes(StandardCharsets.UTF_8));
                }
            }
        }
        return new ShaderPackSource(name, null, false, raw);
    }

    private static ShaderPackSource openInternal(Path path) throws IOException {
        if (path == null || !Files.exists(path)) {
            throw new IOException("ShaderPack path does not exist: " + path);
        }
        if (Files.isDirectory(path)) {
            return loadFromDirectory(path);
        }
        String fileName = path.getFileName() != null ? path.getFileName().toString() : "";
        if (fileName.toLowerCase(Locale.ROOT).endsWith(".zip") || Files.isRegularFile(path)) {
            return loadFromZip(path);
        }
        throw new IOException("Unsupported shaderpack path: " + path);
    }

    private static ShaderPackSource loadFromDirectory(Path dir) throws IOException {
        Path absDir = dir.toAbsolutePath().normalize();
        String packName = derivePackName(absDir, false);

        Path shadersRoot = resolveShadersDirectoryRoot(absDir);
        Map<String, byte[]> loaded = new LinkedHashMap<>();

        if (shadersRoot != null && Files.isDirectory(shadersRoot)) {
            try (Stream<Path> walk = Files.walk(shadersRoot)) {
                List<Path> files = walk.filter(Files::isRegularFile).sorted().toList();
                for (Path file : files) {
                    String rel = shadersRoot.relativize(file).toString().replace('\\', '/');
                    String key = normalizeEntryKey("shaders/" + rel);
                    loaded.put(key, Files.readAllBytes(file));
                }
            }
        }
        return new ShaderPackSource(packName, absDir, false, loaded);
    }

    private static Path resolveShadersDirectoryRoot(Path dir) throws IOException {
        Path directShaders = dir.resolve("shaders");
        if (Files.isDirectory(directShaders) && hasShaderContent(directShaders)) {
            return directShaders;
        }
        if (hasShaderContent(dir)) {
            return dir;
        }
        // Check single nested folder (e.g. extracted pack folder containing shaders/)
        try (Stream<Path> children = Files.list(dir)) {
            List<Path> subDirs = children.filter(Files::isDirectory).toList();
            for (Path sub : subDirs) {
                Path nestedShaders = sub.resolve("shaders");
                if (Files.isDirectory(nestedShaders) && hasShaderContent(nestedShaders)) {
                    return nestedShaders;
                }
            }
        }
        if (Files.isDirectory(directShaders)) {
            return directShaders;
        }
        return null;
    }

    static boolean hasShaderContent(Path candidateShadersDir) {
        if (candidateShadersDir == null || !Files.isDirectory(candidateShadersDir)) {
            return false;
        }
        if (Files.isRegularFile(candidateShadersDir.resolve("shaders.properties"))
                || Files.isDirectory(candidateShadersDir.resolve(WORLD_OVERWORLD))
                || Files.isDirectory(candidateShadersDir.resolve(WORLD_NETHER))
                || Files.isDirectory(candidateShadersDir.resolve(WORLD_END))) {
            return true;
        }
        try (Stream<Path> stream = Files.walk(candidateShadersDir, 2)) {
            return stream.filter(Files::isRegularFile).anyMatch(p -> {
                String fn = p.getFileName().toString().toLowerCase(Locale.ROOT);
                return fn.endsWith(".vsh") || fn.endsWith(".fsh") || fn.endsWith(".csh")
                        || fn.endsWith(".glsl") || fn.equals("shaders.properties");
            });
        } catch (IOException e) {
            return false;
        }
    }

    private static ShaderPackSource loadFromZip(Path zipPath) throws IOException {
        Path absZip = zipPath.toAbsolutePath().normalize();
        String packName = derivePackName(absZip, true);

        Map<String, byte[]> rawEntries = new LinkedHashMap<>();
        try (ZipFile zf = new ZipFile(absZip.toFile(), StandardCharsets.UTF_8)) {
            Enumeration<? extends ZipEntry> enumeration = zf.entries();
            while (enumeration.hasMoreElements()) {
                ZipEntry entry = enumeration.nextElement();
                if (entry.isDirectory()) {
                    continue;
                }
                String normName = normalizeEntryKey(entry.getName());
                if (normName.isEmpty()) {
                    continue;
                }
                try (InputStream in = zf.getInputStream(entry)) {
                    rawEntries.put(normName, readAllBytes(in));
                }
            }
        }

        // Normalize keys so that files under ".../shaders/..." start at "shaders/..."
        Map<String, byte[]> normalized = new LinkedHashMap<>();
        for (Map.Entry<String, byte[]> entry : rawEntries.entrySet()) {
            String key = entry.getKey();
            if (key.startsWith("shaders/")) {
                normalized.put(key, entry.getValue());
            } else {
                int idx = key.indexOf("/shaders/");
                if (idx >= 0) {
                    String stripped = key.substring(idx + 1); // starts with "shaders/"
                    normalized.put(stripped, entry.getValue());
                }
            }
        }

        // Fallback: if zip had .vsh/.fsh directly at root without a "shaders/" folder
        if (normalized.isEmpty() && !rawEntries.isEmpty()) {
            boolean anyShaderAtRoot = rawEntries.keySet().stream().anyMatch(k ->
                    k.endsWith(".vsh") || k.endsWith(".fsh") || k.endsWith(".glsl")
                            || k.equals("shaders.properties") || k.startsWith("world0/"));
            if (anyShaderAtRoot) {
                for (Map.Entry<String, byte[]> entry : rawEntries.entrySet()) {
                    normalized.put("shaders/" + entry.getKey(), entry.getValue());
                }
            }
        }

        return new ShaderPackSource(packName, absZip, true, normalized);
    }

    private static byte[] readAllBytes(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(4096);
        byte[] buf = new byte[4096];
        int n;
        while ((n = in.read(buf)) != -1) {
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }

    private static String derivePackName(Path path, boolean isZip) {
        if (path == null || path.getFileName() == null) {
            return "shaderpack";
        }
        String fileName = path.getFileName().toString();
        if (isZip && fileName.toLowerCase(Locale.ROOT).endsWith(".zip")) {
            return fileName.substring(0, fileName.length() - 4);
        }
        return fileName;
    }

    /**
     * Returns the display/logical name of the shaderpack (without {@code .zip} suffix).
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the underlying filesystem path, or {@code null} for in-memory sources.
     */
    public Path getSourcePath() {
        return sourcePath;
    }

    /**
     * Returns {@code true} if this shaderpack was loaded from a {@code .zip} archive.
     */
    public boolean isZip() {
        return zipArchive;
    }

    /**
     * Returns {@code true} if this pack contains at least one file under {@code shaders/}.
     */
    public boolean isValidShaderPack() {
        return entries.keySet().stream().anyMatch(k ->
                k.endsWith(".vsh") || k.endsWith(".fsh") || k.endsWith(".glsl")
                        || k.equals("shaders/shaders.properties"));
    }

    /**
     * Returns all normalized file paths in this shaderpack (all starting with {@code shaders/}).
     */
    public Set<String> listFiles() {
        return new LinkedHashSet<>(entries.keySet());
    }

    /**
     * Alias for {@link #listFiles()}.
     */
    public Set<String> listShaderFiles() {
        return listFiles();
    }

    /**
     * Checks whether a file exists in the pack. Accepts paths with or without leading
     * {@code shaders/} or {@code /}, and also checks {@code shaders/world0/} fallback.
     */
    public boolean exists(String path) {
        return resolveExistingKey(path).isPresent();
    }

    /**
     * Reads a text file from the shaderpack as UTF-8, if present.
     */
    public Optional<String> readText(String path) {
        return readBytes(path).map(bytes -> stripBom(new String(bytes, StandardCharsets.UTF_8)));
    }

    /**
     * Reads a text file from the shaderpack as UTF-8, or returns {@code null} if not found.
     */
    public String readTextOrNull(String path) {
        return readText(path).orElse(null);
    }

    /**
     * Reads raw bytes of a file in the shaderpack, if present.
     */
    public Optional<byte[]> readBytes(String path) {
        Optional<String> key = resolveExistingKey(path);
        if (key.isEmpty()) {
            return Optional.empty();
        }
        byte[] data = entries.get(key.get());
        return data != null ? Optional.of(data.clone()) : Optional.empty();
    }

    /**
     * Resolves a shader file path using Overworld ({@code world0}) dimension priority:
     * checks {@code shaders/world0/<relativePath>} first, then {@code shaders/<relativePath>}.
     */
    public Optional<String> resolveShaderPath(String relativePath) {
        return resolveShaderPath(WORLD_OVERWORLD, relativePath);
    }

    /**
     * Resolves a shader file path for a specific dimension folder (e.g. {@code world0},
     * {@code world-1}, {@code world1}):
     * <ol>
     *     <li>If {@code relativePath} already explicitly names an existing file, returns it</li>
     *     <li>Checks {@code shaders/<worldFolder>/<rel>}</li>
     *     <li>Checks {@code shaders/<rel>}</li>
     *     <li>Checks {@code shaders/world0/<rel>} as fallback</li>
     * </ol>
     */
    public Optional<String> resolveShaderPath(String worldFolder, String relativePath) {
        if (relativePath == null || relativePath.isBlank()) {
            return Optional.empty();
        }
        String cleanRel = stripShadersPrefix(normalizeEntryKey(relativePath));
        String wf = (worldFolder == null || worldFolder.isBlank()) ? WORLD_OVERWORLD : worldFolder.trim();

        // If caller already passed an explicit world path like "world-1/gbuffers_terrain.vsh"
        if (cleanRel.startsWith("world")) {
            String direct = normalizeEntryKey("shaders/" + cleanRel);
            if (entries.containsKey(direct)) {
                return Optional.of(direct);
            }
        }

        // 1. Check shaders/<worldFolder>/<cleanRel>
        if (!cleanRel.startsWith(wf + "/")) {
            String worldCandidate = normalizeEntryKey("shaders/" + wf + "/" + cleanRel);
            if (entries.containsKey(worldCandidate)) {
                return Optional.of(worldCandidate);
            }
        }

        // 2. Check shaders/<cleanRel>
        String rootCandidate = normalizeEntryKey("shaders/" + cleanRel);
        if (entries.containsKey(rootCandidate)) {
            return Optional.of(rootCandidate);
        }

        // 3. Fallback to shaders/world0/<cleanRel>
        if (!WORLD_OVERWORLD.equals(wf) && !cleanRel.startsWith(WORLD_OVERWORLD + "/")) {
            String overworldCandidate = normalizeEntryKey("shaders/" + WORLD_OVERWORLD + "/" + cleanRel);
            if (entries.containsKey(overworldCandidate)) {
                return Optional.of(overworldCandidate);
            }
        }

        return Optional.empty();
    }

    /**
     * Resolves an {@code #include} target path relative to either {@code currentFilePath}'s
     * directory (for relative includes) or {@code /shaders} root (when {@code includeTarget}
     * starts with {@code /}), falling back between the two if needed.
     *
     * @param currentFilePath normalized path of the file containing the {@code #include}
     * @param includeTarget   the raw path inside {@code #include "..."}
     * @return the resolved key in this {@link ShaderPackSource}, if found
     */
    public Optional<String> resolveInclude(String currentFilePath, String includeTarget) {
        if (includeTarget == null || includeTarget.isBlank()) {
            return Optional.empty();
        }
        String trimmed = includeTarget.trim().replace('\\', '/');
        String currentNorm = currentFilePath != null ? normalizeToShadersKey(currentFilePath) : "shaders/";
        int lastSlash = currentNorm.lastIndexOf('/');
        String currentDir = lastSlash >= 0 ? currentNorm.substring(0, lastSlash) : "shaders";

        if (trimmed.startsWith("/")) {
            // Root-relative include: e.g. #include "/lib/config.glsl" -> "shaders/lib/config.glsl"
            String rootKey = normalizeEntryKey("shaders" + trimmed);
            if (entries.containsKey(rootKey)) {
                return Optional.of(rootKey);
            }
            // Fallback: relative to current file directory
            String relKey = normalizeEntryKey(currentDir + trimmed);
            if (entries.containsKey(relKey)) {
                return Optional.of(relKey);
            }
        } else {
            // Relative to current file directory first
            String relKey = normalizeEntryKey(currentDir + "/" + trimmed);
            if (entries.containsKey(relKey)) {
                return Optional.of(relKey);
            }
            // Fallback: relative to /shaders root
            String rootKey = normalizeEntryKey("shaders/" + trimmed);
            if (entries.containsKey(rootKey)) {
                return Optional.of(rootKey);
            }
        }

        // Case-insensitive fallback for packs authored on Windows with mixed-case filenames
        String targetRoot = trimmed.startsWith("/")
                ? normalizeEntryKey("shaders" + trimmed)
                : normalizeEntryKey(currentDir + "/" + trimmed);
        for (String existing : entries.keySet()) {
            if (existing.equalsIgnoreCase(targetRoot)) {
                return Optional.of(existing);
            }
        }
        String altRoot = normalizeEntryKey("shaders/" + stripShadersPrefix(trimmed));
        for (String existing : entries.keySet()) {
            if (existing.equalsIgnoreCase(altRoot)) {
                return Optional.of(existing);
            }
        }
        return Optional.empty();
    }

    /**
     * Computes a normalized {@code shaders/...} path for an include target relative to {@code currentFilePath}.
     */
    public static String normalizeShaderPath(String currentFilePath, String includeTarget) {
        if (includeTarget == null || includeTarget.isBlank()) {
            return "shaders/";
        }
        String trimmed = includeTarget.trim().replace('\\', '/');
        if (trimmed.startsWith("/")) {
            return normalizeEntryKey("shaders" + trimmed);
        }
        String currentNorm = currentFilePath != null ? normalizeToShadersKey(currentFilePath) : "shaders/";
        int lastSlash = currentNorm.lastIndexOf('/');
        String currentDir = lastSlash >= 0 ? currentNorm.substring(0, lastSlash) : "shaders";
        return normalizeEntryKey(currentDir + "/" + trimmed);
    }

    private Optional<String> resolveExistingKey(String path) {
        if (path == null || path.isBlank()) {
            return Optional.empty();
        }
        String direct = normalizeToShadersKey(path);
        if (entries.containsKey(direct)) {
            return Optional.of(direct);
        }
        String cleanRel = stripShadersPrefix(direct);
        String world0Key = normalizeEntryKey("shaders/" + WORLD_OVERWORLD + "/" + cleanRel);
        if (entries.containsKey(world0Key)) {
            return Optional.of(world0Key);
        }
        // Case-insensitive match
        for (String k : entries.keySet()) {
            if (k.equalsIgnoreCase(direct) || k.equalsIgnoreCase(world0Key)) {
                return Optional.of(k);
            }
        }
        return Optional.empty();
    }

    static String normalizeToShadersKey(String rawPath) {
        String norm = normalizeEntryKey(rawPath);
        if (norm.startsWith("shaders/")) {
            return norm;
        }
        return normalizeEntryKey("shaders/" + norm);
    }

    static String stripShadersPrefix(String normPath) {
        String s = normalizeEntryKey(normPath);
        if (s.startsWith("shaders/")) {
            return s.substring("shaders/".length());
        }
        return s;
    }

    static String normalizeEntryKey(String rawPath) {
        if (rawPath == null) {
            return "";
        }
        String replaced = rawPath.trim().replace('\\', '/');
        String[] parts = replaced.split("/");
        Deque<String> stack = new ArrayDeque<>();
        for (String part : parts) {
            if (part.isEmpty() || ".".equals(part)) {
                continue;
            }
            if ("..".equals(part)) {
                if (!stack.isEmpty() && !"shaders".equals(stack.peekLast())) {
                    stack.removeLast();
                }
            } else {
                stack.addLast(part);
            }
        }
        return String.join("/", stack);
    }

    private static String stripBom(String text) {
        if (text != null && !text.isEmpty() && text.charAt(0) == '\uFEFF') {
            return text.substring(1);
        }
        return text;
    }

    @Override
    public void close() {
        // Entries are cached in memory on load so no open file/zip handles are retained.
    }
}
