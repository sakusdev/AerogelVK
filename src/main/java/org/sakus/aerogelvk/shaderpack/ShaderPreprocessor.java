package org.sakus.aerogelvk.shaderpack;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Full GLSL preprocessor for OptiFine / Iris shaderpacks.
 *
 * <p>Features:
 * <ul>
 *     <li>Recursive {@code #include} resolution relative to either the current file's directory
 *         or the pack's {@code /shaders} root, with active-stack cycle protection</li>
 *     <li>{@code #define} (both object-like and function-like macros) and {@code #undef}</li>
 *     <li>{@code #ifdef}, {@code #ifndef}, {@code #if}, {@code #elif}, {@code #else}, {@code #endif}
 *         supporting {@code defined(X)}, {@code defined X}, {@code &&}, {@code ||}, {@code !},
 *         {@code ==}, {@code !=}, {@code <}, {@code <=}, {@code >}, {@code >=}, arithmetic
 *         ({@code +}, {@code -}, {@code *}, {@code /}, {@code %}), bitwise ops, ternary {@code ?:},
 *         and nested parentheses</li>
 *     <li>Extracts and strips OptiFine comment directives ({@code /* DRAWBUFFERS:... * /},
 *         {@code /* RENDERTARGETS:... * /}, {@code /* const int colortex*Format = ...; * /}, etc.)
 *         and active {@code const} configuration directives into {@link PreprocessResult} and
 *         {@link ShaderPackProperties}</li>
 * </ul>
 */
public final class ShaderPreprocessor {

    private static final int MAX_INCLUDE_DEPTH = 64;
    private static final int MAX_MACRO_EXPANSION_PASSES = 16;

    private static final Pattern DIRECTIVE_PATTERN = Pattern.compile(
            "^\\s*#\\s*([A-Za-z_][A-Za-z0-9_]*)\\b(.*)$");

    private static final Pattern INCLUDE_TARGET_PATTERN = Pattern.compile(
            "^\\s*[\"<]([^\">]+)[\">]");

    private static final Pattern FUNC_DEFINE_PATTERN = Pattern.compile(
            "^([A-Za-z_][A-Za-z0-9_]*)\\(([^)]*)\\)\\s*(.*)$");

    private static final Pattern OBJ_DEFINE_PATTERN = Pattern.compile(
            "^([A-Za-z_][A-Za-z0-9_]*)(?:\\s+(.*))?$");

    private static final Pattern IDENTIFIER_PATTERN = Pattern.compile(
            "\\b([A-Za-z_][A-Za-z0-9_]*)\\b");

    private static final Pattern UNQUOTED_FORMAT_CONST_PATTERN = Pattern.compile(
            "^\\s*const\\s+int\\s+(colortex(?:1[0-5]|[0-9])|gcolor|gdepth|gnormal|composite|gaux[1-4]|shadowcolor[01]|noisetex)Format\\s*=\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*;\\s*$");

    /**
     * Default built-in macros exposed to OptiFine / Iris shaderpacks.
     */
    public static Map<String, String> defaultBuiltInMacros() {
        Map<String, String> defs = new LinkedHashMap<>();
        defs.put("MC_VERSION", "12001");
        defs.put("MC_GL_VERSION", "460");
        defs.put("MC_GLSL_VERSION", "460");
        defs.put("MC_OS_LINUX", "1");
        defs.put("MC_HAND_DEPTH", "0.125");
        defs.put("MC_RENDER_QUALITY", "1.0");
        defs.put("MC_SHADOW_QUALITY", "1.0");
        defs.put("MC_NORMAL_MAP", "1");
        defs.put("MC_SPECULAR_MAP", "1");
        defs.put("MC_RENDER_STAGE_NONE", "0");
        defs.put("MC_RENDER_STAGE_SKY", "1");
        defs.put("MC_RENDER_STAGE_SUNSET", "2");
        defs.put("MC_RENDER_STAGE_CUSTOM_SKY", "3");
        defs.put("MC_RENDER_STAGE_SUN", "4");
        defs.put("MC_RENDER_STAGE_MOON", "5");
        defs.put("MC_RENDER_STAGE_STARS", "6");
        defs.put("MC_RENDER_STAGE_VOID", "7");
        defs.put("MC_RENDER_STAGE_TERRAIN_SOLID", "8");
        defs.put("MC_RENDER_STAGE_TERRAIN_CUTOUT_MIPPED", "9");
        defs.put("MC_RENDER_STAGE_TERRAIN_CUTOUT", "10");
        defs.put("MC_RENDER_STAGE_ENTITIES", "11");
        defs.put("MC_RENDER_STAGE_BLOCK_ENTITIES", "12");
        defs.put("MC_RENDER_STAGE_DESTROY", "13");
        defs.put("MC_RENDER_STAGE_OUTLINE", "14");
        defs.put("MC_RENDER_STAGE_DEBUG", "15");
        defs.put("MC_RENDER_STAGE_HAND_SOLID", "16");
        defs.put("MC_RENDER_STAGE_TERRAIN_TRANSLUCENT", "17");
        defs.put("MC_RENDER_STAGE_TRIPWIRE", "18");
        defs.put("MC_RENDER_STAGE_PARTICLES", "19");
        defs.put("MC_RENDER_STAGE_CLOUDS", "20");
        defs.put("MC_RENDER_STAGE_RAIN_SNOW", "21");
        defs.put("MC_RENDER_STAGE_WORLD_BORDER", "22");
        defs.put("MC_RENDER_STAGE_HAND_TRANSLUCENT", "23");
        return defs;
    }

    /**
     * Result of preprocessing a shader stage or file.
     */
    public static final class PreprocessResult {
        private final String source;
        private final int[] drawBuffers;
        private final boolean explicitDrawBuffers;
        private final Map<String, String> definedMacros;
        private final ShaderPackProperties properties;
        private final List<String> includedFiles;
        private final List<String> collectedDirectiveComments;

        public PreprocessResult(
                String source,
                int[] drawBuffers,
                boolean explicitDrawBuffers,
                Map<String, String> definedMacros,
                ShaderPackProperties properties,
                List<String> includedFiles,
                List<String> collectedDirectiveComments) {
            this.source = source != null ? source : "";
            this.drawBuffers = drawBuffers != null && drawBuffers.length > 0 ? drawBuffers.clone() : new int[]{0};
            this.explicitDrawBuffers = explicitDrawBuffers;
            this.definedMacros = Collections.unmodifiableMap(new LinkedHashMap<>(definedMacros));
            this.properties = properties;
            this.includedFiles = Collections.unmodifiableList(new ArrayList<>(includedFiles));
            this.collectedDirectiveComments = Collections.unmodifiableList(new ArrayList<>(collectedDirectiveComments));
        }

        public String getSource() {
            return source;
        }

        public int[] getDrawBuffers() {
            return drawBuffers.clone();
        }

        public boolean hasExplicitDrawBuffers() {
            return explicitDrawBuffers;
        }

        public Map<String, String> getDefinedMacros() {
            return definedMacros;
        }

        public ShaderPackProperties getProperties() {
            return properties;
        }

        public List<String> getIncludedFiles() {
            return includedFiles;
        }

        public List<String> getCollectedDirectiveComments() {
            return collectedDirectiveComments;
        }
    }

    private static final class MacroDefinition {
        final String name;
        final List<String> params; // null for object-like macro
        final String body;

        MacroDefinition(String name, List<String> params, String body) {
            this.name = name;
            this.params = params;
            this.body = body != null ? body.trim() : "";
        }

        boolean isFunctionLike() {
            return params != null;
        }
    }

    private static final class ConditionalFrame {
        final boolean parentActive;
        boolean branchActive;
        boolean anyBranchTaken;

        ConditionalFrame(boolean parentActive, boolean initialBranchActive) {
            this.parentActive = parentActive;
            this.branchActive = parentActive && initialBranchActive;
            this.anyBranchTaken = this.branchActive;
        }
    }

    public ShaderPreprocessor() {
    }

    /**
     * Preprocesses an entry shader file from a {@link ShaderPackSource} using default macros.
     */
    public static PreprocessResult preprocess(ShaderPackSource packSource, String entryPath) {
        return preprocess(packSource, entryPath, null, null);
    }

    /**
     * Preprocesses an entry shader file from a {@link ShaderPackSource}, updating {@code properties}
     * with any discovered OptiFine comment or {@code const} directives.
     */
    public static PreprocessResult preprocess(
            ShaderPackSource packSource,
            String entryPath,
            Map<String, String> initialDefines,
            ShaderPackProperties properties) {
        if (packSource == null) {
            throw new IllegalArgumentException("packSource must not be null");
        }
        if (entryPath == null || entryPath.isBlank()) {
            throw new IllegalArgumentException("entryPath must not be null or blank");
        }
        String resolvedEntry = packSource.resolveShaderPath(entryPath)
                .or(() -> packSource.exists(entryPath) ? Optional.of(ShaderPackSource.normalizeToShadersKey(entryPath)) : Optional.empty())
                .orElseThrow(() -> new IllegalArgumentException("Shader entry file not found in pack: " + entryPath));

        String rawContent = packSource.readText(resolvedEntry)
                .orElseThrow(() -> new IllegalArgumentException("Unable to read shader entry file: " + resolvedEntry));

        String passName = derivePassName(resolvedEntry);
        ShaderPackProperties targetProps = properties != null ? properties : ShaderPackProperties.empty();

        BiFunction<String, String, Optional<ResolvedInclude>> resolver = (currentFile, target) -> {
            Optional<String> incKey = packSource.resolveInclude(currentFile, target);
            if (incKey.isEmpty()) {
                return Optional.empty();
            }
            return packSource.readText(incKey.get()).map(txt -> new ResolvedInclude(incKey.get(), txt));
        };

        return runPreprocessor(resolvedEntry, passName, rawContent, resolver, initialDefines, targetProps);
    }

    /**
     * Convenience method returning only the preprocessed GLSL string for an entry file in {@code packSource}.
     */
    public static String preprocessSource(ShaderPackSource packSource, String entryPath) {
        return preprocess(packSource, entryPath).getSource();
    }

    /**
     * Preprocesses a standalone raw GLSL string with optional initial macro definitions.
     */
    public static PreprocessResult preprocessRaw(String rawGlsl, Map<String, String> initialDefines) {
        return preprocessRaw(rawGlsl, (Function<String, String>) null, initialDefines, null);
    }

    /**
     * Preprocesses a raw GLSL string using a custom include lookup function {@code includePath -> content}.
     */
    public static PreprocessResult preprocessRaw(
            String rawGlsl,
            Function<String, String> includeResolver,
            Map<String, String> initialDefines,
            ShaderPackProperties properties) {
        ShaderPackProperties targetProps = properties != null ? properties : ShaderPackProperties.empty();
        BiFunction<String, String, Optional<ResolvedInclude>> biResolver = (currentFile, incTarget) -> {
            if (includeResolver == null) {
                return Optional.empty();
            }
            String norm = ShaderPackSource.normalizeShaderPath(currentFile, incTarget);
            String content = includeResolver.apply(incTarget);
            if (content == null) {
                content = includeResolver.apply(norm);
            }
            if (content == null && incTarget.startsWith("/")) {
                content = includeResolver.apply(incTarget.substring(1));
            }
            return content != null ? Optional.of(new ResolvedInclude(norm, content)) : Optional.empty();
        };
        return runPreprocessor("shaders/inline.glsl", "inline", rawGlsl != null ? rawGlsl : "", biResolver, initialDefines, targetProps);
    }

    private record ResolvedInclude(String normalizedPath, String content) {
    }

    private static PreprocessResult runPreprocessor(
            String entryFilePath,
            String passName,
            String entryContent,
            BiFunction<String, String, Optional<ResolvedInclude>> includeResolver,
            Map<String, String> initialDefines,
            ShaderPackProperties properties) {

        Map<String, MacroDefinition> macros = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : defaultBuiltInMacros().entrySet()) {
            macros.put(e.getKey(), new MacroDefinition(e.getKey(), null, e.getValue()));
        }
        // Automatically define dimension macro if entryFilePath is under world-1 or world1
        if (entryFilePath.contains("/world-1/")) {
            macros.put("NETHER", new MacroDefinition("NETHER", null, "1"));
            macros.put("WORLD_NETHER", new MacroDefinition("WORLD_NETHER", null, "1"));
        } else if (entryFilePath.contains("/world1/")) {
            macros.put("THE_END", new MacroDefinition("THE_END", null, "1"));
            macros.put("WORLD_END", new MacroDefinition("WORLD_END", null, "1"));
        } else {
            macros.put("OVERWORLD", new MacroDefinition("OVERWORLD", null, "1"));
            macros.put("WORLD_OVERWORLD", new MacroDefinition("WORLD_OVERWORLD", null, "1"));
        }

        if (initialDefines != null) {
            for (Map.Entry<String, String> e : initialDefines.entrySet()) {
                if (e.getKey() != null && !e.getKey().isBlank()) {
                    macros.put(e.getKey().trim(), new MacroDefinition(e.getKey().trim(), null, e.getValue() != null ? e.getValue() : ""));
                }
            }
        }

        Deque<ConditionalFrame> condStack = new ArrayDeque<>();
        Deque<String> activeIncludeStack = new ArrayDeque<>();
        List<String> includedFiles = new ArrayList<>();
        List<String> directiveComments = new ArrayList<>();
        StringBuilder output = new StringBuilder(entryContent.length() + 1024);
        int[][] activeDrawBuffersHolder = new int[][]{null};

        activeIncludeStack.push(entryFilePath);
        processFileContent(
                entryFilePath,
                passName,
                entryContent,
                includeResolver,
                macros,
                condStack,
                activeIncludeStack,
                includedFiles,
                directiveComments,
                properties,
                activeDrawBuffersHolder,
                output);
        activeIncludeStack.pop();

        boolean explicitDbs = activeDrawBuffersHolder[0] != null && activeDrawBuffersHolder[0].length > 0;
        int[] finalDbs = explicitDbs ? activeDrawBuffersHolder[0] : new int[]{0};
        if (explicitDbs && passName != null && !passName.isBlank()) {
            properties.setDrawBuffers(passName, finalDbs);
        }

        Map<String, String> flatMacros = new LinkedHashMap<>();
        for (Map.Entry<String, MacroDefinition> entry : macros.entrySet()) {
            flatMacros.put(entry.getKey(), entry.getValue().body);
        }

        return new PreprocessResult(
                output.toString(),
                finalDbs,
                explicitDbs,
                flatMacros,
                properties,
                includedFiles,
                directiveComments);
    }

    private static void processFileContent(
            String currentFilePath,
            String passName,
            String rawText,
            BiFunction<String, String, Optional<ResolvedInclude>> includeResolver,
            Map<String, MacroDefinition> macros,
            Deque<ConditionalFrame> condStack,
            Deque<String> activeIncludeStack,
            List<String> includedFiles,
            List<String> directiveComments,
            ShaderPackProperties properties,
            int[][] activeDrawBuffersHolder,
            StringBuilder output) {

        List<String> lines = joinBackslashLines(rawText);
        boolean inBlockComment = false;
        StringBuilder currentBlockComment = new StringBuilder();

        for (String rawLine : lines) {
            boolean branchActive = isCurrentlyActive(condStack);

            // Strip comments on this line while capturing /* ... */ blocks for OptiFine directives
            StringBuilder cleanLineBuilder = new StringBuilder(rawLine.length());
            int i = 0;
            int len = rawLine.length();
            String trailingLineComment = null;

            while (i < len) {
                if (inBlockComment) {
                    int endIdx = rawLine.indexOf("*/", i);
                    if (endIdx >= 0) {
                        currentBlockComment.append(rawLine, i, endIdx + 2);
                        if (branchActive) {
                            handleCommentBlock(passName, currentBlockComment.toString(), directiveComments, properties, activeDrawBuffersHolder);
                        }
                        currentBlockComment.setLength(0);
                        inBlockComment = false;
                        cleanLineBuilder.append(' ');
                        i = endIdx + 2;
                    } else {
                        currentBlockComment.append(rawLine, i, len).append('\n');
                        i = len;
                    }
                } else {
                    int nextLineComment = rawLine.indexOf("//", i);
                    int nextBlockComment = rawLine.indexOf("/*", i);

                    if (nextLineComment >= 0 && (nextBlockComment < 0 || nextLineComment < nextBlockComment)) {
                        cleanLineBuilder.append(rawLine, i, nextLineComment);
                        trailingLineComment = rawLine.substring(nextLineComment + 2);
                        break;
                    } else if (nextBlockComment >= 0) {
                        cleanLineBuilder.append(rawLine, i, nextBlockComment);
                        inBlockComment = true;
                        currentBlockComment.setLength(0);
                        currentBlockComment.append("/*");
                        i = nextBlockComment + 2;
                    } else {
                        cleanLineBuilder.append(rawLine, i, len);
                        break;
                    }
                }
            }

            String line = cleanLineBuilder.toString();
            String trimmed = line.trim();
            if (trimmed.isEmpty()) {
                continue;
            }

            Matcher dirMatcher = DIRECTIVE_PATTERN.matcher(trimmed);
            if (dirMatcher.matches()) {
                String directive = dirMatcher.group(1);
                String rest = dirMatcher.group(2) != null ? dirMatcher.group(2).trim() : "";

                switch (directive) {
                    case "ifdef" -> {
                        String macroName = extractFirstToken(rest);
                        boolean cond = macros.containsKey(macroName);
                        condStack.push(new ConditionalFrame(branchActive, cond));
                    }
                    case "ifndef" -> {
                        String macroName = extractFirstToken(rest);
                        boolean cond = !macros.containsKey(macroName);
                        condStack.push(new ConditionalFrame(branchActive, cond));
                    }
                    case "if" -> {
                        boolean cond = branchActive && evaluateConditionInternal(rest, macros);
                        condStack.push(new ConditionalFrame(branchActive, cond));
                    }
                    case "elif" -> {
                        if (!condStack.isEmpty()) {
                            ConditionalFrame top = condStack.peek();
                            if (!top.parentActive || top.anyBranchTaken) {
                                top.branchActive = false;
                            } else {
                                boolean cond = evaluateConditionInternal(rest, macros);
                                top.branchActive = cond;
                                top.anyBranchTaken = cond;
                            }
                        }
                    }
                    case "else" -> {
                        if (!condStack.isEmpty()) {
                            ConditionalFrame top = condStack.peek();
                            top.branchActive = top.parentActive && !top.anyBranchTaken;
                            top.anyBranchTaken = true;
                        }
                    }
                    case "endif" -> {
                        if (!condStack.isEmpty()) {
                            condStack.pop();
                        }
                    }
                    case "define" -> {
                        if (branchActive) {
                            parseAndRegisterDefine(rest, trailingLineComment, macros, properties);
                        }
                    }
                    case "undef" -> {
                        if (branchActive) {
                            String name = extractFirstToken(rest);
                            if (!name.isEmpty()) {
                                macros.remove(name);
                            }
                        }
                    }
                    case "include" -> {
                        if (branchActive) {
                            Matcher incMatcher = INCLUDE_TARGET_PATTERN.matcher(rest);
                            if (incMatcher.find()) {
                                String target = incMatcher.group(1).trim();
                                Optional<ResolvedInclude> resolved = includeResolver.apply(currentFilePath, target);
                                if (resolved.isPresent()) {
                                    String incPath = resolved.get().normalizedPath();
                                    if (!activeIncludeStack.contains(incPath) && activeIncludeStack.size() < MAX_INCLUDE_DEPTH) {
                                        if (!includedFiles.contains(incPath)) {
                                            includedFiles.add(incPath);
                                        }
                                        activeIncludeStack.push(incPath);
                                        processFileContent(
                                                incPath,
                                                passName,
                                                resolved.get().content(),
                                                includeResolver,
                                                macros,
                                                condStack,
                                                activeIncludeStack,
                                                includedFiles,
                                                directiveComments,
                                                properties,
                                                activeDrawBuffersHolder,
                                                output);
                                        activeIncludeStack.pop();
                                    }
                                }
                            }
                        }
                    }
                    case "version", "extension" -> {
                        if (branchActive) {
                            output.append('#').append(directive).append(' ').append(rest).append('\n');
                        }
                    }
                    default -> {
                        // Ignore #pragma, #line, #error, #warning
                    }
                }
                continue;
            }

            if (!branchActive) {
                continue;
            }

            // Active GLSL code line: expand macros first
            String expandedLine = expandMacrosInCodeLine(line, macros);

            // Record any active OptiFine const directives (e.g. shadowMapResolution, shadowDistance, sunPathRotation, *MipmapEnabled, *Format)
            properties.scanShaderDirectives(passName, expandedLine);

            // If this line is a `const int colortex0Format = R11F_G11F_B10F;` outside a comment where the RHS is a format token,
            // strip it from GLSL output so shaderc doesn't fail on undeclared format identifier.
            Matcher unquotedFormat = UNQUOTED_FORMAT_CONST_PATTERN.matcher(expandedLine);
            if (unquotedFormat.matches()) {
                String rhs = unquotedFormat.group(2);
                if (!isNumericLiteral(rhs) && !macros.containsKey(rhs)) {
                    directiveComments.add("/* " + expandedLine.trim() + " */");
                    continue;
                }
            }

            output.append(expandedLine).append('\n');
        }
    }

    private static void handleCommentBlock(
            String passName,
            String commentBlock,
            List<String> directiveComments,
            ShaderPackProperties properties,
            int[][] activeDrawBuffersHolder) {
        Optional<int[]> dbs = ShaderPackProperties.parseExplicitDrawBuffers(commentBlock);
        if (dbs.isPresent()) {
            activeDrawBuffersHolder[0] = dbs.get();
            directiveComments.add(commentBlock.trim());
        } else if (commentBlock.contains("Format") || commentBlock.contains("FORMAT")
                || commentBlock.contains("Mipmap") || commentBlock.contains("SHADOWRES")
                || commentBlock.contains("SHADOWHPL") || commentBlock.contains("shadowMapResolution")
                || commentBlock.contains("shadowDistance") || commentBlock.contains("sunPathRotation")) {
            directiveComments.add(commentBlock.trim());
        }
        properties.scanShaderDirectives(passName, commentBlock);
    }

    private static boolean isCurrentlyActive(Deque<ConditionalFrame> condStack) {
        if (condStack.isEmpty()) {
            return true;
        }
        return condStack.peek().branchActive;
    }

    private static void parseAndRegisterDefine(
            String rest,
            String trailingLineComment,
            Map<String, MacroDefinition> macros,
            ShaderPackProperties properties) {
        if (rest == null || rest.isEmpty()) {
            return;
        }
        Matcher funcMatcher = FUNC_DEFINE_PATTERN.matcher(rest);
        if (funcMatcher.matches()) {
            String name = funcMatcher.group(1).trim();
            String rawParams = funcMatcher.group(2).trim();
            String body = funcMatcher.group(3) != null ? funcMatcher.group(3).trim() : "";
            List<String> params = new ArrayList<>();
            if (!rawParams.isEmpty()) {
                for (String p : rawParams.split(",")) {
                    params.add(p.trim());
                }
            }
            macros.put(name, new MacroDefinition(name, params, body));
            return;
        }

        Matcher objMatcher = OBJ_DEFINE_PATTERN.matcher(rest);
        if (objMatcher.matches()) {
            String name = objMatcher.group(1).trim();
            String body = objMatcher.group(2) != null ? objMatcher.group(2).trim() : "";
            macros.put(name, new MacroDefinition(name, null, body));
            if (trailingLineComment != null && trailingLineComment.contains("[")) {
                properties.recordOptionDefault(name, body.isEmpty() ? "true" : body);
            }
        }
    }

    static String expandMacrosInCodeLine(String line, Map<String, MacroDefinition> macros) {
        if (line == null || line.isEmpty() || macros.isEmpty()) {
            return line;
        }
        String current = line;
        for (int pass = 0; pass < MAX_MACRO_EXPANSION_PASSES; pass++) {
            String next = expandMacrosSinglePass(current, macros, false);
            if (next.equals(current)) {
                break;
            }
            current = next;
        }
        return current;
    }

    private static String expandMacrosSinglePass(
            String text,
            Map<String, MacroDefinition> macros,
            boolean forIfExpression) {
        StringBuilder out = new StringBuilder(text.length());
        int i = 0;
        int n = text.length();

        while (i < n) {
            char c = text.charAt(i);
            if (isIdentifierStart(c)) {
                int start = i;
                i++;
                while (i < n && isIdentifierPart(text.charAt(i))) {
                    i++;
                }
                String ident = text.substring(start, i);

                if (forIfExpression && "defined".equals(ident)) {
                    out.append(ident);
                    // Copy the target of `defined(X)` or `defined X` without macro-expanding X
                    while (i < n && Character.isWhitespace(text.charAt(i))) {
                        out.append(text.charAt(i++));
                    }
                    if (i < n && text.charAt(i) == '(') {
                        out.append(text.charAt(i++));
                        while (i < n && text.charAt(i) != ')') {
                            out.append(text.charAt(i++));
                        }
                        if (i < n && text.charAt(i) == ')') {
                            out.append(text.charAt(i++));
                        }
                    } else if (i < n && isIdentifierStart(text.charAt(i))) {
                        while (i < n && isIdentifierPart(text.charAt(i))) {
                            out.append(text.charAt(i++));
                        }
                    }
                    continue;
                }

                MacroDefinition def = macros.get(ident);
                if (def == null) {
                    out.append(ident);
                    continue;
                }

                if (def.isFunctionLike()) {
                    int ws = i;
                    while (ws < n && Character.isWhitespace(text.charAt(ws))) {
                        ws++;
                    }
                    if (ws < n && text.charAt(ws) == '(') {
                        ParsedCallArgs call = parseFunctionMacroArgs(text, ws);
                        if (call != null && call.args.size() == def.params.size()) {
                            out.append(substituteFunctionMacro(def, call.args));
                            i = call.nextIndex;
                            continue;
                        }
                    }
                    out.append(ident);
                } else {
                    if (def.body.isEmpty()) {
                        if (forIfExpression) {
                            out.append("1");
                        } else {
                            out.append(ident);
                        }
                    } else if (def.body.equals(ident)) {
                        out.append(ident);
                    } else {
                        out.append(def.body);
                    }
                }
            } else {
                out.append(c);
                i++;
            }
        }
        return out.toString();
    }

    private record ParsedCallArgs(List<String> args, int nextIndex) {
    }

    private static ParsedCallArgs parseFunctionMacroArgs(String text, int openParenIdx) {
        int i = openParenIdx + 1;
        int n = text.length();
        int depth = 1;
        List<String> args = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        while (i < n && depth > 0) {
            char c = text.charAt(i);
            if (c == '(') {
                depth++;
                current.append(c);
            } else if (c == ')') {
                depth--;
                if (depth == 0) {
                    String last = current.toString().trim();
                    if (!last.isEmpty() || !args.isEmpty()) {
                        args.add(last);
                    }
                    return new ParsedCallArgs(args, i + 1);
                } else {
                    current.append(c);
                }
            } else if (c == ',' && depth == 1) {
                args.add(current.toString().trim());
                current.setLength(0);
            } else {
                current.append(c);
            }
            i++;
        }
        return null;
    }

    private static String substituteFunctionMacro(MacroDefinition def, List<String> args) {
        Map<String, String> paramMap = new LinkedHashMap<>();
        for (int idx = 0; idx < def.params.size(); idx++) {
            paramMap.put(def.params.get(idx), args.get(idx));
        }
        Matcher m = IDENTIFIER_PATTERN.matcher(def.body);
        StringBuilder sb = new StringBuilder(def.body.length() + 16);
        while (m.find()) {
            String tok = m.group(1);
            String replacement = paramMap.get(tok);
            if (replacement != null) {
                m.appendReplacement(sb, Matcher.quoteReplacement(replacement));
            } else {
                m.appendReplacement(sb, Matcher.quoteReplacement(tok));
            }
        }
        m.appendTail(sb);
        return sb.toString();
    }

    // --- Preprocessor #if / #elif Expression Evaluator ---

    public static boolean evaluateCondition(String expr, Map<String, String> simpleMacros) {
        Map<String, MacroDefinition> defs = new LinkedHashMap<>();
        if (simpleMacros != null) {
            for (Map.Entry<String, String> e : simpleMacros.entrySet()) {
                defs.put(e.getKey(), new MacroDefinition(e.getKey(), null, e.getValue()));
            }
        }
        return evaluateConditionInternal(expr, defs);
    }

    private static boolean evaluateConditionInternal(String expr, Map<String, MacroDefinition> macros) {
        if (expr == null || expr.isBlank()) {
            return false;
        }
        String current = expr;
        for (int pass = 0; pass < MAX_MACRO_EXPANSION_PASSES; pass++) {
            String next = expandMacrosSinglePass(current, macros, true);
            if (next.equals(current)) {
                break;
            }
            current = next;
        }
        try {
            ExprParser parser = new ExprParser(current, macros);
            double val = parser.parseExpression();
            return Math.abs(val) > 1e-12;
        } catch (Exception e) {
            return false;
        }
    }

    private static final class ExprParser {
        private final String s;
        private final int n;
        private final Map<String, MacroDefinition> macros;
        private int pos;

        ExprParser(String s, Map<String, MacroDefinition> macros) {
            this.s = s;
            this.n = s.length();
            this.macros = macros;
            this.pos = 0;
        }

        double parseExpression() {
            return parseTernary();
        }

        private double parseTernary() {
            double cond = parseLogicalOr();
            skipWhitespace();
            if (match('?')) {
                double trueVal = parseExpression();
                skipWhitespace();
                match(':');
                double falseVal = parseTernary();
                return (Math.abs(cond) > 1e-12) ? trueVal : falseVal;
            }
            return cond;
        }

        private double parseLogicalOr() {
            double left = parseLogicalAnd();
            while (true) {
                skipWhitespace();
                if (match("||")) {
                    double right = parseLogicalAnd();
                    left = (Math.abs(left) > 1e-12 || Math.abs(right) > 1e-12) ? 1.0 : 0.0;
                } else {
                    return left;
                }
            }
        }

        private double parseLogicalAnd() {
            double left = parseBitwiseOr();
            while (true) {
                skipWhitespace();
                if (match("&&")) {
                    double right = parseBitwiseOr();
                    left = (Math.abs(left) > 1e-12 && Math.abs(right) > 1e-12) ? 1.0 : 0.0;
                } else {
                    return left;
                }
            }
        }

        private double parseBitwiseOr() {
            double left = parseBitwiseXor();
            while (true) {
                skipWhitespace();
                if (peek('|') && !peek("||")) {
                    pos++;
                    double right = parseBitwiseXor();
                    left = (double) ((long) left | (long) right);
                } else {
                    return left;
                }
            }
        }

        private double parseBitwiseXor() {
            double left = parseBitwiseAnd();
            while (true) {
                skipWhitespace();
                if (match('^')) {
                    double right = parseBitwiseAnd();
                    left = (double) ((long) left ^ (long) right);
                } else {
                    return left;
                }
            }
        }

        private double parseBitwiseAnd() {
            double left = parseEquality();
            while (true) {
                skipWhitespace();
                if (peek('&') && !peek("&&")) {
                    pos++;
                    double right = parseEquality();
                    left = (double) ((long) left & (long) right);
                } else {
                    return left;
                }
            }
        }

        private double parseEquality() {
            double left = parseRelational();
            while (true) {
                skipWhitespace();
                if (match("==")) {
                    double right = parseRelational();
                    left = (Math.abs(left - right) <= 1e-9) ? 1.0 : 0.0;
                } else if (match("!=")) {
                    double right = parseRelational();
                    left = (Math.abs(left - right) > 1e-9) ? 1.0 : 0.0;
                } else {
                    return left;
                }
            }
        }

        private double parseRelational() {
            double left = parseShift();
            while (true) {
                skipWhitespace();
                if (match("<=")) {
                    double right = parseShift();
                    left = (left <= right) ? 1.0 : 0.0;
                } else if (match(">=")) {
                    double right = parseShift();
                    left = (left >= right) ? 1.0 : 0.0;
                } else if (peek('<') && !peek("<<")) {
                    pos++;
                    double right = parseShift();
                    left = (left < right) ? 1.0 : 0.0;
                } else if (peek('>') && !peek(">>")) {
                    pos++;
                    double right = parseShift();
                    left = (left > right) ? 1.0 : 0.0;
                } else {
                    return left;
                }
            }
        }

        private double parseShift() {
            double left = parseAdditive();
            while (true) {
                skipWhitespace();
                if (match("<<")) {
                    double right = parseAdditive();
                    left = (double) ((long) left << (long) right);
                } else if (match(">>")) {
                    double right = parseAdditive();
                    left = (double) ((long) left >> (long) right);
                } else {
                    return left;
                }
            }
        }

        private double parseAdditive() {
            double left = parseMultiplicative();
            while (true) {
                skipWhitespace();
                if (match('+')) {
                    left += parseMultiplicative();
                } else if (match('-')) {
                    left -= parseMultiplicative();
                } else {
                    return left;
                }
            }
        }

        private double parseMultiplicative() {
            double left = parseUnary();
            while (true) {
                skipWhitespace();
                if (match('*')) {
                    left *= parseUnary();
                } else if (match('/')) {
                    double divisor = parseUnary();
                    left = (Math.abs(divisor) > 1e-12) ? (left / divisor) : 0.0;
                } else if (match('%')) {
                    double divisor = parseUnary();
                    left = (Math.abs(divisor) > 1e-12) ? ((long) left % (long) divisor) : 0.0;
                } else {
                    return left;
                }
            }
        }

        private double parseUnary() {
            skipWhitespace();
            if (match('!')) {
                double v = parseUnary();
                return (Math.abs(v) <= 1e-12) ? 1.0 : 0.0;
            }
            if (match('+')) {
                return parseUnary();
            }
            if (match('-')) {
                return -parseUnary();
            }
            if (match('~')) {
                return (double) (~((long) parseUnary()));
            }
            return parsePrimary();
        }

        private double parsePrimary() {
            skipWhitespace();
            if (match('(')) {
                double val = parseExpression();
                skipWhitespace();
                match(')');
                return val;
            }
            if (pos < n && (Character.isDigit(s.charAt(pos)) || s.charAt(pos) == '.')) {
                return parseNumber();
            }
            if (pos < n && isIdentifierStart(s.charAt(pos))) {
                String ident = readIdentifier();
                if ("defined".equals(ident)) {
                    skipWhitespace();
                    boolean hasParen = match('(');
                    skipWhitespace();
                    String target = readIdentifier();
                    if (hasParen) {
                        skipWhitespace();
                        match(')');
                    }
                    return (!target.isEmpty() && macros.containsKey(target)) ? 1.0 : 0.0;
                }
                if ("true".equalsIgnoreCase(ident)) {
                    return 1.0;
                }
                if ("false".equalsIgnoreCase(ident)) {
                    return 0.0;
                }
                // Unknown/undefined identifiers evaluate to 0 in GLSL #if expressions
                return 0.0;
            }
            return 0.0;
        }

        private double parseNumber() {
            int start = pos;
            if (pos + 1 < n && s.charAt(pos) == '0' && (s.charAt(pos + 1) == 'x' || s.charAt(pos + 1) == 'X')) {
                pos += 2;
                int hexStart = pos;
                while (pos < n && isHexDigit(s.charAt(pos))) {
                    pos++;
                }
                while (pos < n && (s.charAt(pos) == 'u' || s.charAt(pos) == 'U' || s.charAt(pos) == 'l' || s.charAt(pos) == 'L')) {
                    pos++;
                }
                String hex = s.substring(hexStart, pos).replaceAll("[uUlL]", "");
                return hex.isEmpty() ? 0.0 : (double) Long.parseUnsignedLong(hex, 16);
            }
            while (pos < n && (Character.isDigit(s.charAt(pos)) || s.charAt(pos) == '.')) {
                pos++;
            }
            if (pos < n && (s.charAt(pos) == 'e' || s.charAt(pos) == 'E')) {
                pos++;
                if (pos < n && (s.charAt(pos) == '+' || s.charAt(pos) == '-')) {
                    pos++;
                }
                while (pos < n && Character.isDigit(s.charAt(pos))) {
                    pos++;
                }
            }
            String numStr = s.substring(start, pos);
            while (pos < n && Character.isLetter(s.charAt(pos))) {
                pos++;
            }
            try {
                return Double.parseDouble(numStr);
            } catch (NumberFormatException e) {
                return 0.0;
            }
        }

        private String readIdentifier() {
            skipWhitespace();
            int start = pos;
            if (pos < n && isIdentifierStart(s.charAt(pos))) {
                pos++;
                while (pos < n && isIdentifierPart(s.charAt(pos))) {
                    pos++;
                }
            }
            return s.substring(start, pos);
        }

        private void skipWhitespace() {
            while (pos < n && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
        }

        private boolean match(char expected) {
            if (pos < n && s.charAt(pos) == expected) {
                pos++;
                return true;
            }
            return false;
        }

        private boolean match(String expected) {
            if (peek(expected)) {
                pos += expected.length();
                return true;
            }
            return false;
        }

        private boolean peek(char expected) {
            return pos < n && s.charAt(pos) == expected;
        }

        private boolean peek(String expected) {
            return s.startsWith(expected, pos);
        }
    }

    private static List<String> joinBackslashLines(String text) {
        String[] raw = text.replace("\r\n", "\n").replace('\r', '\n').split("\n", -1);
        List<String> out = new ArrayList<>(raw.length);
        StringBuilder current = new StringBuilder();
        for (String line : raw) {
            String stripped = line.stripTrailing();
            if (stripped.endsWith("\\")) {
                current.append(stripped, 0, stripped.length() - 1).append(' ');
            } else {
                if (current.length() > 0) {
                    current.append(line);
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

    private static String extractFirstToken(String s) {
        if (s == null || s.isBlank()) {
            return "";
        }
        String trimmed = s.trim();
        int i = 0;
        while (i < trimmed.length() && isIdentifierPart(trimmed.charAt(i))) {
            i++;
        }
        return trimmed.substring(0, i);
    }

    private static String derivePassName(String path) {
        if (path == null) {
            return "";
        }
        int slash = path.lastIndexOf('/');
        String file = slash >= 0 ? path.substring(slash + 1) : path;
        int dot = file.lastIndexOf('.');
        return dot > 0 ? file.substring(0, dot) : file;
    }

    private static boolean isIdentifierStart(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }

    private static boolean isIdentifierPart(char c) {
        return isIdentifierStart(c) || (c >= '0' && c <= '9');
    }

    private static boolean isHexDigit(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }

    private static boolean isNumericLiteral(String token) {
        if (token == null || token.isEmpty()) {
            return false;
        }
        char first = token.charAt(0);
        return (first >= '0' && first <= '9') || first == '+' || first == '-';
    }
}
