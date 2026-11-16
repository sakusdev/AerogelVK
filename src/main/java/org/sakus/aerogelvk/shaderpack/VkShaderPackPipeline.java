package org.sakus.aerogelvk.shaderpack;

import org.sakus.aerogelvk.meshing.AtlasUVTable;
import org.sakus.aerogelvk.shaderpack.GlslToVulkanTranspiler.PassKind;
import org.sakus.aerogelvk.shaderpack.GlslToVulkanTranspiler.TranspiledProgram;
import org.sakus.aerogelvk.shaderpack.ShaderPackManager.PassProgram;
import org.sakus.aerogelvk.shaderpack.ShaderPackManager.ShaderPackPrograms;
import org.sakus.aerogelvk.vulkan.VkContext;
import org.sakus.aerogelvk.vulkan.VkGpuProfiler;
import org.sakus.aerogelvk.vulkan.VkMegaBuffer;
import org.sakus.aerogelvk.vulkan.VkPipelineManager;
import org.sakus.aerogelvk.vulkan.VkShaderCompiler;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.vulkan.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.*;

import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK12.vkCmdDrawIndexedIndirectCount;
import static org.lwjgl.vulkan.VK13.*;

/**
 * Vulkan 1.3 dynamic-rendering execution pipeline for OptiFine/Iris shaderpacks.
 *
 * <p>Consumes {@link ShaderPackManager}, {@link GlslToVulkanTranspiler}, {@link ShaderUniformLayout},
 * {@link ShaderDescriptorBindings}, {@link VkShaderCompiler}, and {@link VkShaderPackTargets} to:
 * <ul>
 *   <li>Compile loaded shaderpack passes ({@code shadow}, {@code gbuffers_terrain}, {@code gbuffers_water},
 *       {@code deferred*}, {@code composite*}, {@code final}) into Vulkan 1.3 dynamic-rendering
 *       {@code VkPipeline}s, logging per-pass warnings and skipping broken optional sub-passes gracefully.</li>
 *   <li>Update the 1024-byte {@code ShaderPackUBO} at Set 0 Binding 0 every frame from
 *       {@link ShaderUniformLayout.UniformData} or {@link FrameParams}.</li>
 *   <li>Record the complete multi-pass Vulkan frame execution ({@code chunk_cull} &rarr; {@code shadow} &rarr;
 *       {@code gbuffers_terrain} &rarr; {@code gbuffers_water} &rarr; {@code deferred*} &rarr; {@code composite*}
 *       &rarr; {@code final} writing directly into the shared interop color/depth images).</li>
 * </ul>
 */
public class VkShaderPackPipeline implements AutoCloseable {
    static {
        if (System.getProperty("org.lwjgl.system.stackSize") == null) {
            System.setProperty("org.lwjgl.system.stackSize", "512");
        }
    }

    private static final Logger LOGGER = LoggerFactory.getLogger(VkShaderPackPipeline.class);

    private static final int MAX_PASS_DESCRIPTOR_SETS = 48;
    private static final int MAX_COLOR_ATTACHMENTS_PER_PASS = 8;

    private static final String SHADERPACK_CULL_COMP_SRC =
        "#version 460\n" +
        "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n" +
        ShaderUniformLayout.GLSL_UBO_BLOCK + "\n" +
        "struct ChunkDrawMeta {\n" +
        "    ivec3 chunkPos;\n" +
        "    uint vertexOffset;\n" +
        "    uint quadCount;\n" +
        "    uint aabbPacked;\n" +
        "    uint flags;\n" +
        "    uint padding;\n" +
        "};\n" +
        "layout(std430, set = 0, binding = 2) readonly buffer ChunkMetaSSBO {\n" +
        "    ChunkDrawMeta chunks[];\n" +
        "};\n" +
        "struct VkDrawIndexedIndirectCommand {\n" +
        "    uint indexCount;\n" +
        "    uint instanceCount;\n" +
        "    uint firstIndex;\n" +
        "    int  vertexOffset;\n" +
        "    uint firstInstance;\n" +
        "};\n" +
        "layout(std430, set = 0, binding = 3) buffer IndirectCmdSSBO {\n" +
        "    VkDrawIndexedIndirectCommand cmds[];\n" +
        "};\n" +
        "layout(std430, set = 0, binding = 4) buffer DrawCountSSBO {\n" +
        "    uint visibleChunkCount;\n" +
        "};\n" +
        "layout(std430, set = 0, binding = 5) buffer VisibleChunkIdSSBO {\n" +
        "    uint instanceToChunkIdx[];\n" +
        "};\n" +
        "layout(push_constant) uniform CullPushConstants {\n" +
        "    uint totalChunkCount;\n" +
        "    uint passFlags;\n" +
        "} pc;\n" +
        "void main() {\n" +
        "    uint chunkIdx = gl_GlobalInvocationID.x;\n" +
        "    if (chunkIdx >= pc.totalChunkCount) {\n" +
        "        return;\n" +
        "    }\n" +
        "    ChunkDrawMeta meta = chunks[chunkIdx];\n" +
        "    if (meta.quadCount == 0u) {\n" +
        "        return;\n" +
        "    }\n" +
        "    if (pc.passFlags == 1u && (meta.flags & 1u) != 0u) {\n" +
        "        return;\n" +
        "    }\n" +
        "    if (pc.passFlags == 2u && (meta.flags & 1u) == 0u) {\n" +
        "        return;\n" +
        "    }\n" +
        "    uint cmdIdx = atomicAdd(visibleChunkCount, 1u);\n" +
        "    cmds[cmdIdx].indexCount = meta.quadCount * 6u;\n" +
        "    cmds[cmdIdx].instanceCount = 1u;\n" +
        "    cmds[cmdIdx].firstIndex = 0u;\n" +
        "    cmds[cmdIdx].vertexOffset = int(meta.vertexOffset);\n" +
        "    cmds[cmdIdx].firstInstance = cmdIdx;\n" +
        "    instanceToChunkIdx[cmdIdx] = chunkIdx;\n" +
        "}\n";

    private static final String FALLBACK_TERRAIN_VSH =
        "#version 120\n" +
        "varying vec2 texcoord;\n" +
        "varying vec4 color;\n" +
        "void main() {\n" +
        "    gl_Position = ftransform();\n" +
        "    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;\n" +
        "    color = gl_Color;\n" +
        "}\n";

    private static final String FALLBACK_TERRAIN_FSH =
        "#version 120\n" +
        "uniform sampler2D texture;\n" +
        "varying vec2 texcoord;\n" +
        "varying vec4 color;\n" +
        "/* DRAWBUFFERS:0 */\n" +
        "void main() {\n" +
        "    vec4 tex = texture2D(texture, texcoord) * color;\n" +
        "    if (tex.a < 0.1) discard;\n" +
        "    gl_FragData[0] = tex;\n" +
        "}\n";

    private static final String FALLBACK_FINAL_VSH =
        "#version 120\n" +
        "varying vec2 texcoord;\n" +
        "void main() {\n" +
        "    gl_Position = ftransform();\n" +
        "    texcoord = gl_MultiTexCoord0.xy;\n" +
        "}\n";

    private static final String FALLBACK_FINAL_FSH =
        "#version 120\n" +
        "uniform sampler2D colortex0;\n" +
        "varying vec2 texcoord;\n" +
        "void main() {\n" +
        "    gl_FragColor = texture2D(colortex0, texcoord);\n" +
        "}\n";

    /**
     * Convenience per-frame parameter carrier for callers that do not construct
     * {@link ShaderUniformLayout.UniformData} directly.
     */
    public static final class FrameParams {
        public final Matrix4f viewMatrix = new Matrix4f().identity();
        public final Matrix4f projMatrix = new Matrix4f().identity();
        public final Vector3f cameraPos = new Vector3f();
        public final Vector3f sunDirection = new Vector3f(0.4f, 0.85f, 0.35f).normalize();
        public final Vector3f skyColor = new Vector3f(0.48f, 0.68f, 0.95f);
        public final Vector3f fogColor = new Vector3f(0.70f, 0.82f, 0.98f);
        public float nearPlane = 0.05f;
        public float farPlane = 256.0f;
        public int worldTime = 6000;
        public int frameCounter = 0;
        public float frameTime = 1.0f / 60.0f;
        public float frameTimeCounter = 0.0f;
        public float sunAngle = 0.25f;
        public float rainStrength = 0.0f;
        public float wetness = 0.0f;
        public int isEyeInWater = 0;
    }

    /**
     * Compiled Vulkan pipeline for an individual shaderpack pass.
     */
    public static final class CompiledPass {
        private final String name;
        private final PassKind kind;
        private final long pipeline;
        private final int[] drawBuffers;
        private final int colorAttachmentCount;
        private final boolean depthTest;
        private final boolean depthWrite;
        private final boolean alphaBlend;

        public CompiledPass(
                String name,
                PassKind kind,
                long pipeline,
                int[] drawBuffers,
                int colorAttachmentCount,
                boolean depthTest,
                boolean depthWrite,
                boolean alphaBlend) {
            this.name = name;
            this.kind = kind;
            this.pipeline = pipeline;
            this.drawBuffers = drawBuffers.clone();
            this.colorAttachmentCount = colorAttachmentCount;
            this.depthTest = depthTest;
            this.depthWrite = depthWrite;
            this.alphaBlend = alphaBlend;
        }

        public String getName() {
            return name;
        }

        public PassKind getKind() {
            return kind;
        }

        public long getPipeline() {
            return pipeline;
        }

        public int[] getDrawBuffers() {
            return drawBuffers.clone();
        }

        public int getColorAttachmentCount() {
            return colorAttachmentCount;
        }

        public boolean isDepthTest() {
            return depthTest;
        }

        public boolean isDepthWrite() {
            return depthWrite;
        }

        public boolean isAlphaBlend() {
            return alphaBlend;
        }
    }

    private final VkContext ctx;
    private final VkMegaBuffer megaBuffer;
    private final VkShaderPackTargets targets;

    private long descriptorSetLayout0 = VK_NULL_HANDLE;
    private long descriptorSetLayout1 = VK_NULL_HANDLE;
    private long pipelineLayout = VK_NULL_HANDLE;
    private long descriptorPool = VK_NULL_HANDLE;
    private long descriptorSet0 = VK_NULL_HANDLE;
    private final long[] descriptorSets1 = new long[MAX_PASS_DESCRIPTOR_SETS];

    // ShaderPackUBO (Set 0 Binding 0, 1024 bytes, host-visible coherent)
    private long uboBuffer = VK_NULL_HANDLE;
    private long uboAllocation = VK_NULL_HANDLE;
    private long uboMappedPtr = MemoryUtil.NULL;

    // Optional explicit block atlas texture view & sampler (falls back to VkPipelineManager or targets.getWhiteView())
    private long blockAtlasImageView = VK_NULL_HANDLE;
    private long blockAtlasSampler = VK_NULL_HANDLE;

    // Self-contained compute culling pipeline for shaderpack frames
    private long cullPipeline = VK_NULL_HANDLE;

    // Compiled shaderpack passes
    private CompiledPass shadowPass;
    private CompiledPass gbuffersTerrainPass;
    private CompiledPass gbuffersWaterPass;
    private final List<CompiledPass> preparePasses = new ArrayList<>();
    private final List<CompiledPass> deferredPasses = new ArrayList<>();
    private final List<CompiledPass> compositePasses = new ArrayList<>();
    private CompiledPass finalPass;
    private final Map<String, CompiledPass> compiledPassesByName = new LinkedHashMap<>();
    private final List<String> compileWarnings = new ArrayList<>();

    // Previous-frame matrix state for temporal passes
    private final Matrix4f prevModelView = new Matrix4f().identity();
    private final Matrix4f prevProjection = new Matrix4f().identity();
    private final Vector3f prevCameraPos = new Vector3f();
    private boolean hasPrevFrameState = false;
    private int internalFrameCounter = 0;

    // Clear color from latest uniform update (used by recordClearAllTargets)
    private float frameClearR = 0.48f;
    private float frameClearG = 0.68f;
    private float frameClearB = 0.95f;

    private int finalColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    private int displayWidth = 1;
    private int displayHeight = 1;
    private float activeRenderScale = 1.0f;
    private int activeShadowCap = 0;
    private String activePackName = "OFF";
    private ShaderPackProperties activeProperties = ShaderPackProperties.empty();
    private boolean initialized = false;
    private boolean packReady = false;
    private PassTraceListener passTraceListener;
    private int passTraceColorIndex = 0;
    private VkGpuProfiler gpuProfiler;
    private int profilerFrameSlot = 0;

    // Static-frame shadow skip tracking
    private static final float SHADOW_POS_EPSILON = 1e-4f;
    private static final float SHADOW_MAT_EPSILON = 1e-4f;
    private boolean shadowMapValid = false;
    private boolean shadowSceneDirty = true;
    private int packLoadGeneration = 0;
    private final Vector3f prevShadowCamPos = new Vector3f();
    private final Matrix4f prevShadowModelView = new Matrix4f();
    private final Matrix4f prevShadowProj = new Matrix4f();
    private final Vector3f prevShadowSunDir = new Vector3f();
    private final Vector3f prevShadowLightPos = new Vector3f();
    private int prevShadowWorldTime = Integer.MIN_VALUE;
    private int prevShadowWorldDay = Integer.MIN_VALUE;
    private int prevShadowMoonPhase = Integer.MIN_VALUE;
    private float prevShadowAngle = Float.NaN;
    private float prevShadowSunAngle = Float.NaN;
    private float prevShadowRainStrength = Float.NaN;
    private int prevShadowResolution = -1;
    private int prevShadowCap = -1;
    private int prevShadowPackGeneration = -1;

    public void invalidateShadow() {
        this.shadowSceneDirty = true;
        this.shadowMapValid = false;
    }

    public void setGpuProfiler(VkGpuProfiler gpuProfiler, int frameSlot) {
        this.gpuProfiler = gpuProfiler;
        this.profilerFrameSlot = frameSlot;
    }

    @FunctionalInterface
    public interface PassTraceListener {
        void onPassRecorded(VkCommandBuffer cmd, MemoryStack stack, String passName, long colortex0Image, int colortex0Format);
    }

    public void setPassTraceListener(PassTraceListener listener) {
        this.passTraceListener = listener;
    }

    public void setPassTraceColorIndex(int index) {
        this.passTraceColorIndex = Math.max(0, index);
    }

    private void notifyPassTrace(VkCommandBuffer cmd, MemoryStack stack, String passName) {
        PassTraceListener listener = this.passTraceListener;
        if (listener != null && targets != null) {
            listener.onPassRecorded(cmd, stack, passName, targets.getColorReadImage(passTraceColorIndex), targets.getColorFormat(passTraceColorIndex));
        }
    }

    public VkShaderPackPipeline() {
        this(VkContext.getInstance(), VkMegaBuffer.getInstance(), new VkShaderPackTargets());
    }

    public VkShaderPackPipeline(VkContext ctx, VkMegaBuffer megaBuffer) {
        this(ctx, megaBuffer, new VkShaderPackTargets());
    }

    public VkShaderPackPipeline(VkContext ctx, VkMegaBuffer megaBuffer, VkShaderPackTargets targets) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.megaBuffer = Objects.requireNonNull(megaBuffer, "megaBuffer");
        this.targets = Objects.requireNonNull(targets, "targets");
    }

    /**
     * Initializes descriptor layouts, descriptor sets, UBO buffer, offscreen targets, and internal cull pipeline.
     */
    public synchronized void init(int width, int height) {
        init(width, height, VK_FORMAT_R8G8B8A8_UNORM);
    }

    /**
     * Initializes descriptor layouts, descriptor sets, UBO buffer, offscreen targets, and internal cull pipeline
     * with the specified final output color format.
     */
    public synchronized void init(int width, int height, int finalColorFormat) {
        if (initialized) {
            return;
        }
        this.finalColorFormat = finalColorFormat;

        if (!ctx.isInitialized()) {
            ctx.init();
        }
        if (!megaBuffer.isInitialized()) {
            megaBuffer.init();
        }
        if (!targets.isInitialized()) {
            this.displayWidth = Math.max(1, width);
            this.displayHeight = Math.max(1, height);
            this.activeRenderScale = org.sakus.aerogelvk.config.ValkyrieConfig.get().getShaderPackRenderScale();
            this.activeShadowCap = org.sakus.aerogelvk.config.ValkyrieConfig.get().getShaderPackShadowResolutionCap();
            int scaledW = computeScaledDim(this.displayWidth, this.activeRenderScale);
            int scaledH = computeScaledDim(this.displayHeight, this.activeRenderScale);
            targets.init(scaledW, scaledH, ShaderPackProperties.empty());
        }

        createUboBuffer();
        ensureAtlasUvPopulated();
        createDescriptorLayoutsAndPool();
        updateDescriptorSet0();
        createCullPipeline();

        initialized = true;
        LOGGER.info("[Valkyrie] VkShaderPackPipeline initialized ({}x{}, finalFormat={}).",
                targets.getWidth(), targets.getHeight(), finalColorFormat);
    }

    /**
     * Resolves and compiles a named shaderpack from {@code gameDir} (e.g. {@code "auto"}, {@code "OFF"}, or zip/folder name).
     *
     * @return true if a shaderpack was loaded and compiled successfully
     */
    public synchronized boolean loadShaderPack(Path gameDir, String packName, int width, int height) {
        if (packName == null || packName.isBlank() || "off".equalsIgnoreCase(packName.trim())
                || "none".equalsIgnoreCase(packName.trim())) {
            unloadPack();
            return false;
        }
        try {
            Optional<ShaderPackPrograms> loaded = ShaderPackManager.loadShaderPack(gameDir, packName);
            if (loaded.isEmpty()) {
                LOGGER.warn("[Valkyrie] Shaderpack '{}' not found in {}", packName, gameDir);
                unloadPack();
                return false;
            }
            return compilePack(loaded.get(), width, height, finalColorFormat);
        } catch (IOException e) {
            LOGGER.warn("[Valkyrie] Failed to load shaderpack '{}' from {}: {}", packName, gameDir, e.getMessage());
            unloadPack();
            return false;
        }
    }

    /**
     * Loads and compiles a shaderpack directly from a {@code .zip} or folder {@link Path}.
     */
    public synchronized boolean loadShaderPack(Path packPath, int width, int height) {
        if (packPath == null) {
            unloadPack();
            return false;
        }
        try {
            ShaderPackPrograms programs = ShaderPackManager.compileShaderPack(packPath);
            return compilePack(programs, width, height, finalColorFormat);
        } catch (IOException e) {
            LOGGER.warn("[Valkyrie] Failed to load shaderpack from {}: {}", packPath, e.getMessage());
            unloadPack();
            return false;
        }
    }

    /**
     * Compiles all passes in {@link ShaderPackPrograms} into Vulkan 1.3 dynamic-rendering pipelines
     * and configures {@link VkShaderPackTargets} according to {@link ShaderPackProperties}.
     *
     * @return true if compilation succeeded for the core render chain
     */
    public synchronized boolean compilePack(ShaderPackPrograms programs, int width, int height) {
        return compilePack(programs, width, height, this.finalColorFormat);
    }

    /**
     * Compiles all passes in {@link ShaderPackPrograms} into Vulkan 1.3 dynamic-rendering pipelines
     * with an explicit final target color format.
     */
    public synchronized boolean compilePack(ShaderPackPrograms programs, int width, int height, int finalTargetFormat) {
        Objects.requireNonNull(programs, "programs");
        this.finalColorFormat = finalTargetFormat;
        if (!initialized) {
            init(width, height, finalTargetFormat);
        }

        vkDeviceWaitIdle(ctx.getVkDevice());
        destroyPassPipelines();
        compileWarnings.clear();

        this.activeProperties = programs.getProperties() != null
                ? programs.getProperties()
                : ShaderPackProperties.empty();
        this.displayWidth = Math.max(1, width);
        this.displayHeight = Math.max(1, height);
        this.activeRenderScale = org.sakus.aerogelvk.config.ValkyrieConfig.get().getShaderPackRenderScale();
        this.activeShadowCap = org.sakus.aerogelvk.config.ValkyrieConfig.get().getShaderPackShadowResolutionCap();
        int scaledW = computeScaledDim(this.displayWidth, this.activeRenderScale);
        int scaledH = computeScaledDim(this.displayHeight, this.activeRenderScale);
        targets.reconfigure(scaledW, scaledH, activeProperties);
        updateDescriptorSet0();

        activePackName = programs.getPackName() != null ? programs.getPackName() : "shaderpack";

        // 1. Optional shadow pass
        if (programs.hasProgram("shadow")) {
            PassProgram p = programs.getProgram("shadow");
            shadowPass = compilePassProgram(p, false);
            if (shadowPass != null) {
                compiledPassesByName.put(shadowPass.getName(), shadowPass);
            }
        }

        // 2. Primary gbuffers_terrain pass (falls back to gbuffers_basic or built-in terrain pass if missing/broken)
        PassProgram terrainProg = programs.getGbuffersTerrain();
        if (terrainProg == null) {
            terrainProg = programs.getGbuffersBasic();
        }
        if (terrainProg != null) {
            gbuffersTerrainPass = compilePassProgram(terrainProg, false);
        }
        if (gbuffersTerrainPass == null) {
            gbuffersTerrainPass = compileFallbackTerrainPass();
        }
        if (gbuffersTerrainPass != null) {
            compiledPassesByName.put("gbuffers_terrain", gbuffersTerrainPass);
        }

        // 3. Optional gbuffers_water pass (only if distinct or explicitly present)
        if (programs.hasProgram("gbuffers_water")) {
            PassProgram waterProg = programs.getGbuffersWater();
            if (waterProg != null && (terrainProg == null
                    || !Objects.equals(waterProg.getResolvedFrom(), terrainProg.getResolvedFrom()))) {
                gbuffersWaterPass = compilePassProgram(waterProg, true);
                if (gbuffersWaterPass != null) {
                    compiledPassesByName.put("gbuffers_water", gbuffersWaterPass);
                }
            }
        }

        // 0. Prepare passes (prepare, prepare1..15)
        for (PassProgram prepProg : programs.getPreparePasses()) {
            CompiledPass cp = compilePassProgram(prepProg, false);
            if (cp != null) {
                preparePasses.add(cp);
                compiledPassesByName.put(cp.getName(), cp);
            }
        }

        // 4. Deferred passes (deferred, deferred1..deferred15)
        for (PassProgram deferredProg : programs.getDeferredPasses()) {
            CompiledPass cp = compilePassProgram(deferredProg, false);
            if (cp != null) {
                deferredPasses.add(cp);
                compiledPassesByName.put(cp.getName(), cp);
            }
        }

        // 5. Composite passes (composite, composite1..composite15)
        for (PassProgram compositeProg : programs.getCompositePasses()) {
            CompiledPass cp = compilePassProgram(compositeProg, false);
            if (cp != null) {
                compositePasses.add(cp);
                compiledPassesByName.put(cp.getName(), cp);
            }
        }

        // 6. Final pass (falls back to built-in colortex0 blitter if omitted or broken)
        if (programs.hasProgram("final")) {
            finalPass = compilePassProgram(programs.getFinalPass(), false);
        }
        if (finalPass == null) {
            finalPass = compileFallbackFinalPass();
        }
        if (finalPass != null) {
            compiledPassesByName.put("final", finalPass);
        }

        packReady = (gbuffersTerrainPass != null && finalPass != null);
        hasPrevFrameState = false;
        packLoadGeneration++;
        invalidateShadow();

        LOGGER.info("[Valkyrie] Compiled shaderpack '{}': ready={}, passes={} (shadow={}, terrain={}, water={}, deferred={}, composite={}, final={}), warnings={}",
                activePackName,
                packReady,
                compiledPassesByName.size(),
                shadowPass != null,
                gbuffersTerrainPass != null,
                gbuffersWaterPass != null,
                deferredPasses.size(),
                compositePasses.size(),
                finalPass != null,
                compileWarnings.size());

        return packReady;
    }

    /**
     * Unloads the active shaderpack pipelines and marks the shaderpack pipeline inactive.
     */
    public synchronized void unloadPack() {
        if (initialized && ctx.isInitialized()) {
            vkDeviceWaitIdle(ctx.getVkDevice());
        }
        destroyPassPipelines();
        activePackName = "OFF";
        packLoadGeneration++;
        invalidateShadow();
        packReady = false;
        hasPrevFrameState = false;
    }

    /**
     * Resizes all viewport-sized offscreen targets ({@code colortex0..7}, {@code depthtex0..2}).
     */
    public synchronized void resize(int width, int height) {
        if (!initialized) {
            init(width, height, finalColorFormat);
            return;
        }
        this.displayWidth = Math.max(1, width);
        this.displayHeight = Math.max(1, height);
        this.activeRenderScale = org.sakus.aerogelvk.config.ValkyrieConfig.get().getShaderPackRenderScale();
        int scaledW = computeScaledDim(this.displayWidth, this.activeRenderScale);
        int scaledH = computeScaledDim(this.displayHeight, this.activeRenderScale);
        targets.resize(scaledW, scaledH);
    }

    private static int computeScaledDim(int dim, float scale) {
        float clamped = Math.max(0.5f, Math.min(1.0f, Float.isFinite(scale) ? scale : 1.0f));
        return Math.max(1, Math.round(dim * clamped));
    }

    /**
     * Updates the block atlas texture (Set 0 Binding 7 and Set 1 Binding 15) and AtlasUV SSBO (Set 0 Binding 6).
     */
    public synchronized void updateBlockAtlas(long atlasImageView, long atlasSampler, AtlasUVTable uvTable) {
        this.blockAtlasImageView = atlasImageView;
        this.blockAtlasSampler = atlasSampler;

        if (uvTable != null && uvTable.getSpriteCount() > 0) {
            ByteBuffer uvBuf = uvTable.getUvTableBuffer();
            if (uvBuf != null) {
                megaBuffer.uploadAtlasUvTable(uvBuf);
            }
        } else {
            ensureAtlasUvPopulated();
        }

        if (initialized) {
            updateDescriptorSet0();
        }
    }

    /**
     * Builds a normalized {@link ShaderUniformLayout.UniformData} from {@link FrameParams}.
     */
    public ShaderUniformLayout.UniformData buildUniformData(FrameParams params, int width, int height) {
        ShaderUniformLayout.UniformData u = new ShaderUniformLayout.UniformData();
        if (params != null) {
            u.cameraPosition.set(params.cameraPos);
            u.gbufferModelView.set(params.viewMatrix);
            u.gbufferProjection.set(params.projMatrix);
            u.skyColor.set(params.skyColor);
            u.fogColor.set(params.fogColor);
            u.near = params.nearPlane > 0.0f ? params.nearPlane : 0.05f;
            u.far = params.farPlane > u.near ? params.farPlane : 256.0f;
            u.worldTime = ((params.worldTime % 24000) + 24000) % 24000;
            u.frameCounter = params.frameCounter > 0 ? params.frameCounter : internalFrameCounter;
            u.frameTime = params.frameTime > 0.0f ? params.frameTime : (1.0f / 60.0f);
            u.frameTimeCounter = params.frameTimeCounter > 0.0f ? params.frameTimeCounter : (u.frameCounter * u.frameTime);
            u.sunAngle = params.sunAngle;
            u.rainStrength = params.rainStrength;
            u.wetness = params.wetness;
            u.isEyeInWater = params.isEyeInWater;

            Vector3f sunWorldDir = new Vector3f(params.sunDirection);
            if (sunWorldDir.lengthSquared() > 1e-6f) {
                sunWorldDir.normalize();
            } else {
                sunWorldDir.set(computeSunWorldDir(u.sunAngle));
            }
            populateShadowAndCelestialVectors(u, sunWorldDir, false);
        }
        u.viewWidth = Math.max(1.0f, (float) width);
        u.viewHeight = Math.max(1.0f, (float) height);
        u.aspectRatio = u.viewWidth / u.viewHeight;
        return u;
    }

    /**
     * Updates the 1024-byte {@code ShaderPackUBO} from {@link FrameParams}.
     */
    public void updateUniforms(FrameParams params, int width, int height) {
        ShaderUniformLayout.UniformData u = buildUniformData(params, width, height);
        updateUniforms(u, width, height);
    }

    /**
     * Updates the 1024-byte {@code ShaderPackUBO} from {@link ShaderUniformLayout.UniformData}.
     *
     * <p>Automatically normalizes {@code gbufferModelView} to camera-relative rotation-only
     * (matching {@link GlslToVulkanTranspiler}'s {@code worldPos - cameraPosition} vertex pulling),
     * converts {@code gbufferProjection} to Vulkan Standard-Z {@code [0, 1]} while preserving the
     * OpenGL {@code [-1, 1]} inverse in {@code gbufferProjectionInverse}, and maintains temporal
     * previous-frame matrices.</p>
     */
    public void updateUniforms(ShaderUniformLayout.UniformData input, int width, int height) {
        if (uboMappedPtr == MemoryUtil.NULL) {
            return;
        }
        ShaderUniformLayout.UniformData u = (input != null) ? input : new ShaderUniformLayout.UniformData();

        u.viewWidth = Math.max(1.0f, (float) width);
        u.viewHeight = Math.max(1.0f, (float) height);
        u.aspectRatio = u.viewWidth / u.viewHeight;
        if (u.near <= 0.0f) {
            u.near = 0.05f;
        }
        if (u.far <= u.near) {
            u.far = 256.0f;
        }

        // GlslToVulkanTranspiler computes vk_Vertex = vec4(worldPos - cameraPosition, 1.0),
        // so gbufferModelView must be rotation-only (no camera translation).
        u.gbufferModelView.m30(0.0f).m31(0.0f).m32(0.0f).m33(1.0f);
        u.gbufferModelViewInverse = new Matrix4f(u.gbufferModelView).invert();

        // Convert projection to OpenGL [-1, 1] for gbufferProjectionInverse and Vulkan [0, 1] for gbufferProjection
        Matrix4f glClipProj = toOpenGlClipProjection(u.gbufferProjection, u.near, u.far);
        u.gbufferProjectionInverse = new Matrix4f(glClipProj).invert();
        u.gbufferProjection.set(openGlToVulkanStandardZ(glClipProj));

        if (hasPrevFrameState) {
            u.gbufferPreviousModelView.set(prevModelView);
            u.gbufferPreviousProjection.set(prevProjection);
            u.previousCameraPosition.set(prevCameraPos);
        } else {
            u.gbufferPreviousModelView.set(u.gbufferModelView);
            u.gbufferPreviousProjection.set(u.gbufferProjection);
            u.previousCameraPosition.set(u.cameraPosition);
        }

        // Ensure shadow matrices are camera-relative and valid
        u.shadowModelView.m30(0.0f).m31(0.0f).m32(0.0f).m33(1.0f);
        if (Math.abs(u.shadowProjection.m00() - 1.0f) < 1e-5f && Math.abs(u.shadowProjection.m11() - 1.0f) < 1e-5f) {
            boolean hasCallerSunPos = u.sunPosition.lengthSquared() > 1e-4f;
            Vector3f sunWorldDir;
            if (hasCallerSunPos) {
                Vector4f worldSun = u.gbufferModelViewInverse.transform(
                        new Vector4f(u.sunPosition.x, u.sunPosition.y, u.sunPosition.z, 0.0f));
                sunWorldDir = new Vector3f(worldSun.x, worldSun.y, worldSun.z);
                if (sunWorldDir.lengthSquared() > 1e-6f) {
                    sunWorldDir.normalize();
                } else {
                    sunWorldDir.set(computeSunWorldDir(u.sunAngle));
                }
            } else {
                sunWorldDir = computeSunWorldDir(u.sunAngle);
            }
            populateShadowAndCelestialVectors(u, sunWorldDir, hasCallerSunPos);
        } else {
            u.shadowModelViewInverse = new Matrix4f(u.shadowModelView).invert();
            u.shadowProjectionInverse = new Matrix4f(u.shadowProjection).invert();
        }

        if (u.frameCounter == 0 && internalFrameCounter > 0) {
            u.frameCounter = internalFrameCounter;
        }
        internalFrameCounter++;

        frameClearR = u.skyColor.x;
        frameClearG = u.skyColor.y;
        frameClearB = u.skyColor.z;

        ByteBuffer uboBuf = MemoryUtil.memByteBuffer(uboMappedPtr, ShaderUniformLayout.UBO_SIZE_BYTES);
        ShaderUniformLayout.pack(uboBuf, u);

        prevModelView.set(u.gbufferModelView);
        prevProjection.set(u.gbufferProjection);
        prevCameraPos.set(u.cameraPosition);
        hasPrevFrameState = true;
    }

    private Vector3f computeSunWorldDir(float sunAngle) {
        float rad = sunAngle * (float) (Math.PI * 2.0);
        float pathRotDeg = activeProperties != null ? activeProperties.getSunPathRotation() : 0.0f;
        float pathRotRad = (float) Math.toRadians(pathRotDeg);
        float baseY = (float) Math.sin(rad);
        Vector3f dir = new Vector3f(
                (float) Math.cos(rad),
                baseY * (float) Math.cos(pathRotRad),
                baseY * (float) Math.sin(pathRotRad));
        if (dir.lengthSquared() > 1e-6f) {
            return dir.normalize();
        }
        return new Vector3f(0.0f, 1.0f, 0.0f);
    }

    private void populateShadowAndCelestialVectors(
            ShaderUniformLayout.UniformData u,
            Vector3f sunWorldDir,
            boolean preserveCallerVectors) {
        u.gbufferModelView.m30(0.0f).m31(0.0f).m32(0.0f).m33(1.0f);
        float shadowDist = activeProperties != null && activeProperties.getShadowDistance() > 0.0f
                ? activeProperties.getShadowDistance()
                : 96.0f;

        Vector3f shadowLightWorldDir = sunWorldDir.y >= 0.0f ? sunWorldDir : new Vector3f(sunWorldDir).negate();
        u.shadowModelView.identity().lookAt(
                new Vector3f(shadowLightWorldDir).mul(64.0f),
                new Vector3f(0.0f, 0.0f, 0.0f),
                Math.abs(shadowLightWorldDir.y) > 0.99f
                        ? new Vector3f(0.0f, 0.0f, 1.0f)
                        : new Vector3f(0.0f, 1.0f, 0.0f));
        u.shadowModelView.m30(0.0f).m31(0.0f).m32(0.0f).m33(1.0f);
        u.shadowModelViewInverse = new Matrix4f(u.shadowModelView).invert();

        // OpenGL [-1, 1] Z ortho projection matching OptiFine/Iris shadowClip.z * 0.5 + 0.5 convention;
        // GlslToVulkanTranspiler remaps shadow.vsh gl_Position.z to Vulkan [0, 1] after user main().
        u.shadowProjection.identity().setOrtho(-shadowDist, shadowDist, -shadowDist, shadowDist, -128.0f, 256.0f, false);
        u.shadowProjectionInverse = new Matrix4f(u.shadowProjection).invert();

        if (!preserveCallerVectors) {
            Vector4f sunView = u.gbufferModelView.transform(
                    new Vector4f(sunWorldDir.x, sunWorldDir.y, sunWorldDir.z, 0.0f));
            Vector3f sunViewDir = new Vector3f(sunView.x, sunView.y, sunView.z);
            if (sunViewDir.lengthSquared() > 1e-6f) {
                sunViewDir.normalize();
            } else {
                sunViewDir.set(0.0f, 1.0f, 0.0f);
            }
            u.sunPosition.set(sunViewDir).mul(100.0f);
            u.moonPosition.set(sunViewDir).negate().mul(100.0f);
            Vector3f shadowLightDir = sunWorldDir.y >= 0.0f ? sunViewDir : new Vector3f(sunViewDir).negate();
            u.shadowLightPosition.set(shadowLightDir).mul(100.0f);
        }

        if (!preserveCallerVectors || u.upPosition.lengthSquared() <= 1e-6f) {
            Vector4f upView = u.gbufferModelView.transform(new Vector4f(0.0f, 1.0f, 0.0f, 0.0f));
            Vector3f upViewDir = new Vector3f(upView.x, upView.y, upView.z);
            if (upViewDir.lengthSquared() > 1e-6f) {
                upViewDir.normalize();
            } else {
                upViewDir.set(0.0f, 1.0f, 0.0f);
            }
            u.upPosition.set(upViewDir).mul(100.0f);
        }
    }

    /**
     * Records a complete multi-pass shaderpack frame using {@link FrameParams}.
     */
    public void recordFrame(
            VkCommandBuffer cmd,
            long sharedColorImage,
            long sharedColorImageView,
            long sharedDepthImage,
            int width,
            int height,
            FrameParams params) {
        ShaderUniformLayout.UniformData uniforms = buildUniformData(params, width, height);
        recordFrame(cmd, sharedColorImage, sharedColorImageView, sharedDepthImage, width, height, uniforms);
    }

    /**
     * Records a complete multi-pass shaderpack frame into {@code cmd}:
     * <ol>
     *   <li>{@code chunk_cull} compute pass (populates {@code indirectCmdBuffer} and {@code drawCountBuffer})</li>
     *   <li>Clear &amp; initialize all offscreen {@code colortex}, {@code depthtex}, and {@code shadowtex} targets</li>
     *   <li>{@code shadow} pass (if present in pack) + copy {@code shadowtex0} &rarr; {@code shadowtex1}</li>
     *   <li>{@code gbuffers_terrain} pass (indirect terrain draw) + copy {@code depthtex0} &rarr; {@code depthtex1}/{@code depthtex2}</li>
     *   <li>{@code gbuffers_water} pass (if present in pack)</li>
     *   <li>{@code deferred*} fullscreen passes with {@code colortex0..7} ping-ponging</li>
     *   <li>{@code composite*} fullscreen passes with {@code colortex0..7} ping-ponging</li>
     *   <li>{@code final} fullscreen pass writing directly into {@code sharedColorImageView} + depth copy into {@code sharedDepthImage}</li>
     * </ol>
     */
    public void recordFrame(
            VkCommandBuffer cmd,
            long sharedColorImage,
            long sharedColorImageView,
            long sharedDepthImage,
            int width,
            int height,
            ShaderUniformLayout.UniformData uniforms) {
        if (!initialized || !packReady) {
            throw new IllegalStateException("VkShaderPackPipeline is not initialized or no shaderpack is compiled");
        }

        float desiredScale = org.sakus.aerogelvk.config.ValkyrieConfig.get().getShaderPackRenderScale();
        int desiredShadowCap = org.sakus.aerogelvk.config.ValkyrieConfig.get().getShaderPackShadowResolutionCap();
        int scaledWidth = computeScaledDim(width, desiredScale);
        int scaledHeight = computeScaledDim(height, desiredScale);

        if (desiredShadowCap != activeShadowCap) {
            this.displayWidth = Math.max(1, width);
            this.displayHeight = Math.max(1, height);
            this.activeRenderScale = desiredScale;
            this.activeShadowCap = desiredShadowCap;
            targets.reconfigure(scaledWidth, scaledHeight, activeProperties);
            invalidateShadow();
        } else if (scaledWidth != targets.getWidth() || scaledHeight != targets.getHeight()
                || width != displayWidth || height != displayHeight
                || Math.abs(desiredScale - activeRenderScale) > 1e-4f) {
            resize(width, height);
            scaledWidth = targets.getWidth();
            scaledHeight = targets.getHeight();
        }

        updateUniforms(uniforms, scaledWidth, scaledHeight);
        updateDescriptorSet0();

        boolean hasShadowPass = (shadowPass != null && shadowPass.getPipeline() != VK_NULL_HANDLE);
        boolean skipShadow = shouldSkipStaticShadowPass(uniforms, hasShadowPass, desiredShadowCap);

        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Flush any pending mesh uploads into MegaBuffer if not already flushed by caller
            megaBuffer.recordPendingTransfers(cmd);

            int slotLimit = megaBuffer.getSlotDispatchLimit();
            int maxDrawCount = Math.max(1, slotLimit);

            // 1. Clear all shaderpack targets (preserving shadowtex/shadowcolor when skipping static shadow pass)
            targets.recordClearAllTargets(cmd, frameClearR, frameClearG, frameClearB, !skipShadow, stack);

            int passSetSlot = 0;

            // 2. Prepare fullscreen passes (prepare, prepare1..15)
            for (CompiledPass pass : preparePasses) {
                if (pass.getPipeline() == VK_NULL_HANDLE) {
                    continue;
                }
                int qPass = (gpuProfiler != null) ? gpuProfiler.beginPass(cmd, profilerFrameSlot, pass.getName()) : -1;
                long set1 = updateDescriptorSet1ForCurrentState(passSetSlot++);
                recordFullscreenColorPass(cmd, stack, pass, set1, scaledWidth, scaledHeight);
                if (gpuProfiler != null) gpuProfiler.endPass(cmd, profilerFrameSlot, qPass);
                notifyPassTrace(cmd, stack, pass.getName());
            }

            boolean hasWaterPass = (gbuffersWaterPass != null && gbuffersWaterPass.getPipeline() != VK_NULL_HANDLE);

            // 3. Optional shadow pass (skipped on static frames when shadowSkipStaticFrames is enabled)
            if (hasShadowPass && !skipShadow) {
                int qShadow = (gpuProfiler != null) ? gpuProfiler.beginPass(cmd, profilerFrameSlot, "shadow") : -1;
                dispatchShaderPackCull(cmd, stack, slotLimit, 0);
                long set1 = updateDescriptorSet1ForCurrentState(passSetSlot++);
                recordShadowPass(cmd, stack, shadowPass, set1, maxDrawCount);
                targets.recordCopyShadowDepth(cmd, 0, 1, stack);
                if (gpuProfiler != null) gpuProfiler.endPass(cmd, profilerFrameSlot, qShadow);
                recordShadowState(uniforms, desiredShadowCap);
            }

            // 4. Primary gbuffers_terrain pass (opaque-only when gbuffers_water is present, else all)
            if (gbuffersTerrainPass != null && gbuffersTerrainPass.getPipeline() != VK_NULL_HANDLE) {
                int qTerrain = (gpuProfiler != null) ? gpuProfiler.beginPass(cmd, profilerFrameSlot, "gbuffers_terrain") : -1;
                dispatchShaderPackCull(cmd, stack, slotLimit, hasWaterPass ? 1 : 0);
                long set1 = updateDescriptorSet1ForCurrentState(passSetSlot++);
                recordGeometryPass(cmd, stack, gbuffersTerrainPass, set1, maxDrawCount, scaledWidth, scaledHeight, false);
                if (gpuProfiler != null) gpuProfiler.endPass(cmd, profilerFrameSlot, qTerrain);
                notifyPassTrace(cmd, stack, "gbuffers_terrain");
            }

            // Copy solid terrain depth depthtex0 -> depthtex1 and depthtex2
            targets.recordCopyDepth(cmd, 0, 1, stack);
            targets.recordCopyDepth(cmd, 0, 2, stack);

            // 5. Optional gbuffers_water pass (translucent geometry filtered by passFlags=2)
            if (hasWaterPass && megaBuffer.hasTranslucentSections()) {
                int qWater = (gpuProfiler != null) ? gpuProfiler.beginPass(cmd, profilerFrameSlot, "gbuffers_water") : -1;
                dispatchShaderPackCull(cmd, stack, slotLimit, 2);
                long set1 = updateDescriptorSet1ForCurrentState(passSetSlot++);
                recordGeometryPass(cmd, stack, gbuffersWaterPass, set1, maxDrawCount, scaledWidth, scaledHeight, true);
                if (gpuProfiler != null) gpuProfiler.endPass(cmd, profilerFrameSlot, qWater);
                notifyPassTrace(cmd, stack, "gbuffers_water");
            }

            // 6. Deferred fullscreen passes (deferred, deferred1..15)
            for (CompiledPass pass : deferredPasses) {
                if (pass.getPipeline() == VK_NULL_HANDLE) {
                    continue;
                }
                int qDef = (gpuProfiler != null) ? gpuProfiler.beginPass(cmd, profilerFrameSlot, pass.getName()) : -1;
                long set1 = updateDescriptorSet1ForCurrentState(passSetSlot++);
                recordFullscreenColorPass(cmd, stack, pass, set1, scaledWidth, scaledHeight);
                if (gpuProfiler != null) gpuProfiler.endPass(cmd, profilerFrameSlot, qDef);
                notifyPassTrace(cmd, stack, pass.getName());
            }

            // 7. Composite fullscreen passes (composite, composite1..15)
            for (CompiledPass pass : compositePasses) {
                if (pass.getPipeline() == VK_NULL_HANDLE) {
                    continue;
                }
                int qComp = (gpuProfiler != null) ? gpuProfiler.beginPass(cmd, profilerFrameSlot, pass.getName()) : -1;
                long set1 = updateDescriptorSet1ForCurrentState(passSetSlot++);
                recordFullscreenColorPass(cmd, stack, pass, set1, scaledWidth, scaledHeight);
                if (gpuProfiler != null) gpuProfiler.endPass(cmd, profilerFrameSlot, qComp);
                notifyPassTrace(cmd, stack, pass.getName());
            }

            // 8. Final fullscreen pass -> writes directly into full-resolution sharedColorImageView
            int qFinal = (gpuProfiler != null) ? gpuProfiler.beginPass(cmd, profilerFrameSlot, "final") : -1;
            long finalSet1 = updateDescriptorSet1ForCurrentState(passSetSlot);
            recordFinalPass(cmd, stack, finalPass, finalSet1, sharedColorImage, sharedColorImageView, width, height);
            if (gpuProfiler != null) gpuProfiler.endPass(cmd, profilerFrameSlot, qFinal);

            // 9. Copy/blit depthtex0 -> sharedDepthImage (leaves sharedDepthImage in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            if (sharedDepthImage != VK_NULL_HANDLE) {
                int qDepth = (gpuProfiler != null) ? gpuProfiler.beginPass(cmd, profilerFrameSlot, "depth_copy") : -1;
                targets.recordCopyDepthToExternal(cmd, 0, sharedDepthImage, width, height, stack);
                if (gpuProfiler != null) gpuProfiler.endPass(cmd, profilerFrameSlot, qDepth);
            }
        }
    }

    private void dispatchShaderPackCull(VkCommandBuffer cmd, MemoryStack stack, int slotLimit, int passFlags) {
        vkCmdFillBuffer(cmd, megaBuffer.getDrawCountBuffer().buffer, 0, 4, 0);

        VkBufferMemoryBarrier.Buffer fillBarrier = VkBufferMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .buffer(megaBuffer.getDrawCountBuffer().buffer)
                .offset(0)
                .size(4);
        vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, null, fillBarrier, null);

        if (slotLimit > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                    ShaderDescriptorBindings.SET_GLOBAL, stack.longs(descriptorSet0), null);

            ByteBuffer pc = stack.malloc(8);
            pc.putInt(0, slotLimit);
            pc.putInt(4, passFlags);
            vkCmdPushConstants(cmd, pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                    0, pc);

            int groupsX = (slotLimit + 63) / 64;
            vkCmdDispatch(cmd, groupsX, 1, 1);
        }

        VkMemoryBarrier.Buffer cullBarrier = VkMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
        vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                0, cullBarrier, null, null);
    }

    private void recordShadowPass(
            VkCommandBuffer cmd,
            MemoryStack stack,
            CompiledPass pass,
            long set1,
            int maxDrawCount) {
        int shadowRes = targets.getShadowResolution();
        int[] dbs = pass.getDrawBuffers();
        int attCount = Math.min(pass.getColorAttachmentCount(), dbs.length);

        for (int i = 0; i < attCount; i++) {
            int scIdx = Math.min(Math.max(dbs[i], 0), VkShaderPackTargets.MAX_SHADOWCOLOR - 1);
            VkShaderPackTargets.recordImageBarrier(cmd, targets.getShadowColorImage(scIdx),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, stack);
        }

        VkShaderPackTargets.recordImageBarrier(cmd, targets.getShadowDepthImage(0),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, stack);

        VkRenderingAttachmentInfo.Buffer colorAttachments = null;
        if (attCount > 0) {
            colorAttachments = VkRenderingAttachmentInfo.calloc(attCount, stack);
            for (int i = 0; i < attCount; i++) {
                int scIdx = Math.min(Math.max(dbs[i], 0), VkShaderPackTargets.MAX_SHADOWCOLOR - 1);
                colorAttachments.get(i)
                        .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                        .imageView(targets.getShadowColorView(scIdx))
                        .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                        .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
            }
        }

        VkRenderingAttachmentInfo depthAttachment = VkRenderingAttachmentInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(targets.getShadowDepthView(0))
                .imageLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);

        VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_INFO)
                .renderArea(ra -> ra.offset(o -> o.set(0, 0)).extent(e -> e.set(shadowRes, shadowRes)))
                .layerCount(1)
                .pColorAttachments(colorAttachments)
                .pDepthAttachment(depthAttachment);

        vkCmdBeginRendering(cmd, renderingInfo);
        setViewportAndScissor(cmd, stack, shadowRes, shadowRes);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass.getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                0, stack.longs(descriptorSet0, set1), null);
        vkCmdBindIndexBuffer(cmd, megaBuffer.getGlobalQuadIndexBuffer().buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirectCount(cmd,
                megaBuffer.getIndirectCmdBuffer().buffer, 0,
                megaBuffer.getDrawCountBuffer().buffer, 0,
                maxDrawCount, VkMegaBuffer.INDIRECT_CMD_SIZE_BYTES);
        vkCmdEndRendering(cmd);

        for (int i = 0; i < attCount; i++) {
            int scIdx = Math.min(Math.max(dbs[i], 0), VkShaderPackTargets.MAX_SHADOWCOLOR - 1);
            VkShaderPackTargets.recordImageBarrier(cmd, targets.getShadowColorImage(scIdx),
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);
        }

        VkShaderPackTargets.recordImageBarrier(cmd, targets.getShadowDepthImage(0),
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);
    }

    private void recordGeometryPass(
            VkCommandBuffer cmd,
            MemoryStack stack,
            CompiledPass pass,
            long set1,
            int maxDrawCount,
            int width,
            int height,
            boolean seedWriteFromRead) {
        int[] dbs = pass.getDrawBuffers();
        int attCount = Math.min(pass.getColorAttachmentCount(), dbs.length);

        for (int i = 0; i < attCount; i++) {
            int ctIdx = dbs[i];
            if (seedWriteFromRead) {
                recordSeedWriteFromRead(cmd, ctIdx, width, height, stack);
            } else {
                VkShaderPackTargets.recordImageBarrier(cmd, targets.getColorWriteImage(ctIdx),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, stack);
            }
        }

        VkShaderPackTargets.recordImageBarrier(cmd, targets.getDepthImage(0),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, stack);

        VkRenderingAttachmentInfo.Buffer colorAttachments = VkRenderingAttachmentInfo.calloc(attCount, stack);
        for (int i = 0; i < attCount; i++) {
            int ctIdx = dbs[i];
            colorAttachments.get(i)
                    .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                    .imageView(targets.getColorWriteAttachmentView(ctIdx))
                    .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                    .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                    .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        }

        VkRenderingAttachmentInfo depthAttachment = VkRenderingAttachmentInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(targets.getDepthView(0))
                .imageLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);

        VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_INFO)
                .renderArea(ra -> ra.offset(o -> o.set(0, 0)).extent(e -> e.set(width, height)))
                .layerCount(1)
                .pColorAttachments(colorAttachments)
                .pDepthAttachment(depthAttachment);

        vkCmdBeginRendering(cmd, renderingInfo);
        setViewportAndScissor(cmd, stack, width, height);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass.getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                0, stack.longs(descriptorSet0, set1), null);
        vkCmdBindIndexBuffer(cmd, megaBuffer.getGlobalQuadIndexBuffer().buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirectCount(cmd,
                megaBuffer.getIndirectCmdBuffer().buffer, 0,
                megaBuffer.getDrawCountBuffer().buffer, 0,
                maxDrawCount, VkMegaBuffer.INDIRECT_CMD_SIZE_BYTES);
        vkCmdEndRendering(cmd);

        VkShaderPackTargets.recordImageBarrier(cmd, targets.getDepthImage(0),
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);

        for (int i = 0; i < attCount; i++) {
            int ctIdx = dbs[i];
            long writtenImg = targets.getColorWriteImage(ctIdx);
            int mips = targets.getColorMipLevels(ctIdx);
            targets.recordTransitionOrMipmapAfterColorWrite(cmd, writtenImg, mips, stack);
            targets.flip(ctIdx);
        }
    }

    private void recordFullscreenColorPass(
            VkCommandBuffer cmd,
            MemoryStack stack,
            CompiledPass pass,
            long set1,
            int width,
            int height) {
        int[] dbs = pass.getDrawBuffers();
        int attCount = Math.min(pass.getColorAttachmentCount(), dbs.length);
        if (attCount <= 0) {
            return;
        }

        for (int i = 0; i < attCount; i++) {
            int ctIdx = dbs[i];
            recordSeedWriteFromRead(cmd, ctIdx, width, height, stack);
        }

        VkRenderingAttachmentInfo.Buffer colorAttachments = VkRenderingAttachmentInfo.calloc(attCount, stack);
        for (int i = 0; i < attCount; i++) {
            int ctIdx = dbs[i];
            colorAttachments.get(i)
                    .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                    .imageView(targets.getColorWriteAttachmentView(ctIdx))
                    .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                    .loadOp(VK_ATTACHMENT_LOAD_OP_LOAD)
                    .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        }

        VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_INFO)
                .renderArea(ra -> ra.offset(o -> o.set(0, 0)).extent(e -> e.set(width, height)))
                .layerCount(1)
                .pColorAttachments(colorAttachments);

        vkCmdBeginRendering(cmd, renderingInfo);
        setViewportAndScissor(cmd, stack, width, height);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass.getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                0, stack.longs(descriptorSet0, set1), null);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);

        for (int i = 0; i < attCount; i++) {
            int ctIdx = dbs[i];
            long writtenImg = targets.getColorWriteImage(ctIdx);
            int mips = targets.getColorMipLevels(ctIdx);
            targets.recordTransitionOrMipmapAfterColorWrite(cmd, writtenImg, mips, stack);
            targets.flip(ctIdx);
        }
    }

    private void recordFinalPass(
            VkCommandBuffer cmd,
            MemoryStack stack,
            CompiledPass pass,
            long set1,
            long sharedColorImage,
            long sharedColorImageView,
            int width,
            int height) {
        VkShaderPackTargets.recordImageBarrier(cmd, sharedColorImage,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                0,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, stack);

        VkRenderingAttachmentInfo.Buffer colorAttachment = VkRenderingAttachmentInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                .imageView(sharedColorImageView)
                .imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        colorAttachment.get(0).clearValue().color()
                .float32(0, frameClearR).float32(1, frameClearG).float32(2, frameClearB).float32(3, 1.0f);

        VkRenderingInfo renderingInfo = VkRenderingInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDERING_INFO)
                .renderArea(ra -> ra.offset(o -> o.set(0, 0)).extent(e -> e.set(width, height)))
                .layerCount(1)
                .pColorAttachments(colorAttachment);

        vkCmdBeginRendering(cmd, renderingInfo);
        setViewportAndScissor(cmd, stack, width, height);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass.getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                0, stack.longs(descriptorSet0, set1), null);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);

        VkShaderPackTargets.recordImageBarrier(cmd, sharedColorImage,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, stack);
    }

    /**
     * Seeds {@code getColorWriteImage(ctIdx)} (mip 0) from {@code getColorReadImage(ctIdx)} before a pass renders
     * so shaders that discard pixels or alpha-blend preserve the previous contents of {@code colortexN} while
     * still reading from {@code getColorReadImage(ctIdx)} in {@code VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}.
     */
    private void recordSeedWriteFromRead(
            VkCommandBuffer cmd,
            int ctIdx,
            int width,
            int height,
            MemoryStack stack) {
        long readImg = targets.getColorReadImage(ctIdx);
        long writeImg = targets.getColorWriteImage(ctIdx);

        VkShaderPackTargets.recordImageBarrier(cmd, readImg,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        VkShaderPackTargets.recordImageBarrier(cmd, writeImg,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, stack);

        VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
        region.srcSubresource()
                .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                .mipLevel(0)
                .baseArrayLayer(0)
                .layerCount(1);
        region.dstSubresource()
                .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                .mipLevel(0)
                .baseArrayLayer(0)
                .layerCount(1);
        region.extent().set(width, height, 1);

        vkCmdCopyImage(cmd,
                readImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                writeImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                region);

        VkShaderPackTargets.recordImageBarrier(cmd, readImg,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, stack);

        VkShaderPackTargets.recordImageBarrier(cmd, writeImg,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, stack);
    }

    private void setViewportAndScissor(VkCommandBuffer cmd, MemoryStack stack, int width, int height) {
        VkViewport.Buffer viewport = VkViewport.calloc(1, stack)
                .x(0.0f)
                .y(0.0f)
                .width((float) width)
                .height((float) height)
                .minDepth(0.0f)
                .maxDepth(1.0f);
        vkCmdSetViewport(cmd, 0, viewport);

        VkRect2D.Buffer scissor = VkRect2D.calloc(1, stack)
                .offset(o -> o.set(0, 0))
                .extent(e -> e.set(width, height));
        vkCmdSetScissor(cmd, 0, scissor);
    }

    private static long compileShaderModule(VkDevice device, String shaderName, String glslSource, int stage) {
        ByteBuffer spirv = VkShaderCompiler.compileGlslToSpirv(shaderName, glslSource, stage);
        try {
            return VkShaderCompiler.createShaderModule(device, spirv);
        } finally {
            MemoryUtil.memFree(spirv);
        }
    }

    private CompiledPass compilePassProgram(PassProgram passProg, boolean alphaBlend) {
        if (passProg == null) {
            return null;
        }
        String passName = passProg.getName();
        PassKind kind = passProg.getPassKind();
        try {
            TranspiledProgram transpiled = passProg.transpile();
            int[] drawBuffers = sanitizeDrawBuffers(kind, passProg.getDrawBuffers(), transpiled.getColorAttachmentCount());
            int colorCount = Math.min(transpiled.getColorAttachmentCount(), drawBuffers.length);

            int[] colorFormats = new int[colorCount];
            for (int i = 0; i < colorCount; i++) {
                if (kind == PassKind.FINAL) {
                    colorFormats[i] = finalColorFormat;
                } else if (kind == PassKind.SHADOW) {
                    colorFormats[i] = VK_FORMAT_R8G8B8A8_UNORM;
                } else {
                    colorFormats[i] = targets.getColorFormat(drawBuffers[i]);
                }
            }

            boolean hasDepthAttachment = (kind == PassKind.GBUFFERS || kind == PassKind.SHADOW);
            int depthFormat = hasDepthAttachment ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED;
            boolean depthTest = hasDepthAttachment;
            boolean depthWrite = hasDepthAttachment && !alphaBlend;

            long vertModule = compileShaderModule(
                    ctx.getVkDevice(), passName + ".vsh", transpiled.getVertexGlsl(), VkShaderCompiler.STAGE_VERTEX);
            long fragModule = VK_NULL_HANDLE;
            try {
                fragModule = compileShaderModule(
                        ctx.getVkDevice(), passName + ".fsh", transpiled.getFragmentGlsl(), VkShaderCompiler.STAGE_FRAGMENT);
                long pipeline = createGraphicsPipeline(
                        vertModule, fragModule, colorFormats, depthFormat, depthTest, depthWrite, alphaBlend,
                        kind == PassKind.SHADOW);
                return new CompiledPass(
                        passName, kind, pipeline, drawBuffers, colorCount, depthTest, depthWrite, alphaBlend);
            } finally {
                if (fragModule != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(ctx.getVkDevice(), fragModule, null);
                }
                if (vertModule != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(ctx.getVkDevice(), vertModule, null);
                }
            }
        } catch (Exception e) {
            String warning = "Pass '" + passName + "' (" + kind + ") failed compilation and will be skipped: " + e.getMessage();
            compileWarnings.add(warning);
            LOGGER.warn("[Valkyrie] {}", warning);
            return null;
        }
    }

    private CompiledPass compileFallbackTerrainPass() {
        try {
            int[] drawBuffers = new int[]{0};
            TranspiledProgram transpiled = GlslToVulkanTranspiler.transpileProgram(
                    FALLBACK_TERRAIN_VSH, FALLBACK_TERRAIN_FSH, PassKind.GBUFFERS, drawBuffers);
            int[] colorFormats = new int[]{targets.getColorFormat(0)};
            long vertModule = compileShaderModule(
                    ctx.getVkDevice(), "fallback_gbuffers_terrain.vsh",
                    transpiled.getVertexGlsl(), VkShaderCompiler.STAGE_VERTEX);
            long fragModule = VK_NULL_HANDLE;
            try {
                fragModule = compileShaderModule(
                        ctx.getVkDevice(), "fallback_gbuffers_terrain.fsh",
                        transpiled.getFragmentGlsl(), VkShaderCompiler.STAGE_FRAGMENT);
                long pipeline = createGraphicsPipeline(
                        vertModule, fragModule, colorFormats, VK_FORMAT_D32_SFLOAT, true, true, false, false);
                return new CompiledPass(
                        "gbuffers_terrain", PassKind.GBUFFERS, pipeline, drawBuffers, 1, true, true, false);
            } finally {
                if (fragModule != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(ctx.getVkDevice(), fragModule, null);
                }
                if (vertModule != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(ctx.getVkDevice(), vertModule, null);
                }
            }
        } catch (Exception e) {
            LOGGER.error("[Valkyrie] Failed to compile fallback gbuffers_terrain pass", e);
            return null;
        }
    }

    private CompiledPass compileFallbackFinalPass() {
        try {
            int[] drawBuffers = new int[]{0};
            TranspiledProgram transpiled = GlslToVulkanTranspiler.transpileProgram(
                    FALLBACK_FINAL_VSH, FALLBACK_FINAL_FSH, PassKind.FINAL, drawBuffers);
            int[] colorFormats = new int[]{finalColorFormat};
            long vertModule = compileShaderModule(
                    ctx.getVkDevice(), "fallback_final.vsh",
                    transpiled.getVertexGlsl(), VkShaderCompiler.STAGE_VERTEX);
            long fragModule = VK_NULL_HANDLE;
            try {
                fragModule = compileShaderModule(
                        ctx.getVkDevice(), "fallback_final.fsh",
                        transpiled.getFragmentGlsl(), VkShaderCompiler.STAGE_FRAGMENT);
                long pipeline = createGraphicsPipeline(
                        vertModule, fragModule, colorFormats, VK_FORMAT_UNDEFINED, false, false, false, false);
                return new CompiledPass(
                        "final", PassKind.FINAL, pipeline, drawBuffers, 1, false, false, false);
            } finally {
                if (fragModule != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(ctx.getVkDevice(), fragModule, null);
                }
                if (vertModule != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(ctx.getVkDevice(), vertModule, null);
                }
            }
        } catch (Exception e) {
            LOGGER.error("[Valkyrie] Failed to compile fallback final pass", e);
            return null;
        }
    }

    private static int[] sanitizeDrawBuffers(PassKind kind, int[] rawDrawBuffers, int count) {
        if (kind == PassKind.FINAL) {
            return new int[]{0};
        }
        if (kind == PassKind.SHADOW && count <= 0) {
            return new int[0];
        }
        int maxAttachments = (kind == PassKind.SHADOW)
                ? VkShaderPackTargets.MAX_SHADOWCOLOR
                : MAX_COLOR_ATTACHMENTS_PER_PASS;
        int clampedCount = Math.max(1, Math.min(count, maxAttachments));
        int maxTargetIndex = (kind == PassKind.SHADOW)
                ? VkShaderPackTargets.MAX_SHADOWCOLOR - 1
                : VkShaderPackTargets.MAX_COLORTEX - 1;

        int[] result = new int[clampedCount];
        boolean[] used = new boolean[maxTargetIndex + 1];
        for (int i = 0; i < clampedCount; i++) {
            int candidate = (rawDrawBuffers != null && i < rawDrawBuffers.length) ? rawDrawBuffers[i] : i;
            if (candidate < 0 || candidate > maxTargetIndex || used[candidate]) {
                // Allocate spare slots from the top down so invalid/duplicate indices never clobber colortex0..7.
                for (int alt = maxTargetIndex; alt >= 0; alt--) {
                    if (!used[alt]) {
                        candidate = alt;
                        break;
                    }
                }
            }
            used[candidate] = true;
            result[i] = candidate;
        }
        return result;
    }

    private long createGraphicsPipeline(
            long vertModule,
            long fragModule,
            int[] colorFormats,
            int depthFormat,
            boolean depthTest,
            boolean depthWrite,
            boolean alphaBlend,
            boolean isShadowPass) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(2, stack);
            stages.get(0)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                    .stage(VK_SHADER_STAGE_VERTEX_BIT)
                    .module(vertModule)
                    .pName(stack.UTF8("main"));
            stages.get(1)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                    .stage(VK_SHADER_STAGE_FRAGMENT_BIT)
                    .module(fragModule)
                    .pName(stack.UTF8("main"));

            VkPipelineVertexInputStateCreateInfo vertexInput = VkPipelineVertexInputStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);

            VkPipelineInputAssemblyStateCreateInfo inputAssembly = VkPipelineInputAssemblyStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
                    .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                    .primitiveRestartEnable(false);

            VkPipelineViewportStateCreateInfo viewportState = VkPipelineViewportStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
                    .viewportCount(1)
                    .scissorCount(1);

            VkPipelineRasterizationStateCreateInfo rasterizer = VkPipelineRasterizationStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
                    .depthClampEnable(isShadowPass && VkContext.getInstance().isDepthClampSupported())
                    .rasterizerDiscardEnable(false)
                    .polygonMode(VK_POLYGON_MODE_FILL)
                    .lineWidth(1.0f)
                    .cullMode(VK_CULL_MODE_NONE)
                    .frontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE)
                    .depthBiasEnable(isShadowPass)
                    .depthBiasConstantFactor(isShadowPass ? 1.5f : 0.0f)
                    .depthBiasSlopeFactor(isShadowPass ? 1.75f : 0.0f);

            VkPipelineMultisampleStateCreateInfo multisampling = VkPipelineMultisampleStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
                    .sampleShadingEnable(false)
                    .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT);

            VkPipelineDepthStencilStateCreateInfo depthStencil = VkPipelineDepthStencilStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO)
                    .depthTestEnable(depthTest)
                    .depthWriteEnable(depthWrite)
                    .depthCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL)
                    .depthBoundsTestEnable(false)
                    .stencilTestEnable(false);

            VkPipelineColorBlendAttachmentState.Buffer blendAttachments =
                    colorFormats.length > 0 ? VkPipelineColorBlendAttachmentState.calloc(colorFormats.length, stack) : null;
            boolean indBlend = VkContext.getInstance().isIndependentBlendSupported();
            boolean allBlendable = true;
            if (alphaBlend && !indBlend) {
                for (int fmt : colorFormats) {
                    if (!VkContext.getInstance().supportsColorAttachmentBlend(fmt)) {
                        allBlendable = false;
                        break;
                    }
                }
            }
            for (int i = 0; i < colorFormats.length; i++) {
                boolean attBlend = alphaBlend && (indBlend
                        ? VkContext.getInstance().supportsColorAttachmentBlend(colorFormats[i])
                        : allBlendable);
                VkPipelineColorBlendAttachmentState att = blendAttachments.get(i)
                        .colorWriteMask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
                        .blendEnable(attBlend);
                if (attBlend) {
                    att.srcColorBlendFactor(VK_BLEND_FACTOR_SRC_ALPHA)
                            .dstColorBlendFactor(VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
                            .colorBlendOp(VK_BLEND_OP_ADD)
                            .srcAlphaBlendFactor(VK_BLEND_FACTOR_ONE)
                            .dstAlphaBlendFactor(VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
                            .alphaBlendOp(VK_BLEND_OP_ADD);
                }
            }

            VkPipelineColorBlendStateCreateInfo colorBlending = VkPipelineColorBlendStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO)
                    .logicOpEnable(false)
                    .pAttachments(blendAttachments);

            IntBuffer dynStates = stack.ints(VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR);
            VkPipelineDynamicStateCreateInfo dynamicState = VkPipelineDynamicStateCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO)
                    .pDynamicStates(dynStates);

            VkPipelineRenderingCreateInfo renderingInfo = VkPipelineRenderingCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
                    .colorAttachmentCount(colorFormats.length)
                    .pColorAttachmentFormats(colorFormats.length > 0 ? stack.ints(colorFormats) : null)
                    .depthAttachmentFormat(depthFormat);

            VkGraphicsPipelineCreateInfo.Buffer pipelineInfo = VkGraphicsPipelineCreateInfo.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
                    .pNext(renderingInfo.address())
                    .pStages(stages)
                    .pVertexInputState(vertexInput)
                    .pInputAssemblyState(inputAssembly)
                    .pViewportState(viewportState)
                    .pRasterizationState(rasterizer)
                    .pMultisampleState(multisampling)
                    .pDepthStencilState(depthStencil)
                    .pColorBlendState(colorBlending)
                    .pDynamicState(dynamicState)
                    .layout(pipelineLayout)
                    .renderPass(VK_NULL_HANDLE);

            LongBuffer pPipeline = stack.mallocLong(1);
            int res = vkCreateGraphicsPipelines(ctx.getVkDevice(), VK_NULL_HANDLE, pipelineInfo, null, pPipeline);
            if (res != VK_SUCCESS) {
                throw new IllegalStateException("vkCreateGraphicsPipelines failed with code: " + res);
            }
            return pPipeline.get(0);
        }
    }

    private void createCullPipeline() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long cullModule = compileShaderModule(
                    ctx.getVkDevice(), "shaderpack_cull.comp",
                    SHADERPACK_CULL_COMP_SRC, VkShaderCompiler.STAGE_COMPUTE);
            try {
                VkComputePipelineCreateInfo.Buffer computeInfo = VkComputePipelineCreateInfo.calloc(1, stack)
                        .sType(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO)
                        .layout(pipelineLayout);
                computeInfo.get(0).stage()
                        .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                        .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                        .module(cullModule)
                        .pName(stack.UTF8("main"));

                LongBuffer pPipeline = stack.mallocLong(1);
                int res = vkCreateComputePipelines(ctx.getVkDevice(), VK_NULL_HANDLE, computeInfo, null, pPipeline);
                if (res != VK_SUCCESS) {
                    throw new IllegalStateException("vkCreateComputePipelines failed for shaderpack_cull.comp: " + res);
                }
                cullPipeline = pPipeline.get(0);
            } finally {
                vkDestroyShaderModule(ctx.getVkDevice(), cullModule, null);
            }
        }
    }

    private void createUboBuffer() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkBufferCreateInfo bufInfo = VkBufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(ShaderUniformLayout.UBO_SIZE_BYTES)
                    .usage(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);

            VmaAllocationCreateInfo allocInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_AUTO)
                    .flags(Vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                            | Vma.VMA_ALLOCATION_CREATE_MAPPED_BIT);

            LongBuffer pBuf = stack.mallocLong(1);
            PointerBuffer pAlloc = stack.mallocPointer(1);
            int res = Vma.vmaCreateBuffer(ctx.getVmaAllocator(), bufInfo, allocInfo, pBuf, pAlloc, null);
            if (res != VK_SUCCESS) {
                throw new IllegalStateException("Failed to allocate ShaderPackUBO buffer: " + res);
            }
            uboBuffer = pBuf.get(0);
            uboAllocation = pAlloc.get(0);

            PointerBuffer pData = stack.mallocPointer(1);
            Vma.vmaMapMemory(ctx.getVmaAllocator(), uboAllocation, pData);
            uboMappedPtr = pData.get(0);
            MemoryUtil.memSet(uboMappedPtr, 0, ShaderUniformLayout.UBO_SIZE_BYTES);
        }
    }

    private void ensureAtlasUvPopulated() {
        AtlasUVTable uvTable = AtlasUVTable.getInstance();
        if (uvTable.getSpriteCount() > 0) {
            ByteBuffer uvBuf = uvTable.getUvTableBuffer();
            if (uvBuf != null && uvBuf.remaining() >= AtlasUVTable.ENTRY_BYTES) {
                megaBuffer.uploadAtlasUvTable(uvBuf);
                return;
            }
        }
        ByteBuffer defaultEntry = MemoryUtil.memAlloc(AtlasUVTable.ENTRY_BYTES);
        try {
            defaultEntry.putFloat(0.0f).putFloat(0.0f).putFloat(1.0f).putFloat(1.0f);
            defaultEntry.putFloat(1.0f).putFloat(1.0f).putFloat(1.0f).putFloat(1.0f);
            defaultEntry.flip();
            megaBuffer.uploadAtlasUvTable(defaultEntry);
        } finally {
            MemoryUtil.memFree(defaultEntry);
        }
    }

    private void createDescriptorLayoutsAndPool() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int allStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

            // Set 0: Global UBO + MegaBuffer SSBOs + Atlas UV SSBO + Block Atlas Sampler
            VkDescriptorSetLayoutBinding.Buffer bindings0 = VkDescriptorSetLayoutBinding.calloc(8, stack);
            bindings0.get(0)
                    .binding(ShaderDescriptorBindings.BINDING_SHADER_PACK_UBO)
                    .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(allStages);
            for (int b = 1; b <= 6; b++) {
                bindings0.get(b)
                        .binding(b)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .stageFlags(allStages);
            }
            bindings0.get(7)
                    .binding(ShaderDescriptorBindings.BINDING_BLOCK_ATLAS)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .stageFlags(allStages);

            VkDescriptorSetLayoutCreateInfo layoutInfo0 = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                    .pBindings(bindings0);
            LongBuffer pLayout = stack.mallocLong(1);
            if (vkCreateDescriptorSetLayout(ctx.getVkDevice(), layoutInfo0, null, pLayout) != VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Set 0 descriptor set layout");
            }
            descriptorSetLayout0 = pLayout.get(0);

            // Set 1: Shaderpack Offscreen Samplers (colortex0..19, depthtex0..2, shadow*, etc.)
            VkDescriptorSetLayoutBinding.Buffer bindings1 =
                    VkDescriptorSetLayoutBinding.calloc(ShaderDescriptorBindings.SET1_BINDING_COUNT, stack);
            for (int b = 0; b < ShaderDescriptorBindings.SET1_BINDING_COUNT; b++) {
                bindings1.get(b)
                        .binding(b)
                        .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1)
                        .stageFlags(allStages);
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo1 = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                    .pBindings(bindings1);
            if (vkCreateDescriptorSetLayout(ctx.getVkDevice(), layoutInfo1, null, pLayout) != VK_SUCCESS) {
                throw new IllegalStateException("Failed to create Set 1 descriptor set layout");
            }
            descriptorSetLayout1 = pLayout.get(0);

            // Push Constant Range: 16 bytes for vertex/fragment/compute stages
            VkPushConstantRange.Buffer pushConstants = VkPushConstantRange.calloc(1, stack)
                    .stageFlags(allStages)
                    .offset(0)
                    .size(16);

            // Pipeline layout with [Set 0, Set 1] + push constants
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                    .pSetLayouts(stack.longs(descriptorSetLayout0, descriptorSetLayout1))
                    .pPushConstantRanges(pushConstants);
            if (vkCreatePipelineLayout(ctx.getVkDevice(), pipelineLayoutInfo, null, pLayout) != VK_SUCCESS) {
                throw new IllegalStateException("Failed to create VkShaderPackPipeline layout");
            }
            pipelineLayout = pLayout.get(0);

            // Descriptor pool for 1 Set 0 + MAX_PASS_DESCRIPTOR_SETS Set 1s
            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(3, stack);
            poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).descriptorCount(4);
            poolSizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(16);
            poolSizes.get(2).type(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(4 + MAX_PASS_DESCRIPTOR_SETS * ShaderDescriptorBindings.SET1_BINDING_COUNT);

            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                    .maxSets(1 + MAX_PASS_DESCRIPTOR_SETS)
                    .pPoolSizes(poolSizes);
            LongBuffer pPool = stack.mallocLong(1);
            if (vkCreateDescriptorPool(ctx.getVkDevice(), poolInfo, null, pPool) != VK_SUCCESS) {
                throw new IllegalStateException("Failed to create VkShaderPackPipeline descriptor pool");
            }
            descriptorPool = pPool.get(0);

            // Allocate Set 0
            VkDescriptorSetAllocateInfo alloc0 = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                    .descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(descriptorSetLayout0));
            LongBuffer pSet0 = stack.mallocLong(1);
            if (vkAllocateDescriptorSets(ctx.getVkDevice(), alloc0, pSet0) != VK_SUCCESS) {
                throw new IllegalStateException("Failed to allocate Set 0 descriptor set");
            }
            descriptorSet0 = pSet0.get(0);

            // Allocate Set 1 array
            LongBuffer layouts1 = stack.mallocLong(MAX_PASS_DESCRIPTOR_SETS);
            for (int i = 0; i < MAX_PASS_DESCRIPTOR_SETS; i++) {
                layouts1.put(i, descriptorSetLayout1);
            }
            VkDescriptorSetAllocateInfo alloc1 = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                    .descriptorPool(descriptorPool)
                    .pSetLayouts(layouts1);
            LongBuffer pSets1 = stack.mallocLong(MAX_PASS_DESCRIPTOR_SETS);
            if (vkAllocateDescriptorSets(ctx.getVkDevice(), alloc1, pSets1) != VK_SUCCESS) {
                throw new IllegalStateException("Failed to allocate Set 1 descriptor sets");
            }
            for (int i = 0; i < MAX_PASS_DESCRIPTOR_SETS; i++) {
                descriptorSets1[i] = pSets1.get(i);
            }
        }
    }

    private long resolveBlockAtlasView() {
        if (blockAtlasImageView != VK_NULL_HANDLE) {
            return blockAtlasImageView;
        }
        VkPipelineManager pm = VkPipelineManager.getInstance();
        if (pm.isInitialized() && pm.getBlockAtlasView() != VK_NULL_HANDLE) {
            return pm.getBlockAtlasView();
        }
        return targets.getWhiteView();
    }

    private long resolveBlockAtlasSampler() {
        if (blockAtlasSampler != VK_NULL_HANDLE) {
            return blockAtlasSampler;
        }
        return targets.getNearestSampler();
    }

    private void updateDescriptorSet0() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long[] storageBuffers = {
                    megaBuffer.getMegaVertexBuffer().buffer,
                    megaBuffer.getChunkMetaBuffer().buffer,
                    megaBuffer.getIndirectCmdBuffer().buffer,
                    megaBuffer.getDrawCountBuffer().buffer,
                    megaBuffer.getVisibleChunkIdBuffer().buffer,
                    megaBuffer.getAtlasUvBuffer().buffer
            };

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(8, stack);

            // Binding 0: ShaderPackUBO
            VkDescriptorBufferInfo.Buffer uboInfo = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(uboBuffer)
                    .offset(0)
                    .range(ShaderUniformLayout.UBO_SIZE_BYTES);
            writes.get(0)
                    .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                    .dstSet(descriptorSet0)
                    .dstBinding(ShaderDescriptorBindings.BINDING_SHADER_PACK_UBO)
                    .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                    .descriptorCount(1)
                    .pBufferInfo(uboInfo);

            // Bindings 1..6: SSBOs
            for (int i = 0; i < storageBuffers.length; i++) {
                VkDescriptorBufferInfo.Buffer bInfo = VkDescriptorBufferInfo.calloc(1, stack)
                        .buffer(storageBuffers[i])
                        .offset(0)
                        .range(VK_WHOLE_SIZE);
                writes.get(i + 1)
                        .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                        .dstSet(descriptorSet0)
                        .dstBinding(i + 1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .pBufferInfo(bInfo);
            }

            // Binding 7: Block Atlas combined image sampler
            VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    .imageView(resolveBlockAtlasView())
                    .sampler(resolveBlockAtlasSampler());
            writes.get(7)
                    .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                    .dstSet(descriptorSet0)
                    .dstBinding(ShaderDescriptorBindings.BINDING_BLOCK_ATLAS)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .pImageInfo(imgInfo);

            vkUpdateDescriptorSets(ctx.getVkDevice(), writes, null);
        }
    }

    /**
     * Updates a dedicated Set 1 descriptor set slot with the current ping-pong read views for
     * {@code colortex0..19}, {@code depthtex0..2}, {@code shadowtex0..1}, {@code shadowcolor0..1},
     * {@code gtexture}, {@code lightmap}, {@code noisetex}, {@code normals}, and {@code specular}.
     */
    private long updateDescriptorSet1ForCurrentState(int slot) {
        int clampedSlot = Math.floorMod(slot, MAX_PASS_DESCRIPTOR_SETS);
        long dstSet = descriptorSets1[clampedSlot];

        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSet.Buffer writes =
                    VkWriteDescriptorSet.calloc(ShaderDescriptorBindings.SET1_BINDING_COUNT, stack);

            long linearSampler = targets.getLinearSampler();
            long nearestSampler = targets.getNearestSampler();
            long repeatSampler = targets.getRepeatLinearSampler();
            long shadowSampler = targets.getShadowSampler();

            // 0..19: colortex0..19 (current ping-pong read views)
            for (int i = 0; i < ShaderDescriptorBindings.COLORTEX_COUNT; i++) {
                bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_COLORTEX_BASE + i,
                        targets.getColorReadView(i), linearSampler);
            }
            // 8..10: depthtex0..2
            for (int i = 0; i < ShaderDescriptorBindings.DEPTHTEX_COUNT; i++) {
                bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_DEPTHTEX_BASE + i,
                        targets.getDepthView(i), nearestSampler);
            }
            // 11..12: shadowtex0..1
            for (int i = 0; i < ShaderDescriptorBindings.SHADOWTEX_COUNT; i++) {
                bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_SHADOWTEX_BASE + i,
                        targets.getShadowDepthView(i), shadowSampler);
            }
            // 13..14: shadowcolor0..1
            for (int i = 0; i < ShaderDescriptorBindings.SHADOWCOLOR_COUNT; i++) {
                bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_SHADOWCOLOR_BASE + i,
                        targets.getShadowColorView(i), linearSampler);
            }
            // 15: gtexture / texture / gcolor (block atlas)
            bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_GTEXTURE,
                    resolveBlockAtlasView(), resolveBlockAtlasSampler());

            // 16: lightmap (16x16 procedural block/sky lightmap)
            bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_LIGHTMAP,
                    targets.getLightmapView(), linearSampler);

            // 17: noisetex (procedural RGBA8 noise with repeat sampler)
            bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_NOISETEX,
                    targets.getNoiseView(), repeatSampler);

            // 18: normals (flat-normal 1x1 fallback)
            bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_NORMALS,
                    targets.getFlatNormalView(), nearestSampler);

            // 19: specular (black 1x1 fallback)
            bindSamplerSlot(stack, writes, dstSet, ShaderDescriptorBindings.BINDING_SPECULAR,
                    targets.getBlackView(), nearestSampler);

            vkUpdateDescriptorSets(ctx.getVkDevice(), writes, null);
        }
        return dstSet;
    }

    private static void bindSamplerSlot(
            MemoryStack stack,
            VkWriteDescriptorSet.Buffer writes,
            long dstSet,
            int binding,
            long imageView,
            long sampler) {
        VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(1, stack)
                .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .imageView(imageView)
                .sampler(sampler);
        writes.get(binding)
                .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                .dstSet(dstSet)
                .dstBinding(binding)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .pImageInfo(imgInfo);
    }

    /**
     * Converts any perspective projection matrix (whether standard OpenGL [-1, 1], Vulkan [0, 1], or Valkyrie Reversed-Z [1, 0])
     * into the canonical OpenGL [-1, 1] clip-space projection matrix expected by OptiFine/Iris {@code gbufferProjectionInverse}.
     */
    private static Matrix4f toOpenGlClipProjection(Matrix4f proj, float nearPlane, float farPlane) {
        Matrix4f gl = new Matrix4f(proj);
        float n = Math.max(1e-4f, nearPlane);
        float f = Math.max(n + 1.0f, farPlane);
        if (Math.abs(gl.m23() + 1.0f) < 1e-3f) {
            gl.m22(-(f + n) / (f - n));
            gl.m32(-(2.0f * f * n) / (f - n));
        }
        return gl;
    }

    /**
     * Converts an OpenGL [-1, 1] clip-space projection matrix to Vulkan Standard-Z [0, 1] depth
     * ({@code z_vk = 0.5 * z_gl + 0.5 * w_gl}), so hardware depth in {@code depthtex0} matches OpenGL's {@code [0, 1]} window depth.
     */
    private static Matrix4f openGlToVulkanStandardZ(Matrix4f glProj) {
        Matrix4f vk = new Matrix4f(glProj);
        vk.m02(0.5f * glProj.m02() + 0.5f * glProj.m03());
        vk.m12(0.5f * glProj.m12() + 0.5f * glProj.m13());
        vk.m22(0.5f * glProj.m22() + 0.5f * glProj.m23());
        vk.m32(0.5f * glProj.m32() + 0.5f * glProj.m33());
        return vk;
    }

    private void destroyPassPipelines() {
        for (CompiledPass cp : compiledPassesByName.values()) {
            if (cp != null && cp.getPipeline() != VK_NULL_HANDLE) {
                vkDestroyPipeline(ctx.getVkDevice(), cp.getPipeline(), null);
            }
        }
        compiledPassesByName.clear();
        preparePasses.clear();
        deferredPasses.clear();
        compositePasses.clear();
        shadowPass = null;
        gbuffersTerrainPass = null;
        gbuffersWaterPass = null;
        finalPass = null;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public boolean isPackReady() {
        return packReady;
    }

    public String getActivePackName() {
        return activePackName;
    }

    public ShaderPackProperties getActiveProperties() {
        return activeProperties;
    }

    public VkShaderPackTargets getTargets() {
        return targets;
    }

    public Map<String, CompiledPass> getCompiledPasses() {
        return Collections.unmodifiableMap(compiledPassesByName);
    }

    public CompiledPass getCompiledPass(String name) {
        return compiledPassesByName.get(name);
    }

    public List<String> getCompileWarnings() {
        return Collections.unmodifiableList(compileWarnings);
    }

    public long getPipelineLayout() {
        return pipelineLayout;
    }

    public long getDescriptorSet0() {
        return descriptorSet0;
    }

    public long getUboBuffer() {
        return uboBuffer;
    }

    public void resetTemporalState() {
        this.hasPrevFrameState = false;
        this.internalFrameCounter = 0;
        invalidateShadow();
    }

    private boolean shouldSkipStaticShadowPass(
            ShaderUniformLayout.UniformData u,
            boolean hasShadowPass,
            int desiredShadowCap) {
        if (!hasShadowPass || !shadowMapValid || shadowSceneDirty || u == null) {
            return false;
        }
        if (!org.sakus.aerogelvk.config.ValkyrieConfig.get().isShadowSkipStaticFrames()) {
            return false;
        }
        if (packLoadGeneration != prevShadowPackGeneration
                || targets.getShadowResolution() != prevShadowResolution
                || desiredShadowCap != prevShadowCap) {
            return false;
        }
        if (u.worldTime != prevShadowWorldTime
                || u.worldDay != prevShadowWorldDay
                || u.moonPhase != prevShadowMoonPhase) {
            return false;
        }
        if (Math.abs(u.shadowAngle - prevShadowAngle) > SHADOW_POS_EPSILON
                || Math.abs(u.sunAngle - prevShadowSunAngle) > SHADOW_POS_EPSILON
                || Math.abs(u.rainStrength - prevShadowRainStrength) > SHADOW_POS_EPSILON) {
            return false;
        }
        if (!vecWithinEpsilon(u.cameraPosition, prevShadowCamPos, SHADOW_POS_EPSILON)
                || !vecWithinEpsilon(u.sunPosition, prevShadowSunDir, SHADOW_POS_EPSILON)
                || !vecWithinEpsilon(u.shadowLightPosition, prevShadowLightPos, SHADOW_POS_EPSILON)) {
            return false;
        }
        return matrixWithinEpsilon(u.gbufferModelView, prevShadowModelView, SHADOW_MAT_EPSILON)
                && matrixWithinEpsilon(u.gbufferProjection, prevShadowProj, SHADOW_MAT_EPSILON);
    }

    private void recordShadowState(ShaderUniformLayout.UniformData u, int desiredShadowCap) {
        if (u == null) {
            shadowMapValid = false;
            return;
        }
        prevShadowCamPos.set(u.cameraPosition);
        prevShadowModelView.set(u.gbufferModelView);
        prevShadowProj.set(u.gbufferProjection);
        prevShadowSunDir.set(u.sunPosition);
        prevShadowLightPos.set(u.shadowLightPosition);
        prevShadowWorldTime = u.worldTime;
        prevShadowWorldDay = u.worldDay;
        prevShadowMoonPhase = u.moonPhase;
        prevShadowAngle = u.shadowAngle;
        prevShadowSunAngle = u.sunAngle;
        prevShadowRainStrength = u.rainStrength;
        prevShadowResolution = targets.getShadowResolution();
        prevShadowCap = desiredShadowCap;
        prevShadowPackGeneration = packLoadGeneration;
        shadowMapValid = true;
        shadowSceneDirty = false;
    }

    private static boolean vecWithinEpsilon(Vector3f a, Vector3f b, float eps) {
        return Math.abs(a.x - b.x) <= eps
                && Math.abs(a.y - b.y) <= eps
                && Math.abs(a.z - b.z) <= eps;
    }

    private static boolean matrixWithinEpsilon(Matrix4f a, Matrix4f b, float eps) {
        return Math.abs(a.m00() - b.m00()) <= eps && Math.abs(a.m01() - b.m01()) <= eps
                && Math.abs(a.m02() - b.m02()) <= eps && Math.abs(a.m03() - b.m03()) <= eps
                && Math.abs(a.m10() - b.m10()) <= eps && Math.abs(a.m11() - b.m11()) <= eps
                && Math.abs(a.m12() - b.m12()) <= eps && Math.abs(a.m13() - b.m13()) <= eps
                && Math.abs(a.m20() - b.m20()) <= eps && Math.abs(a.m21() - b.m21()) <= eps
                && Math.abs(a.m22() - b.m22()) <= eps && Math.abs(a.m23() - b.m23()) <= eps
                && Math.abs(a.m30() - b.m30()) <= eps && Math.abs(a.m31() - b.m31()) <= eps
                && Math.abs(a.m32() - b.m32()) <= eps && Math.abs(a.m33() - b.m33()) <= eps;
    }

    @Override
    public synchronized void close() {
        if (!initialized) {
            targets.close();
            return;
        }
        if (ctx.isInitialized() && ctx.getVkDevice() != null) {
            VkDevice device = ctx.getVkDevice();
            vkDeviceWaitIdle(device);
            destroyPassPipelines();

            if (cullPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, cullPipeline, null);
                cullPipeline = VK_NULL_HANDLE;
            }
            if (descriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptorPool, null);
                descriptorPool = VK_NULL_HANDLE;
            }
            if (pipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipelineLayout, null);
                pipelineLayout = VK_NULL_HANDLE;
            }
            if (descriptorSetLayout0 != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout0, null);
                descriptorSetLayout0 = VK_NULL_HANDLE;
            }
            if (descriptorSetLayout1 != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout1, null);
                descriptorSetLayout1 = VK_NULL_HANDLE;
            }
            if (uboBuffer != VK_NULL_HANDLE) {
                if (uboMappedPtr != MemoryUtil.NULL) {
                    Vma.vmaUnmapMemory(ctx.getVmaAllocator(), uboAllocation);
                    uboMappedPtr = MemoryUtil.NULL;
                }
                Vma.vmaDestroyBuffer(ctx.getVmaAllocator(), uboBuffer, uboAllocation);
                uboBuffer = VK_NULL_HANDLE;
                uboAllocation = VK_NULL_HANDLE;
            }
        }

        targets.close();
        Arrays.fill(descriptorSets1, VK_NULL_HANDLE);
        descriptorSet0 = VK_NULL_HANDLE;
        initialized = false;
        packReady = false;
    }
}
