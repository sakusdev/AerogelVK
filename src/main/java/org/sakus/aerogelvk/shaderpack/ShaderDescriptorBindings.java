package org.sakus.aerogelvk.shaderpack;

/**
 * Shared Vulkan descriptor set and binding constants for the native shaderpack pipeline.
 *
 * <p>Layout contract shared between {@link GlslToVulkanTranspiler}, {@link ShaderUniformLayout},
 * and {@code VkShaderPackPipeline}:
 * <ul>
 *     <li><b>Set 0 ({@link #SET_GLOBAL})</b>: ShaderPackUBO + Valkyrie vertex pulling SSBOs + Block Atlas</li>
 *     <li><b>Set 1 ({@link #SET_TEXTURES})</b>: OptiFine/Iris offscreen targets, shadow maps, and pack textures</li>
 * </ul>
 */
public final class ShaderDescriptorBindings {

    // ------------------------------------------------------------------------
    // Descriptor Set Indices
    // ------------------------------------------------------------------------

    public static final int SET_GLOBAL = 0;
    public static final int SET_TEXTURES = 1;

    // ------------------------------------------------------------------------
    // Set 0: Uniforms, Terrain SSBOs, and Block Atlas
    // ------------------------------------------------------------------------

    /** Uniform buffer containing {@link ShaderUniformLayout} ({@code std140}). */
    public static final int BINDING_SHADER_PACK_UBO = 0;

    /** 64-bit packed vertex pulling arena ({@code uvec2 vertices[]}). */
    public static final int BINDING_MEGA_VERTEX_SSBO = 1;

    /** Per-section draw metadata ({@code ChunkDrawMeta chunks[]}). */
    public static final int BINDING_CHUNK_META_SSBO = 2;

    /** Reserved indirect draw commands SSBO (shared slot with VkMegaBuffer). */
    public static final int BINDING_INDIRECT_CMD_SSBO = 3;

    /** Reserved indirect draw count SSBO (shared slot with VkMegaBuffer). */
    public static final int BINDING_DRAW_COUNT_SSBO = 4;

    /** Compacted visible instance -> chunk metadata index mapping ({@code uint instanceToChunkIdx[]}). */
    public static final int BINDING_VISIBLE_CHUNK_ID_SSBO = 5;

    /** Block atlas sprite UV and biome tint table ({@code AtlasUVEntry uvTable[]}). */
    public static final int BINDING_ATLAS_UV_SSBO = 6;

    /** Minecraft block texture atlas combined image sampler. */
    public static final int BINDING_BLOCK_ATLAS = 7;

    // ------------------------------------------------------------------------
    // Set 1: OptiFine / Iris Shaderpack Textures and Buffers
    // ------------------------------------------------------------------------

    /** Offscreen color render targets {@code colortex0}..{@code colortex19} (bindings 0..19). */
    public static final int BINDING_COLORTEX_BASE = 0;
    public static final int COLORTEX_COUNT = 20;

    /** Scene depth buffers {@code depthtex0}..{@code depthtex2} (bindings 20..22). */
    public static final int BINDING_DEPTHTEX_BASE = 20;
    public static final int DEPTHTEX_COUNT = 3;

    /** Shadow depth maps {@code shadowtex0}..{@code shadowtex1} (bindings 23..24). */
    public static final int BINDING_SHADOWTEX_BASE = 23;
    public static final int SHADOWTEX_COUNT = 2;

    /** Shadow color maps {@code shadowcolor0}..{@code shadowcolor1} (bindings 25..26). */
    public static final int BINDING_SHADOWCOLOR_BASE = 25;
    public static final int SHADOWCOLOR_COUNT = 2;

    /** Primary geometry texture ({@code gtexture} / {@code texture} -> {@code uBlockAtlas}). */
    public static final int BINDING_GTEXTURE = 27;

    /** Minecraft lightmap texture ({@code lightmap}). */
    public static final int BINDING_LIGHTMAP = 28;

    /** Procedural noise texture ({@code noisetex}). */
    public static final int BINDING_NOISETEX = 29;

    /** PBR normal map atlas ({@code normals}). */
    public static final int BINDING_NORMALS = 30;

    /** PBR specular map atlas ({@code specular}). */
    public static final int BINDING_SPECULAR = 31;

    /** Total number of combined image sampler bindings in Set 1 (bindings 0..31). */
    public static final int SET1_BINDING_COUNT = 32;

    // ------------------------------------------------------------------------
    // Reserved Varying Locations (Vertex -> Fragment)
    // ------------------------------------------------------------------------

    public static final int LOC_FRONT_COLOR = 0;
    public static final int LOC_TEXCOORD_BASE = 1; // occupies locations 1..4 (vec4[4])
    public static final int LOC_SPRITE_META = 5;   // flat uvec4(spriteId, quadW, quadH, active)
    public static final int LOC_USER_VARYING_BASE = 8;

    private ShaderDescriptorBindings() {
    }

    public static int colortexBinding(int index) {
        if (index < 0 || index >= COLORTEX_COUNT) {
            throw new IllegalArgumentException("colortex index out of range [0.." + (COLORTEX_COUNT - 1) + "]: " + index);
        }
        return BINDING_COLORTEX_BASE + index;
    }

    public static int depthtexBinding(int index) {
        if (index < 0 || index >= DEPTHTEX_COUNT) {
            throw new IllegalArgumentException("depthtex index out of range [0.." + (DEPTHTEX_COUNT - 1) + "]: " + index);
        }
        return BINDING_DEPTHTEX_BASE + index;
    }

    public static int shadowtexBinding(int index) {
        if (index < 0 || index >= SHADOWTEX_COUNT) {
            throw new IllegalArgumentException("shadowtex index out of range [0.." + (SHADOWTEX_COUNT - 1) + "]: " + index);
        }
        return BINDING_SHADOWTEX_BASE + index;
    }

    public static int shadowcolorBinding(int index) {
        if (index < 0 || index >= SHADOWCOLOR_COUNT) {
            throw new IllegalArgumentException("shadowcolor index out of range [0.." + (SHADOWCOLOR_COUNT - 1) + "]: " + index);
        }
        return BINDING_SHADOWCOLOR_BASE + index;
    }
}
