package com.example.valkyrie.meshing;

/**
 * Packs vertex attributes into a single 64-bit value (GLSL: uvec2(dataLow, dataHigh)).
 *
 * dataLow (lower 32 bits):
 *  [0..5]   X position (6 bits)
 *  [6..11]  Y position (6 bits)
 *  [12..17] Z position (6 bits)
 *  [18..20] Face Normal Enum (3 bits)
 *  [21..22] AO level (2 bits)
 *  [23..26] Block Light (4 bits)
 *  [27..30] Sky Light (4 bits)
 *  [31]     Wind Waving Flag (1 bit)
 *
 * dataHigh (upper 32 bits):
 *  [0..15]  Texture Atlas Sprite ID (16 bits)
 *  [16..20] Greedy Quad Width (5 bits)
 *  [21..25] Greedy Quad Height (5 bits)
 *  [26..27] Corner Index (2 bits)
 *  [28..31] Material Mask (4 bits)
 */
public final class VertexPacker {

    // ---- Face Normal Enum ----
    public static final int NORM_POS_X = 0;
    public static final int NORM_NEG_X = 1;
    public static final int NORM_POS_Y = 2;
    public static final int NORM_NEG_Y = 3;
    public static final int NORM_POS_Z = 4;
    public static final int NORM_NEG_Z = 5;

    // ---- Material Mask ----
    public static final int MAT_MATTE = 0;
    public static final int MAT_WATER = 1;
    public static final int MAT_METALLIC = 2;
    public static final int MAT_EMISSIVE = 3;
    public static final int MAT_FOLIAGE = 4;

    // ---- dataLow bit layout ----
    private static final int X_SHIFT = 0;
    private static final int Y_SHIFT = 6;
    private static final int Z_SHIFT = 12;
    private static final int NORMAL_SHIFT = 18;
    private static final int AO_SHIFT = 21;
    private static final int BLOCK_LIGHT_SHIFT = 23;
    private static final int SKY_LIGHT_SHIFT = 27;
    private static final int WIND_SHIFT = 31;

    private static final long X_MASK = 0x3FL;
    private static final long Y_MASK = 0x3FL;
    private static final long Z_MASK = 0x3FL;
    private static final long NORMAL_MASK = 0x7L;
    private static final long AO_MASK = 0x3L;
    private static final long LIGHT_MASK = 0xFL;
    private static final long WIND_MASK = 0x1L;

    // ---- dataHigh bit layout ----
    private static final int SPRITE_SHIFT = 0;
    private static final int WIDTH_SHIFT = 16;
    private static final int HEIGHT_SHIFT = 21;
    private static final int CORNER_SHIFT = 26;
    private static final int MATERIAL_SHIFT = 28;

    private static final long SPRITE_MASK = 0xFFFFL;
    private static final long SIZE_MASK = 0x1FL;
    private static final long CORNER_MASK = 0x3L;
    private static final long MATERIAL_MASK = 0xFL;

    private VertexPacker() {
    }

    /**
     * Packs all 13 vertex attributes into a single 64-bit value.
     */
    public static long pack(int x, int y, int z, int normal, int ao, int blockLight, int skyLight,
                            boolean windWaving, int spriteId, int width, int height, int cornerIndex,
                            int materialMask) {
        long dataLow = ((long) (x & 0x3F) << X_SHIFT)
                | ((long) (y & 0x3F) << Y_SHIFT)
                | ((long) (z & 0x3F) << Z_SHIFT)
                | ((long) (normal & 0x7) << NORMAL_SHIFT)
                | ((long) (ao & 0x3) << AO_SHIFT)
                | ((long) (blockLight & 0xF) << BLOCK_LIGHT_SHIFT)
                | ((long) (skyLight & 0xF) << SKY_LIGHT_SHIFT)
                | ((long) (windWaving ? 1 : 0) << WIND_SHIFT);

        long dataHigh = ((long) (spriteId & 0xFFFF) << SPRITE_SHIFT)
                | ((long) (width & 0x1F) << WIDTH_SHIFT)
                | ((long) (height & 0x1F) << HEIGHT_SHIFT)
                | ((long) (cornerIndex & 0x3) << CORNER_SHIFT)
                | ((long) (materialMask & 0xF) << MATERIAL_SHIFT);

        return (dataHigh << 32) | dataLow;
    }

    /**
     * Extracts the lower 32 bits (dataLow) as an int.
     */
    public static int unpackDataLow(long packed) {
        return (int) (packed & 0xFFFFFFFFL);
    }

    /**
     * Extracts the upper 32 bits (dataHigh) as an int.
     */
    public static int unpackDataHigh(long packed) {
        return (int) (packed >>> 32);
    }

    // ---- dataLow getters ----

    public static int unpackX(long packed) {
        return (int) ((packed >>> X_SHIFT) & X_MASK);
    }

    public static int unpackY(long packed) {
        return (int) ((packed >>> Y_SHIFT) & Y_MASK);
    }

    public static int unpackZ(long packed) {
        return (int) ((packed >>> Z_SHIFT) & Z_MASK);
    }

    public static int unpackNormal(long packed) {
        return (int) ((packed >>> NORMAL_SHIFT) & NORMAL_MASK);
    }

    public static int unpackAo(long packed) {
        return (int) ((packed >>> AO_SHIFT) & AO_MASK);
    }

    public static int unpackBlockLight(long packed) {
        return (int) ((packed >>> BLOCK_LIGHT_SHIFT) & LIGHT_MASK);
    }

    public static int unpackSkyLight(long packed) {
        return (int) ((packed >>> SKY_LIGHT_SHIFT) & LIGHT_MASK);
    }

    public static boolean unpackWindWaving(long packed) {
        return ((packed >>> WIND_SHIFT) & WIND_MASK) != 0L;
    }

    // ---- dataHigh getters ----

    public static int unpackSpriteId(long packed) {
        return (int) ((packed >>> (32 + SPRITE_SHIFT)) & SPRITE_MASK);
    }

    public static int unpackWidth(long packed) {
        return (int) ((packed >>> (32 + WIDTH_SHIFT)) & SIZE_MASK);
    }

    public static int unpackHeight(long packed) {
        return (int) ((packed >>> (32 + HEIGHT_SHIFT)) & SIZE_MASK);
    }

    public static int unpackCornerIndex(long packed) {
        return (int) ((packed >>> (32 + CORNER_SHIFT)) & CORNER_MASK);
    }

    public static int unpackMaterialMask(long packed) {
        return (int) ((packed >>> (32 + MATERIAL_SHIFT)) & MATERIAL_MASK);
    }

    // ---- ChunkDrawMeta.aabbPacked ----

    /**
     * Packs a tight AABB (coordinates 0..31, 5 bits each) for ChunkDrawMeta.aabbPacked.
     * Layout: [0..4] minX, [5..9] minY, [10..14] minZ, [15..19] maxX, [20..24] maxY, [25..29] maxZ.
     */
    public static int packTightAabb(int minX, int minY, int minZ, int maxX, int maxY, int maxZ) {
        return (minX & 0x1F)
                | ((minY & 0x1F) << 5)
                | ((minZ & 0x1F) << 10)
                | ((maxX & 0x1F) << 15)
                | ((maxY & 0x1F) << 20)
                | ((maxZ & 0x1F) << 25);
    }

    /**
     * Unpacks the minimum corner of a packed AABB into a 3-element array {x, y, z}.
     */
    public static int[] unpackAabbMin(int aabbPacked) {
        return new int[]{
                aabbPacked & 0x1F,
                (aabbPacked >>> 5) & 0x1F,
                (aabbPacked >>> 10) & 0x1F
        };
    }

    /**
     * Unpacks the maximum corner of a packed AABB into a 3-element array {x, y, z}.
     */
    public static int[] unpackAabbMax(int aabbPacked) {
        return new int[]{
                (aabbPacked >>> 15) & 0x1F,
                (aabbPacked >>> 20) & 0x1F,
                (aabbPacked >>> 25) & 0x1F
        };
    }
}
