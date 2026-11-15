package org.sakus.aerogelvk.meshing;

/**
 * 128-bit (two 64-bit uvec2 slots) dedicated vertex format for 1/16-block precision
 * non-full-cube VoxelShape geometry (subKind 2 & 3, ChunkDrawMeta.flags bit 2 set).
 *
 * <p>Cube greedy meshing continues to use the unchanged 64-bit {@link VertexPacker} format.</p>
 *
 * <pre>
 * Word 0 (low 64 bits -> first uvec2):
 *   w0.x [0..8]   : posX16       (0..256, 1/16 block units within section)
 *   w0.x [9..17]  : posY16       (0..256)
 *   w0.x [18..26] : posZ16       (0..256)
 *   w0.x [27..29] : normalDir    (0..5: +X, -X, +Y, -Y, +Z, -Z)
 *   w0.x [30..31] : aoLevel      (0..3)
 *   w0.y [0..3]   : blockLight   (0..15)
 *   w0.y [4..7]   : skyLight     (0..15)
 *   w0.y [8..11]  : materialMask (0..15)
 *   w0.y [12]     : windWaving   (0..1)
 *   w0.y [13..17] : u16          (0..16, sub-rect U in 1/16 sprite units)
 *   w0.y [18..22] : v16          (0..16, sub-rect V in 1/16 sprite units)
 *   w0.y [23..31] : reserved     (0)
 *
 * Word 1 (high 64 bits -> second uvec2):
 *   w1.x [0..31]  : spriteId     (32-bit AtlasUVTable index)
 *   w1.y [0..3]   : subX4        (0..15, 1/256-block fractional X)
 *   w1.y [4..7]   : subY4        (0..15, 1/256-block fractional Y)
 *   w1.y [8..11]  : subZ4        (0..15, 1/256-block fractional Z)
 *   w1.y [12..31] : reserved     (0)
 * </pre>
 */
public final class VkShapeVertex {
    private VkShapeVertex() {}

    public static long packWord0(
            int posX16, int posY16, int posZ16,
            int normalDir, int aoLevel,
            int blockLight, int skyLight, int materialMask, boolean windWaving,
            int u16, int v16
    ) {
        int lo = (posX16 & 0x1FF)
                | ((posY16 & 0x1FF) << 9)
                | ((posZ16 & 0x1FF) << 18)
                | ((normalDir & 0x7) << 27)
                | ((aoLevel & 0x3) << 30);

        int hi = (blockLight & 0xF)
                | ((skyLight & 0xF) << 4)
                | ((materialMask & 0xF) << 8)
                | ((windWaving ? 1 : 0) << 12)
                | ((u16 & 0x1F) << 13)
                | ((v16 & 0x1F) << 18);

        return ((long) hi << 32) | (lo & 0xFFFFFFFFL);
    }

    public static long packWord1(int spriteId) {
        return packWord1(spriteId, 0, 0, 0);
    }

    public static long packWord1(int spriteId, int subX4, int subY4, int subZ4) {
        int hi = (subX4 & 0xF) | ((subY4 & 0xF) << 4) | ((subZ4 & 0xF) << 8);
        return ((long) hi << 32) | (spriteId & 0xFFFFFFFFL);
    }
}
