package com.example.valkyrie.meshing;

import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.block.state.BlockState;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 6-direction Greedy Mesher executing on Java 21 Virtual Threads.
 * Merges coplanar quad faces inside 16x16x16 chunk sections while respecting
 * AO, light levels, sprite IDs, and material masks to avoid interpolation artifacts.
 */
public final class GreedyMesher {

    private static final GreedyMesher INSTANCE = new GreedyMesher();

    public static GreedyMesher getInstance() {
        return INSTANCE;
    }

    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final ConcurrentLinkedQueue<MeshedSectionResult> completedQueue = new ConcurrentLinkedQueue<>();

    private GreedyMesher() {
    }

    /**
     * Snapshot of a 16x16x16 chunk section plus 1-block halo (18x18x18 total voxels).
     */
    public record SectionSnapshot(
            int sectionX,
            int sectionY,
            int sectionZ,
            BlockState[] blocks18x18x18,
            byte[] blockLight18x18x18,
            byte[] skyLight18x18x18
    ) {
        public static int getIndex(int lx, int ly, int lz) {
            // lx, ly, lz in [0..17]
            return lx + lz * 18 + ly * 324; // 18 * 18 = 324
        }

        public static SectionSnapshot capture(ClientLevel level, int sectionX, int sectionY, int sectionZ) {
            int originX = sectionX << 4;
            int originY = sectionY << 4;
            int originZ = sectionZ << 4;

            BlockState[] blocks = new BlockState[18 * 18 * 18];
            byte[] blockLight = new byte[18 * 18 * 18];
            byte[] skyLight = new byte[18 * 18 * 18];

            BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();

            for (int ly = 0; ly < 18; ly++) {
                int wy = originY + ly - 1;
                for (int lz = 0; lz < 18; lz++) {
                    int wz = originZ + lz - 1;
                    for (int lx = 0; lx < 18; lx++) {
                        int wx = originX + lx - 1;
                        pos.set(wx, wy, wz);
                        int idx = getIndex(lx, ly, lz);

                        BlockState state = level.getBlockState(pos);
                        blocks[idx] = state;
                        blockLight[idx] = (byte) level.getBrightness(LightLayer.BLOCK, pos);
                        skyLight[idx] = (byte) level.getBrightness(LightLayer.SKY, pos);
                    }
                }
            }

            return new SectionSnapshot(sectionX, sectionY, sectionZ, blocks, blockLight, skyLight);
        }
    }

    /**
     * Meshing result ready for upload to the Vulkan Mega-Buffer.
     */
    public record MeshedSectionResult(
            int sectionX,
            int sectionY,
            int sectionZ,
            long[] packedVertices,
            int quadCount,
            int aabbPacked,
            int flags
    ) {}

    public void enqueue(SectionSnapshot snapshot) {
        if (snapshot == null) return;
        executor.submit(() -> {
            try {
                MeshedSectionResult result = meshSection(snapshot);
                if (result != null) {
                    completedQueue.add(result);
                }
            } catch (Throwable t) {
                t.printStackTrace();
            }
        });
    }

    public MeshedSectionResult pollCompleted() {
        return completedQueue.poll();
    }

    public void clear() {
        completedQueue.clear();
    }

    public void shutdown() {
        executor.shutdownNow();
        completedQueue.clear();
    }

    // Temporary storage for face masks during meshing
    private static class FaceData {
        int spriteId;
        int materialMask;
        boolean windWaving;
        int ao0, ao1, ao2, ao3;
        int bl0, bl1, bl2, bl3;
        int sl0, sl1, sl2, sl3;

        boolean matches(FaceData o) {
            if (o == null) return false;
            return this.spriteId == o.spriteId
                    && this.materialMask == o.materialMask
                    && this.windWaving == o.windWaving
                    && this.ao0 == o.ao0 && this.ao1 == o.ao1 && this.ao2 == o.ao2 && this.ao3 == o.ao3
                    && this.bl0 == o.bl0 && this.bl1 == o.bl1 && this.bl2 == o.bl2 && this.bl3 == o.bl3
                    && this.sl0 == o.sl0 && this.sl1 == o.sl1 && this.sl2 == o.sl2 && this.sl3 == o.sl3;
        }
    }

    private MeshedSectionResult meshSection(SectionSnapshot snapshot) {
        AtlasUVTable uvTable = AtlasUVTable.getInstance();
        BlockState[] blocks = snapshot.blocks18x18x18();
        byte[] bLight = snapshot.blockLight18x18x18();
        byte[] sLight = snapshot.skyLight18x18x18();

        int minX = 16, minY = 16, minZ = 16;
        int maxX = 0, maxY = 0, maxZ = 0;
        int flags = 0;

        // Estimated vertex buffer capacity
        long[] vertexBuffer = new long[16384];
        int vertexCount = 0;
        int quadCount = 0;

        FaceData[][] mask = new FaceData[16][16];

        // 6 Directions: 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
        int[] dx = { 1, -1,  0,  0,  0,  0 };
        int[] dy = { 0,  0,  1, -1,  0,  0 };
        int[] dz = { 0,  0,  0,  0,  1, -1 };

        for (int normal = 0; normal < 6; normal++) {
            int ndx = dx[normal];
            int ndy = dy[normal];
            int ndz = dz[normal];

            // Primary axis slice: 0..15
            for (int slice = 0; slice < 16; slice++) {
                // Clear 16x16 mask
                for (int u = 0; u < 16; u++) {
                    for (int v = 0; v < 16; v++) {
                        mask[u][v] = null;
                    }
                }

                boolean sliceHasFaces = false;

                for (int v = 0; v < 16; v++) {
                    for (int u = 0; u < 16; u++) {
                        int lx, ly, lz;
                        if (normal < 2) {
                            // X slices: u = Z, v = Y
                            lx = slice + 1;
                            ly = v + 1;
                            lz = u + 1;
                        } else if (normal < 4) {
                            // Y slices: u = X, v = Z
                            lx = u + 1;
                            ly = slice + 1;
                            lz = v + 1;
                        } else {
                            // Z slices: u = X, v = Y
                            lx = u + 1;
                            ly = v + 1;
                            lz = slice + 1;
                        }

                        BlockState current = blocks[SectionSnapshot.getIndex(lx, ly, lz)];
                        if (current == null || current.isAir()) continue;

                        int nlx = lx + ndx;
                        int nly = ly + ndy;
                        int nlz = lz + ndz;
                        BlockState neighbor = blocks[SectionSnapshot.getIndex(nlx, nly, nlz)];

                        // Cull if neighbor is fully opaque cube
                        if (neighbor != null && !neighbor.isAir() && neighbor.canOcclude()) {
                            continue;
                        }

                        FaceData fd = new FaceData();
                        fd.spriteId = uvTable.getSpriteId(current);
                        fd.materialMask = uvTable.getMaterialMask(current);
                        fd.windWaving = uvTable.isWindWaving(current);

                        if (fd.materialMask == VertexPacker.MAT_WATER) {
                            flags |= 1; // Bit 0: HasTranslucent / Water
                        }

                        // Sample light and AO from neighbor cell
                        int nIdx = SectionSnapshot.getIndex(nlx, nly, nlz);
                        int bl = bLight[nIdx] & 0xFF;
                        int sl = sLight[nIdx] & 0xFF;

                        fd.bl0 = fd.bl1 = fd.bl2 = fd.bl3 = Math.min(15, bl);
                        fd.sl0 = fd.sl1 = fd.sl2 = fd.sl3 = Math.min(15, sl);
                        fd.ao0 = fd.ao1 = fd.ao2 = fd.ao3 = 3; // Full ambient

                        mask[u][v] = fd;
                        sliceHasFaces = true;
                    }
                }

                if (!sliceHasFaces) continue;

                // Greedy Quad Merging on 16x16 mask
                for (int v = 0; v < 16; v++) {
                    for (int u = 0; u < 16; u++) {
                        FaceData fd = mask[u][v];
                        if (fd == null) continue;

                        // Find width
                        int width = 1;
                        while (u + width < 16 && fd.matches(mask[u + width][v])) {
                            width++;
                        }

                        // Find height
                        int height = 1;
                        boolean canExpandH = true;
                        while (v + height < 16 && canExpandH) {
                            for (int w = 0; w < width; w++) {
                                if (!fd.matches(mask[u + w][v + height])) {
                                    canExpandH = false;
                                    break;
                                }
                            }
                            if (canExpandH) height++;
                        }

                        // Clear merged cells
                        for (int h = 0; h < height; h++) {
                            for (int w = 0; w < width; w++) {
                                mask[u + w][v + h] = null;
                            }
                        }

                        // Compute section-space coordinates for 4 corners
                        int x0, y0, z0, x1, y1, z1;
                        if (normal < 2) {
                            // X-plane: slice, u=Z, v=Y
                            int fx = (normal == 0) ? (slice + 1) : slice;
                            x0 = x1 = fx;
                            z0 = u;
                            z1 = u + width;
                            y0 = v;
                            y1 = v + height;
                        } else if (normal < 4) {
                            // Y-plane: slice, u=X, v=Z
                            int fy = (normal == 2) ? (slice + 1) : slice;
                            y0 = y1 = fy;
                            x0 = u;
                            x1 = u + width;
                            z0 = v;
                            z1 = v + height;
                        } else {
                            // Z-plane: slice, u=X, v=Y
                            int fz = (normal == 4) ? (slice + 1) : slice;
                            z0 = z1 = fz;
                            x0 = u;
                            x1 = u + width;
                            y0 = v;
                            y1 = v + height;
                        }

                        // Update tight AABB bounds
                        minX = Math.min(minX, Math.min(x0, x1));
                        minY = Math.min(minY, Math.min(y0, y1));
                        minZ = Math.min(minZ, Math.min(z0, z1));
                        maxX = Math.max(maxX, Math.max(x0, x1));
                        maxY = Math.max(maxY, Math.max(y0, y1));
                        maxZ = Math.max(maxZ, Math.max(z0, z1));

                        // Ensure buffer capacity (4 vertices per quad)
                        if (vertexCount + 4 > vertexBuffer.length) {
                            long[] newBuf = new long[vertexBuffer.length * 2];
                            System.arraycopy(vertexBuffer, 0, newBuf, 0, vertexCount);
                            vertexBuffer = newBuf;
                        }

                        // Emit 4 vertices
                        // Corner 0: (x0, y0, z0)
                        // Corner 1: depending on orientation
                        int c0x = x0, c0y = y0, c0z = z0;
                        int c1x, c1y, c1z, c2x, c2y, c2z, c3x, c3y, c3z;

                        if (normal == 0) { // +X
                            c0x = x0; c0y = y0; c0z = z0;
                            c1x = x0; c1y = y0; c1z = z1;
                            c2x = x0; c2y = y1; c2z = z1;
                            c3x = x0; c3y = y1; c3z = z0;
                        } else if (normal == 1) { // -X
                            c0x = x0; c0y = y0; c0z = z1;
                            c1x = x0; c1y = y0; c1z = z0;
                            c2x = x0; c2y = y1; c2z = z0;
                            c3x = x0; c3y = y1; c3z = z1;
                        } else if (normal == 2) { // +Y
                            c0x = x0; c0y = y0; c0z = z1;
                            c1x = x1; c1y = y0; c1z = z1;
                            c2x = x1; c2y = y0; c2z = z0;
                            c3x = x0; c3y = y0; c3z = z0;
                        } else if (normal == 3) { // -Y
                            c0x = x0; c0y = y0; c0z = z0;
                            c1x = x1; c1y = y0; c1z = z0;
                            c2x = x1; c2y = y0; c2z = z1;
                            c3x = x0; c3y = y0; c3z = z1;
                        } else if (normal == 4) { // +Z
                            c0x = x1; c0y = y0; c0z = z0;
                            c1x = x0; c1y = y0; c1z = z0;
                            c2x = x0; c2y = y1; c2z = z0;
                            c3x = x1; c3y = y1; c3z = z0;
                        } else { // -Z
                            c0x = x0; c0y = y0; c0z = z0;
                            c1x = x1; c1y = y0; c1z = z0;
                            c2x = x1; c2y = y1; c2z = z0;
                            c3x = x0; c3y = y1; c3z = z0;
                        }

                        vertexBuffer[vertexCount++] = VertexPacker.pack(
                                c0x, c0y, c0z, normal, fd.ao0, fd.bl0, fd.sl0, fd.windWaving,
                                fd.spriteId, width, height, 0, fd.materialMask);
                        vertexBuffer[vertexCount++] = VertexPacker.pack(
                                c1x, c1y, c1z, normal, fd.ao1, fd.bl1, fd.sl1, fd.windWaving,
                                fd.spriteId, width, height, 1, fd.materialMask);
                        vertexBuffer[vertexCount++] = VertexPacker.pack(
                                c2x, c2y, c2z, normal, fd.ao2, fd.bl2, fd.sl2, fd.windWaving,
                                fd.spriteId, width, height, 2, fd.materialMask);
                        vertexBuffer[vertexCount++] = VertexPacker.pack(
                                c3x, c3y, c3z, normal, fd.ao3, fd.bl3, fd.sl3, fd.windWaving,
                                fd.spriteId, width, height, 3, fd.materialMask);

                        quadCount++;
                    }
                }
            }
        }

        if (quadCount == 0) {
            return new MeshedSectionResult(
                    snapshot.sectionX(), snapshot.sectionY(), snapshot.sectionZ(),
                    new long[0], 0, 0, flags);
        }

        long[] finalVertices = new long[vertexCount];
        System.arraycopy(vertexBuffer, 0, finalVertices, 0, vertexCount);

        int aabbPacked = VertexPacker.packTightAabb(minX, minY, minZ, maxX, maxY, maxZ);
        flags |= 2; // Bit 1: DirtyShadow

        return new MeshedSectionResult(
                snapshot.sectionX(), snapshot.sectionY(), snapshot.sectionZ(),
                finalVertices, quadCount, aabbPacked, flags);
    }
}
