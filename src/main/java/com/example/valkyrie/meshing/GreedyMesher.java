package com.example.valkyrie.meshing;

import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.chunk.PalettedContainer;
import net.minecraft.world.level.lighting.LayerLightEventListener;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class GreedyMesher {
    public static final int SECTION_SIZE = 16;
    public static final int HALO_SIZE = 18;
    
    private static final ExecutorService EXECUTOR = Executors.newVirtualThreadPerTaskExecutor();
    private static final GreedyMesher INSTANCE = new GreedyMesher();
    private static final java.util.concurrent.ConcurrentLinkedQueue<MeshedSectionResult> COMPLETED_QUEUE = new java.util.concurrent.ConcurrentLinkedQueue<>();

    public static GreedyMesher getInstance() {
        return INSTANCE;
    }

    public void enqueue(SectionSnapshot snapshot) {
        if (snapshot == null) return;
        meshAsync(snapshot).thenAccept(COMPLETED_QUEUE::add);
    }

    public MeshedSectionResult pollCompleted() {
        return COMPLETED_QUEUE.poll();
    }

    public void clear() {
        COMPLETED_QUEUE.clear();
    }

    public void shutdown() {
        EXECUTOR.shutdownNow();
        COMPLETED_QUEUE.clear();
    }

    private GreedyMesher() {}
    
    public record SectionSnapshot(int sectionX, int sectionY, int sectionZ,
                                  BlockState[] blocks18x18x18,
                                  byte[] blockLight18x18x18,
                                  byte[] skyLight18x18x18) {
        
        public static SectionSnapshot capture(ClientLevel level, int sectionX, int sectionY, int sectionZ) {
            BlockState[] blocks = new BlockState[HALO_SIZE * HALO_SIZE * HALO_SIZE];
            byte[] blockLight = new byte[HALO_SIZE * HALO_SIZE * HALO_SIZE];
            byte[] skyLight = new byte[HALO_SIZE * HALO_SIZE * HALO_SIZE];
            
            LayerLightEventListener blockLightListener = level.getLightEngine().getLayerListener(LightLayer.BLOCK);
            LayerLightEventListener skyLightListener = level.getLightEngine().getLayerListener(LightLayer.SKY);
            
            int baseX = sectionX << 4;
            int baseY = sectionY << 4;
            int baseZ = sectionZ << 4;
            
            BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
            
            for (int y = 0; y < HALO_SIZE; y++) {
                for (int z = 0; z < HALO_SIZE; z++) {
                    for (int x = 0; x < HALO_SIZE; x++) {
                        int wx = baseX + x - 1;
                        int wy = baseY + y - 1;
                        int wz = baseZ + z - 1;
                        int idx = (y * HALO_SIZE + z) * HALO_SIZE + x;
                        
                        pos.set(wx, wy, wz);
                        
                        // Get block state
                        BlockState state;
                        if (wy < level.getMinBuildHeight() || wy >= level.getMaxBuildHeight()) {
                            state = net.minecraft.world.level.block.Blocks.AIR.defaultBlockState();
                        } else {
                            LevelChunk chunk = level.getChunk(SectionPos.blockToSectionCoord(wx), SectionPos.blockToSectionCoord(wz));
                            if (chunk == null) {
                                state = net.minecraft.world.level.block.Blocks.AIR.defaultBlockState();
                            } else {
                                LevelChunkSection section = chunk.getSection(chunk.getSectionIndex(wy));
                                if (section == null || section.hasOnlyAir()) {
                                    state = net.minecraft.world.level.block.Blocks.AIR.defaultBlockState();
                                } else {
                                    int lx = wx & 15;
                                    int ly = wy & 15;
                                    int lz = wz & 15;
                                    state = section.getBlockState(lx, ly, lz);
                                }
                            }
                        }
                        blocks[idx] = state;
                        
                        // Get light
                        int bl = 0;
                        int sl = 0;
                        if (wy >= level.getMinBuildHeight() && wy < level.getMaxBuildHeight()) {
                            try {
                                bl = blockLightListener.getLightValue(pos);
                                sl = skyLightListener.getLightValue(pos);
                            } catch (Exception e) {
                                bl = 0;
                                sl = 0;
                            }
                        }
                        blockLight[idx] = (byte) bl;
                        skyLight[idx] = (byte) sl;
                    }
                }
            }
            
            return new SectionSnapshot(sectionX, sectionY, sectionZ, blocks, blockLight, skyLight);
        }
    }
    
    public record MeshedSectionResult(int sectionX, int sectionY, int sectionZ,
                                      long[] packedVertices, int quadCount,
                                      int aabbPacked, int flags) {}
    
    private record FaceInfo(int spriteId, int materialMask, boolean windWaving,
                            int ao0, int ao1, int ao2, int ao3,
                            int blockLight, int skyLight) {}
    
    public static CompletableFuture<MeshedSectionResult> meshAsync(SectionSnapshot snapshot) {
        return CompletableFuture.supplyAsync(() -> mesh(snapshot), EXECUTOR);
    }
    
    public static MeshedSectionResult mesh(SectionSnapshot snapshot) {
        List<Long> vertices = new ArrayList<>();
        int quadCount = 0;
        boolean hasTranslucent = false;
        boolean dirtyShadow = false;
        
        int minX = 16, minY = 16, minZ = 16;
        int maxX = 0, maxY = 0, maxZ = 0;
        boolean anyQuad = false;
        
        BlockState[] blocks = snapshot.blocks18x18x18();
        byte[] blockLight = snapshot.blockLight18x18x18();
        byte[] skyLight = snapshot.skyLight18x18x18();
        
        AtlasUVTable atlas = AtlasUVTable.getInstance();
        
        // Process each of 6 directions
        for (int dir = 0; dir < 6; dir++) {
            // For each slice along the normal axis
            for (int slice = 0; slice < 16; slice++) {
                FaceInfo[][] mask = new FaceInfo[16][16];
                
                // Build mask
                for (int a = 0; a < 16; a++) {
                    for (int b = 0; b < 16; b++) {
                        int x, y, z;
                        switch (dir) {
                            case 0: x = slice; y = a; z = b; break; // +X
                            case 1: x = slice; y = a; z = b; break; // -X
                            case 2: x = a; y = slice; z = b; break; // +Y
                            case 3: x = a; y = slice; z = b; break; // -Y
                            case 4: x = a; y = b; z = slice; break; // +Z
                            case 5: x = a; y = b; z = slice; break; // -Z
                            default: continue;
                        }
                        
                        // Get block at (x,y,z) and neighbor
                        int idx = ((y + 1) * HALO_SIZE + (z + 1)) * HALO_SIZE + (x + 1);
                        BlockState state = blocks[idx];
                        
                        if (state.isAir()) continue;
                        
                        // Check if this face is visible
                        int nx = x, ny = y, nz = z;
                        switch (dir) {
                            case 0: nx = x + 1; break;
                            case 1: nx = x - 1; break;
                            case 2: ny = y + 1; break;
                            case 3: ny = y - 1; break;
                            case 4: nz = z + 1; break;
                            case 5: nz = z - 1; break;
                        }
                        
                        int nIdx = ((ny + 1) * HALO_SIZE + (nz + 1)) * HALO_SIZE + (nx + 1);
                        BlockState neighbor = blocks[nIdx];
                        
                        // Face is visible if neighbor is not opaque
                        if (isOpaqueFullCube(neighbor)) continue;
                        
                        // Also skip if this block is not opaque (e.g., water, glass)
                        // Actually, we should render faces of transparent blocks too
                        // But for simplicity, let's render all non-air blocks
                        
                        // Compute AO and light
                        int[] ao = new int[4];
                        computeAO(blocks, x, y, z, dir, ao);
                        
                        int bl = blockLight[nIdx] & 0xFF;
                        int sl = skyLight[nIdx] & 0xFF;
                        
                        int spriteId = atlas.getSpriteId(state, dir);
                        int materialMask = atlas.getMaterialMask(state);
                        boolean wind = atlas.isWindWaving(state);
                        
                        mask[a][b] = new FaceInfo(spriteId, materialMask, wind,
                                ao[0], ao[1], ao[2], ao[3], bl, sl);
                    }
                }
                
                // Greedy merge
                for (int a = 0; a < 16; a++) {
                    for (int b = 0; b < 16; b++) {
                        FaceInfo info = mask[a][b];
                        if (info == null) continue;
                        
                        // Find width
                        int w = 1;
                        while (a + w < 16 && info.equals(mask[a + w][b])) w++;
                        
                        // Find height
                        int h = 1;
                        outer:
                        while (b + h < 16) {
                            for (int k = 0; k < w; k++) {
                                if (!info.equals(mask[a + k][b + h])) break outer;
                            }
                            h++;
                        }
                        
                        // Emit quad
                        emitQuad(vertices, dir, slice, a, b, w, h, info);
                        quadCount++;
                        
                        // Update AABB
                        // Compute the 3D bounds of this quad
                        int[] bounds = computeQuadBounds(dir, slice, a, b, w, h);
                        if (!anyQuad) {
                            minX = bounds[0]; minY = bounds[1]; minZ = bounds[2];
                            maxX = bounds[3]; maxY = bounds[4]; maxZ = bounds[5];
                            anyQuad = true;
                        } else {
                            minX = Math.min(minX, bounds[0]);
                            minY = Math.min(minY, bounds[1]);
                            minZ = Math.min(minZ, bounds[2]);
                            maxX = Math.max(maxX, bounds[3]);
                            maxY = Math.max(maxY, bounds[4]);
                            maxZ = Math.max(maxZ, bounds[5]);
                        }
                        
                        // Check for translucent
                        if (info.materialMask() == 1) hasTranslucent = true;
                        
                        // Clear mask
                        for (int k = 0; k < w; k++) {
                            for (int l = 0; l < h; l++) {
                                mask[a + k][b + l] = null;
                            }
                        }
                    }
                }
            }
        }
        
        long[] packed = new long[vertices.size()];
        for (int i = 0; i < vertices.size(); i++) packed[i] = vertices.get(i);
        
        int aabbPacked = anyQuad ? VertexPacker.packTightAabb(minX, minY, minZ, maxX, maxY, maxZ) : 0;
        int flags = 0;
        if (hasTranslucent) flags |= 1;
        if (dirtyShadow) flags |= 2;
        
        return new MeshedSectionResult(snapshot.sectionX(), snapshot.sectionY(), snapshot.sectionZ(),
                packed, quadCount, aabbPacked, flags);
    }
    
    private static boolean isOpaqueFullCube(BlockState state) {
        if (state.isAir()) return false;
        return state.isSolidRender(null, null);
    }
    
    private static void computeAO(BlockState[] blocks, int x, int y, int z, int dir, int[] ao) {
        // For each corner, check 3 neighbors
        // Corner order: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1) in (u,v) space
        // u,v depend on direction
        
        int nx = x, ny = y, nz = z;
        switch (dir) {
            case 0: nx = x + 1; break;
            case 1: nx = x - 1; break;
            case 2: ny = y + 1; break;
            case 3: ny = y - 1; break;
            case 4: nz = z + 1; break;
            case 5: nz = z - 1; break;
        }
        
        // u and v axes
        int ux = 0, uy = 0, uz = 0;
        int vx = 0, vy = 0, vz = 0;
        switch (dir) {
            case 0: case 1: ux = 0; uy = 1; uz = 0; vx = 0; vy = 0; vz = 1; break; // u=y, v=z
            case 2: case 3: ux = 1; uy = 0; uz = 0; vx = 0; vy = 0; vz = 1; break; // u=x, v=z
            case 4: case 5: ux = 1; uy = 0; uz = 0; vx = 0; vy = 1; vz = 0; break; // u=x, v=y
        }
        
        for (int c = 0; c < 4; c++) {
            int cu = (c == 0 || c == 3) ? -1 : 1;
            int cv = (c == 0 || c == 1) ? -1 : 1;
            
            int s1x = nx + cu * ux;
            int s1y = ny + cu * uy;
            int s1z = nz + cu * uz;
            
            int s2x = nx + cv * vx;
            int s2y = ny + cv * vy;
            int s2z = nz + cv * vz;
            
            int ccx = nx + cu * ux + cv * vx;
            int ccy = ny + cu * uy + cv * vy;
            int ccz = nz + cu * uz + cv * vz;
            
            boolean s1 = isSolidAt(blocks, s1x, s1y, s1z);
            boolean s2 = isSolidAt(blocks, s2x, s2y, s2z);
            boolean cc = isSolidAt(blocks, ccx, ccy, ccz);
            
            int aoVal;
            if (s1 && s2) aoVal = 0;
            else aoVal = 3 - ((s1 ? 1 : 0) + (s2 ? 1 : 0) + (cc ? 1 : 0));
            ao[c] = aoVal;
        }
    }
    
    private static boolean isSolidAt(BlockState[] blocks, int x, int y, int z) {
        if (x < -1 || x > 16 || y < -1 || y > 16 || z < -1 || z > 16) return false;
        int idx = ((y + 1) * HALO_SIZE + (z + 1)) * HALO_SIZE + (x + 1);
        BlockState s = blocks[idx];
        return s != null && !s.isAir() && s.isSolidRender(null, null);
    }
    
    private static void emitQuad(List<Long> vertices, int dir, int slice, int a, int b, int w, int h, FaceInfo info) {
        // Compute 4 corner positions
        int[][] corners = new int[4][3];
        
        for (int c = 0; c < 4; c++) {
            int ca = (c == 0 || c == 3) ? a : a + w;
            int cb = (c == 0 || c == 1) ? b : b + h;
            
            int x, y, z;
            switch (dir) {
                case 0: x = slice + 1; y = ca; z = cb; break;
                case 1: x = slice; y = ca; z = cb; break;
                case 2: x = ca; y = slice + 1; z = cb; break;
                case 3: x = ca; y = slice; z = cb; break;
                case 4: x = ca; y = cb; z = slice + 1; break;
                case 5: x = ca; y = cb; z = slice; break;
                default: x = 0; y = 0; z = 0;
            }
            corners[c][0] = x;
            corners[c][1] = y;
            corners[c][2] = z;
        }
        
        int[] ao = {info.ao0(), info.ao1(), info.ao2(), info.ao3()};
        
        for (int c = 0; c < 4; c++) {
            long packed = VertexPacker.pack(
                    corners[c][0], corners[c][1], corners[c][2],
                    dir, ao[c], info.blockLight(), info.skyLight(),
                    info.windWaving(), info.spriteId(), w, h, c, info.materialMask()
            );
            vertices.add(packed);
        }
    }
    
    private static int[] computeQuadBounds(int dir, int slice, int a, int b, int w, int h) {
        int minX, minY, minZ, maxX, maxY, maxZ;
        switch (dir) {
            case 0: // +X
                minX = slice + 1; maxX = slice + 1;
                minY = a; maxY = a + h;
                minZ = b; maxZ = b + w;
                break;
            case 1: // -X
                minX = slice; maxX = slice;
                minY = a; maxY = a + h;
                minZ = b; maxZ = b + w;
                break;
            case 2: // +Y
                minX = a; maxX = a + w;
                minY = slice + 1; maxY = slice + 1;
                minZ = b; maxZ = b + h;
                break;
            case 3: // -Y
                minX = a; maxX = a + w;
                minY = slice; maxY = slice;
                minZ = b; maxZ = b + h;
                break;
            case 4: // +Z
                minX = a; maxX = a + w;
                minY = b; maxY = b + h;
                minZ = slice + 1; maxZ = slice + 1;
                break;
            case 5: // -Z
                minX = a; maxX = a + w;
                minY = b; maxY = b + h;
                minZ = slice; maxZ = slice;
                break;
            default:
                minX = minY = minZ = maxX = maxY = maxZ = 0;
        }
        return new int[]{minX, minY, minZ, maxX, maxY, maxZ};
    }
}
