package org.sakus.aerogelvk.meshing;

import net.minecraft.client.Minecraft;
import net.minecraft.client.color.block.BlockColors;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.BiomeColors;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;
import net.minecraft.world.level.EmptyBlockGetter;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.RenderShape;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.lighting.LayerLightEventListener;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.shapes.VoxelShape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class GreedyMesher {
    public static final int SECTION_SIZE = 16;
    public static final int HALO_SIZE = 18;

    private static ExecutorService newWorkerPool() {
        int workers = Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors() / 2));
        return Executors.newFixedThreadPool(workers, r -> {
            Thread t = new Thread(r, "valkyrie-mesher");
            t.setDaemon(true);
            return t;
        });
    }

    private static volatile ExecutorService executor = newWorkerPool();
    private static final GreedyMesher INSTANCE = new GreedyMesher();
    private static final ConcurrentHashMap<Long, MeshedSectionResult> COMPLETED_MAP = new ConcurrentHashMap<>();

    private static synchronized ExecutorService ensureExecutor() {
        if (executor == null || executor.isShutdown()) {
            executor = newWorkerPool();
        }
        return executor;
    }

    public static GreedyMesher getInstance() {
        return INSTANCE;
    }

    public void enqueue(SectionSnapshot snapshot) {
        if (snapshot == null) return;
        long key = SectionPos.asLong(snapshot.sectionX(), snapshot.sectionY(), snapshot.sectionZ());
        meshAsync(snapshot).thenAccept(result -> {
            if (result != null) {
                COMPLETED_MAP.put(key, result);
            }
        });
    }

    public MeshedSectionResult pollCompleted() {
        for (Long key : COMPLETED_MAP.keySet()) {
            MeshedSectionResult result = COMPLETED_MAP.remove(key);
            if (result != null) {
                return result;
            }
        }
        return null;
    }

    public void clear() {
        COMPLETED_MAP.clear();
    }

    public synchronized void shutdown() {
        if (executor != null) {
            executor.shutdownNow();
        }
        COMPLETED_MAP.clear();
    }

    private GreedyMesher() {}
    
    public record SectionSnapshot(int sectionX, int sectionY, int sectionZ,
                                  BlockState[] blocks18x18x18,
                                  byte[] blockLight18x18x18,
                                  byte[] skyLight18x18x18,
                                  int[] tints18x18x18) {

        public SectionSnapshot(int sectionX, int sectionY, int sectionZ,
                               BlockState[] blocks18x18x18,
                               byte[] blockLight18x18x18,
                               byte[] skyLight18x18x18) {
            this(sectionX, sectionY, sectionZ, blocks18x18x18, blockLight18x18x18, skyLight18x18x18, null);
        }
        
        public static SectionSnapshot capture(ClientLevel level, int sectionX, int sectionY, int sectionZ) {
            int baseY = sectionY << 4;
            if (baseY < level.getMinBuildHeight() || baseY >= level.getMaxBuildHeight()) {
                return null;
            }

            LevelChunk centerChunk = level.getChunkSource().getChunkNow(sectionX, sectionZ);
            if (centerChunk == null) {
                return null;
            }
            int centerSectionIdx = centerChunk.getSectionIndex(baseY);
            if (centerSectionIdx < 0 || centerSectionIdx >= centerChunk.getSectionsCount()) {
                return null;
            }
            LevelChunkSection centerSection = centerChunk.getSection(centerSectionIdx);
            if (centerSection == null || centerSection.hasOnlyAir()) {
                return null;
            }

            int haloVolume = HALO_SIZE * HALO_SIZE * HALO_SIZE;
            BlockState[] blocks = new BlockState[haloVolume];
            byte[] blockLight = new byte[haloVolume];
            byte[] skyLight = new byte[haloVolume];
            int[] tints = new int[haloVolume];
            Arrays.fill(tints, -1);

            BlockColors blockColors = null;
            try {
                Minecraft mc = Minecraft.getInstance();
                if (mc != null) {
                    blockColors = mc.getBlockColors();
                }
            } catch (Exception ignored) {
            }
            
            LayerLightEventListener blockLightListener = level.getLightEngine().getLayerListener(LightLayer.BLOCK);
            LayerLightEventListener skyLightListener = level.getLightEngine().getLayerListener(LightLayer.SKY);
            
            int baseX = sectionX << 4;
            int baseZ = sectionZ << 4;

            // Pre-fetch the 3x3 chunk neighborhood once instead of 5,832 map lookups
            LevelChunk[] neighborChunks = new LevelChunk[9];
            for (int dz = -1; dz <= 1; dz++) {
                for (int dx = -1; dx <= 1; dx++) {
                    neighborChunks[(dz + 1) * 3 + (dx + 1)] = (dx == 0 && dz == 0)
                            ? centerChunk
                            : level.getChunkSource().getChunkNow(sectionX + dx, sectionZ + dz);
                }
            }

            BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
            int minBuildY = level.getMinBuildHeight();
            int maxBuildY = level.getMaxBuildHeight();

            for (int y = 0; y < HALO_SIZE; y++) {
                int wy = baseY + y - 1;
                boolean inBoundsY = wy >= minBuildY && wy < maxBuildY;
                for (int z = 0; z < HALO_SIZE; z++) {
                    int wz = baseZ + z - 1;
                    int cz = SectionPos.blockToSectionCoord(wz) - sectionZ + 1;
                    for (int x = 0; x < HALO_SIZE; x++) {
                        int wx = baseX + x - 1;
                        int idx = (y * HALO_SIZE + z) * HALO_SIZE + x;

                        pos.set(wx, wy, wz);

                        BlockState state;
                        if (!inBoundsY) {
                            state = Blocks.AIR.defaultBlockState();
                        } else {
                            int cx = SectionPos.blockToSectionCoord(wx) - sectionX + 1;
                            LevelChunk chunk = neighborChunks[cz * 3 + cx];
                            if (chunk == null) {
                                state = Blocks.AIR.defaultBlockState();
                            } else {
                                int sectionIdx = chunk.getSectionIndex(wy);
                                if (sectionIdx < 0 || sectionIdx >= chunk.getSectionsCount()) {
                                    state = Blocks.AIR.defaultBlockState();
                                } else {
                                    LevelChunkSection section = chunk.getSection(sectionIdx);
                                    if (section == null || section.hasOnlyAir()) {
                                        state = Blocks.AIR.defaultBlockState();
                                    } else {
                                        state = section.getBlockState(wx & 15, wy & 15, wz & 15);
                                    }
                                }
                            }
                        }
                        blocks[idx] = state;

                        int bl = 0;
                        int sl = 0;
                        if (inBoundsY) {
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

                        if (inBoundsY && state != null && !state.isAir()
                                && x >= 1 && x <= SECTION_SIZE && y >= 1 && y <= SECTION_SIZE && z >= 1 && z <= SECTION_SIZE) {
                            int tint = -1;
                            if (!state.getFluidState().isEmpty()) {
                                try {
                                    tint = BiomeColors.getAverageWaterColor(level, pos) & 0xFFFFFF;
                                } catch (Exception ignored) {
                                }
                            } else if (blockColors != null) {
                                try {
                                    int c = blockColors.getColor(state, level, pos, 0);
                                    if (c != -1) {
                                        tint = c & 0xFFFFFF;
                                    }
                                } catch (Exception ignored) {
                                }
                            }
                            tints[idx] = tint;
                        }
                    }
                }
            }
            
            return new SectionSnapshot(sectionX, sectionY, sectionZ, blocks, blockLight, skyLight, tints);
        }
    }
    
    public record MeshedSectionResult(int sectionX, int sectionY, int sectionZ,
                                      long[] packedVertices, int quadCount,
                                      int aabbPacked, int flags,
                                      long[] translucentVertices, int translucentQuadCount, int translucentAabbPacked,
                                      long[] halfScaleOpaqueVertices, int halfScaleOpaqueQuadCount, int halfScaleOpaqueAabbPacked,
                                      long[] halfScaleTranslucentVertices, int halfScaleTranslucentQuadCount, int halfScaleTranslucentAabbPacked) {

        public MeshedSectionResult(int sectionX, int sectionY, int sectionZ,
                                   long[] packedVertices, int quadCount,
                                   int aabbPacked, int flags) {
            this(sectionX, sectionY, sectionZ, packedVertices, quadCount, aabbPacked, flags,
                    null, 0, 0,
                    null, 0, 0,
                    null, 0, 0);
        }
    }
    
    private record FaceInfo(int spriteId, int materialMask, boolean windWaving,
                            int ao0, int ao1, int ao2, int ao3,
                            int blockLight, int skyLight) {}
    
    public static CompletableFuture<MeshedSectionResult> meshAsync(SectionSnapshot snapshot) {
        return CompletableFuture.supplyAsync(() -> mesh(snapshot), ensureExecutor());
    }
    
    private static boolean isTranslucentState(BlockState state) {
        return !state.getFluidState().isEmpty()
                || state.getBlock() instanceof net.minecraft.world.level.block.TransparentBlock;
    }

    private static boolean isFullBoxShape(VoxelShape shape) {
        if (shape == null || shape.isEmpty()) return false;
        List<AABB> boxes = shape.toAabbs();
        if (boxes.size() != 1) return false;
        AABB b = boxes.get(0);
        return b.minX <= 0.01 && b.minY <= 0.01 && b.minZ <= 0.01
                && b.maxX >= 0.99 && b.maxY >= 0.99 && b.maxZ >= 0.99;
    }

    public static MeshedSectionResult mesh(SectionSnapshot snapshot) {
        List<Long> vertices = new ArrayList<>();
        List<Long> translucentVertices = new ArrayList<>();
        List<Long> halfOpaqueVertices = new ArrayList<>();
        List<Long> halfTranslucentVertices = new ArrayList<>();
        int quadCount = 0;
        int translucentQuadCount = 0;
        int halfOpaqueQuadCount = 0;
        int halfTranslucentQuadCount = 0;
        boolean hasTranslucent = false;
        boolean dirtyShadow = false;
        
        int minX = 16, minY = 16, minZ = 16;
        int maxX = 0, maxY = 0, maxZ = 0;
        boolean anyQuad = false;

        int tMinX = 16, tMinY = 16, tMinZ = 16, tMaxX = 0, tMaxY = 0, tMaxZ = 0;
        boolean anyTranslucent = false;
        int hoMinX = 16, hoMinY = 16, hoMinZ = 16, hoMaxX = 0, hoMaxY = 0, hoMaxZ = 0;
        boolean anyHalfOpaque = false;
        int htMinX = 16, htMinY = 16, htMinZ = 16, htMaxX = 0, htMaxY = 0, htMaxZ = 0;
        boolean anyHalfTranslucent = false;
        
        BlockState[] blocks = snapshot.blocks18x18x18();
        byte[] blockLight = snapshot.blockLight18x18x18();
        byte[] skyLight = snapshot.skyLight18x18x18();
        int[] tints = snapshot.tints18x18x18();
        
        AtlasUVTable atlas = AtlasUVTable.getInstance();

        // Pre-pass for non-full-cube VoxelShape blocks (slabs, stairs, fences, walls, panes, etc.)
        for (int y = 0; y < 16; y++) {
            for (int z = 0; z < 16; z++) {
                for (int x = 0; x < 16; x++) {
                    int idx = ((y + 1) * HALO_SIZE + (z + 1)) * HALO_SIZE + (x + 1);
                    BlockState state = blocks[idx];
                    if (state == null || state.isAir()) continue;
                    if (state.getRenderShape() == RenderShape.ENTITYBLOCK_ANIMATED) continue;
                    if (state.getRenderShape() == RenderShape.INVISIBLE && state.getFluidState().isEmpty()) continue;
                    if (isOpaqueFullCube(state)) continue;

                    VoxelShape shape = state.getShape(EmptyBlockGetter.INSTANCE, BlockPos.ZERO);
                    if (shape.isEmpty() || isFullBoxShape(shape)) continue;

                    boolean translucent = isTranslucentState(state);
                    if (translucent) hasTranslucent = true;
                    int tintRgb = (tints != null) ? tints[idx] : -1;
                    int bl = blockLight[idx] & 0xFF;
                    int sl = skyLight[idx] & 0xFF;
                    int mat = atlas.getMaterialMask(state);
                    boolean wind = atlas.isWindWaving(state);

                    List<Long> targetVerts = translucent ? halfTranslucentVertices : halfOpaqueVertices;
                    int added = emitShapeBoxes(targetVerts, shape, x, y, z, state, tintRgb, bl, sl, mat, wind, atlas);
                    if (added > 0) {
                        if (translucent) {
                            halfTranslucentQuadCount += added;
                            if (!anyHalfTranslucent) {
                                htMinX = x; htMinY = y; htMinZ = z;
                                htMaxX = x + 1; htMaxY = y + 1; htMaxZ = z + 1;
                                anyHalfTranslucent = true;
                            } else {
                                htMinX = Math.min(htMinX, x); htMinY = Math.min(htMinY, y); htMinZ = Math.min(htMinZ, z);
                                htMaxX = Math.max(htMaxX, x + 1); htMaxY = Math.max(htMaxY, y + 1); htMaxZ = Math.max(htMaxZ, z + 1);
                            }
                        } else {
                            halfOpaqueQuadCount += added;
                            if (!anyHalfOpaque) {
                                hoMinX = x; hoMinY = y; hoMinZ = z;
                                hoMaxX = x + 1; hoMaxY = y + 1; hoMaxZ = z + 1;
                                anyHalfOpaque = true;
                            } else {
                                hoMinX = Math.min(hoMinX, x); hoMinY = Math.min(hoMinY, y); hoMinZ = Math.min(hoMinZ, z);
                                hoMaxX = Math.max(hoMaxX, x + 1); hoMaxY = Math.max(hoMaxY, y + 1); hoMaxZ = Math.max(hoMaxZ, z + 1);
                            }
                        }
                    }
                }
            }
        }
        
        // Process each of 6 directions for full-cube greedy meshing (opaque + translucent)
        for (int dir = 0; dir < 6; dir++) {
            // For each slice along the normal axis
            for (int slice = 0; slice < 16; slice++) {
                FaceInfo[][] mask = new FaceInfo[16][16];
                FaceInfo[][] translucentMask = new FaceInfo[16][16];
                
                // Build mask
                for (int a = 0; a < 16; a++) {
                    for (int b = 0; b < 16; b++) {
                        int x, y, z;
                        switch (dir) {
                            case 0: x = slice; y = b; z = a; break; // +X (a=Z, b=Y)
                            case 1: x = slice; y = b; z = a; break; // -X (a=Z, b=Y)
                            case 2: x = a; y = slice; z = b; break; // +Y (a=X, b=Z)
                            case 3: x = a; y = slice; z = b; break; // -Y (a=X, b=Z)
                            case 4: x = a; y = b; z = slice; break; // +Z (a=X, b=Y)
                            case 5: x = a; y = b; z = slice; break; // -Z (a=X, b=Y)
                            default: continue;
                        }
                        
                        // Get block at (x,y,z) and neighbor
                        int idx = ((y + 1) * HALO_SIZE + (z + 1)) * HALO_SIZE + (x + 1);
                        BlockState state = blocks[idx];

                        if (state == null || state.isAir()) continue;
                        if (state.getRenderShape() == RenderShape.ENTITYBLOCK_ANIMATED) {
                            continue; // Rendered by vanilla BlockEntityRenderer; avoid ghost cubes
                        }
                        if (state.getRenderShape() == RenderShape.INVISIBLE && state.getFluidState().isEmpty()) {
                            continue; // Skip barriers, structure void, and invisible light blocks
                        }
                        if (!isOpaqueFullCube(state)) {
                            VoxelShape shape = state.getShape(EmptyBlockGetter.INSTANCE, BlockPos.ZERO);
                            if (!shape.isEmpty() && !isFullBoxShape(shape)) {
                                continue; // Already emitted by non-full VoxelShape pass
                            }
                        }
                        boolean translucent = isTranslucentState(state);
                        if (translucent) {
                            hasTranslucent = true;
                        }
                        
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
                        
                        // Face is culled if neighbor is opaque, or if both are identical non-opaque blocks (except fancy leaves)
                        if (isOpaqueFullCube(neighbor)) continue;
                        if (neighbor != null && !isOpaqueFullCube(state)
                                && !(state.getBlock() instanceof net.minecraft.world.level.block.LeavesBlock)
                                && state.getBlock() == neighbor.getBlock()) continue;
                        if (!state.getFluidState().isEmpty() && neighbor != null && !neighbor.getFluidState().isEmpty()
                                && state.getFluidState().getType() == neighbor.getFluidState().getType()) {
                            continue;
                        }
                        
                        // Compute AO and light
                        int[] ao = new int[4];
                        computeAO(blocks, x, y, z, dir, ao);
                        
                        int bl = blockLight[nIdx] & 0xFF;
                        int sl = skyLight[nIdx] & 0xFF;
                        int tintRgb = (tints != null) ? tints[idx] : -1;
                        
                        int spriteId = atlas.getSpriteId(state, dir, tintRgb);
                        int materialMask = atlas.getMaterialMask(state);
                        boolean wind = atlas.isWindWaving(state);
                        
                        FaceInfo faceInfo = new FaceInfo(spriteId, materialMask, wind,
                                ao[0], ao[1], ao[2], ao[3], bl, sl);
                        if (translucent) {
                            translucentMask[a][b] = faceInfo;
                        } else {
                            mask[a][b] = faceInfo;
                        }
                    }
                }
                
                // Greedy merge opaque mask
                for (int a = 0; a < 16; a++) {
                    for (int b = 0; b < 16; b++) {
                        FaceInfo info = mask[a][b];
                        if (info == null) continue;
                        
                        int w = 1;
                        while (a + w < 16 && info.equals(mask[a + w][b])) w++;
                        int h = 1;
                        outer:
                        while (b + h < 16) {
                            for (int k = 0; k < w; k++) {
                                if (!info.equals(mask[a + k][b + h])) break outer;
                            }
                            h++;
                        }
                        
                        emitQuad(vertices, dir, slice, a, b, w, h, info);
                        quadCount++;
                        
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
                        if (info.materialMask() == 1) hasTranslucent = true;
                        
                        for (int k = 0; k < w; k++) {
                            for (int l = 0; l < h; l++) {
                                mask[a + k][b + l] = null;
                            }
                        }
                    }
                }

                // Greedy merge translucent mask
                for (int a = 0; a < 16; a++) {
                    for (int b = 0; b < 16; b++) {
                        FaceInfo info = translucentMask[a][b];
                        if (info == null) continue;

                        int w = 1;
                        while (a + w < 16 && info.equals(translucentMask[a + w][b])) w++;
                        int h = 1;
                        outerT:
                        while (b + h < 16) {
                            for (int k = 0; k < w; k++) {
                                if (!info.equals(translucentMask[a + k][b + h])) break outerT;
                            }
                            h++;
                        }

                        emitQuad(translucentVertices, dir, slice, a, b, w, h, info);
                        translucentQuadCount++;

                        int[] bounds = computeQuadBounds(dir, slice, a, b, w, h);
                        if (!anyTranslucent) {
                            tMinX = bounds[0]; tMinY = bounds[1]; tMinZ = bounds[2];
                            tMaxX = bounds[3]; tMaxY = bounds[4]; tMaxZ = bounds[5];
                            anyTranslucent = true;
                        } else {
                            tMinX = Math.min(tMinX, bounds[0]); tMinY = Math.min(tMinY, bounds[1]); tMinZ = Math.min(tMinZ, bounds[2]);
                            tMaxX = Math.max(tMaxX, bounds[3]); tMaxY = Math.max(tMaxY, bounds[4]); tMaxZ = Math.max(tMaxZ, bounds[5]);
                        }

                        for (int k = 0; k < w; k++) {
                            for (int l = 0; l < h; l++) {
                                translucentMask[a + k][b + l] = null;
                            }
                        }
                    }
                }
            }
        }
        
        long[] packed = toArray(vertices);
        long[] tPacked = toArray(translucentVertices);
        long[] hoPacked = toArray(halfOpaqueVertices);
        long[] htPacked = toArray(halfTranslucentVertices);
        
        int aabbPacked = anyQuad ? VertexPacker.packTightAabb(minX, minY, minZ, maxX, maxY, maxZ) : 0;
        int tAabb = anyTranslucent ? VertexPacker.packTightAabb(tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ) : 0;
        int hoAabb = anyHalfOpaque ? VertexPacker.packTightAabb(hoMinX, hoMinY, hoMinZ, hoMaxX, hoMaxY, hoMaxZ) : 0;
        int htAabb = anyHalfTranslucent ? VertexPacker.packTightAabb(htMinX, htMinY, htMinZ, htMaxX, htMaxY, htMaxZ) : 0;

        int flags = 0;
        if (hasTranslucent) flags |= 1;
        if (dirtyShadow) flags |= 2;
        
        return new MeshedSectionResult(snapshot.sectionX(), snapshot.sectionY(), snapshot.sectionZ(),
                packed, quadCount, aabbPacked, flags,
                tPacked, translucentQuadCount, tAabb,
                hoPacked, halfOpaqueQuadCount, hoAabb,
                htPacked, halfTranslucentQuadCount, htAabb);
    }

    private static long[] toArray(List<Long> list) {
        if (list.isEmpty()) return new long[0];
        long[] arr = new long[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }

    private static int emitShapeBoxes(List<Long> out, VoxelShape shape, int bx, int by, int bz,
                                      BlockState state, int tintRgb, int bl, int sl, int mat, boolean wind,
                                      AtlasUVTable atlas) {
        int quads = 0;
        List<AABB> boxes = shape.toAabbs();
        int maxBoxes = Math.min(boxes.size(), 6);
        for (int bi = 0; bi < maxBoxes; bi++) {
            AABB box = boxes.get(bi);
            int x0 = bx * 2 + Math.max(0, Math.min(2, Math.round((float) box.minX * 2.0f)));
            int x1 = bx * 2 + Math.max(0, Math.min(2, Math.round((float) box.maxX * 2.0f)));
            int y0 = by * 2 + Math.max(0, Math.min(2, Math.round((float) box.minY * 2.0f)));
            int y1 = by * 2 + Math.max(0, Math.min(2, Math.round((float) box.maxY * 2.0f)));
            int z0 = bz * 2 + Math.max(0, Math.min(2, Math.round((float) box.minZ * 2.0f)));
            int z1 = bz * 2 + Math.max(0, Math.min(2, Math.round((float) box.maxZ * 2.0f)));
            if (x1 == x0) { if (x0 < bx * 2 + 2) x1 = x0 + 1; else x0 = x1 - 1; }
            if (y1 == y0) { if (y0 < by * 2 + 2) y1 = y0 + 1; else y0 = y1 - 1; }
            if (z1 == z0) { if (z0 < bz * 2 + 2) z1 = z0 + 1; else z0 = z1 - 1; }

            // Emit 6 faces of quantized half-scale box (coords in 0..32)
            for (int dir = 0; dir < 6; dir++) {
                int spriteId = atlas.getSpriteId(state, dir, tintRgb);
                int[][] c;
                switch (dir) {
                    case 0 -> c = new int[][]{{x1, y0, z0, 3}, {x1, y1, z0, 0}, {x1, y1, z1, 1}, {x1, y0, z1, 2}}; // +X
                    case 1 -> c = new int[][]{{x0, y0, z0, 3}, {x0, y0, z1, 2}, {x0, y1, z1, 1}, {x0, y1, z0, 0}}; // -X
                    case 2 -> c = new int[][]{{x0, y1, z0, 3}, {x0, y1, z1, 0}, {x1, y1, z1, 1}, {x1, y1, z0, 2}}; // +Y
                    case 3 -> c = new int[][]{{x0, y0, z0, 3}, {x1, y0, z0, 2}, {x1, y0, z1, 1}, {x0, y0, z1, 0}}; // -Y
                    case 4 -> c = new int[][]{{x0, y0, z1, 3}, {x1, y0, z1, 2}, {x1, y1, z1, 1}, {x0, y1, z1, 0}}; // +Z
                    default -> c = new int[][]{{x0, y0, z0, 3}, {x0, y1, z0, 0}, {x1, y1, z0, 1}, {x1, y0, z0, 2}}; // -Z
                }
                for (int v = 0; v < 4; v++) {
                    out.add(VertexPacker.pack(c[v][0], c[v][1], c[v][2], dir, 3, bl, sl, wind, spriteId, 1, 1, c[v][3], mat));
                }
                quads++;
            }
        }
        return quads;
    }
    
    private static boolean isOpaqueFullCube(BlockState state) {
        if (state == null || state.isAir()) return false;
        return state.isSolidRender(EmptyBlockGetter.INSTANCE, BlockPos.ZERO);
    }
    
    private static void computeAO(BlockState[] blocks, int x, int y, int z, int dir, int[] ao) {
        // Corner order in (a, b) slice space:
        // 0=(a, b), 1=(a+1, b), 2=(a+1, b+1), 3=(a, b+1)
        int nx = x, ny = y, nz = z;
        switch (dir) {
            case 0: nx = x + 1; break;
            case 1: nx = x - 1; break;
            case 2: ny = y + 1; break;
            case 3: ny = y - 1; break;
            case 4: nz = z + 1; break;
            case 5: nz = z - 1; break;
        }
        
        // u (a-axis) and v (b-axis) matching the slice loops
        int ux = 0, uy = 0, uz = 0;
        int vx = 0, vy = 0, vz = 0;
        switch (dir) {
            case 0: case 1: ux = 0; uy = 0; uz = 1; vx = 0; vy = 1; vz = 0; break; // a=Z, b=Y
            case 2: case 3: ux = 1; uy = 0; uz = 0; vx = 0; vy = 0; vz = 1; break; // a=X, b=Z
            case 4: case 5: ux = 1; uy = 0; uz = 0; vx = 0; vy = 1; vz = 0; break; // a=X, b=Y
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
        return s != null && !s.isAir() && s.isSolidRender(EmptyBlockGetter.INSTANCE, BlockPos.ZERO);
    }
    
    private static void emitQuad(List<Long> vertices, int dir, int slice, int a, int b, int w, int h, FaceInfo info) {
        // Emit 4 vertices in strict CCW geometric order as viewed from outside the face,
        // while assigning cornerIndex (0=(0,0) top-left .. 3=(0,1) bottom-left) so textures
        // appear upright and unmirrored on all 6 faces, paired with matching AO corner values.
        // Global index pattern is (0, 1, 2, 0, 2, 3).
        int[] ao = {info.ao0(), info.ao1(), info.ao2(), info.ao3()};

        // Determine (ca, cb, aoIdx, cornerIdx) for emitted vertices 0..3:
        // Vertex 0: bottom-left (cornerIndex 3 -> UV(0, 1))
        // Vertex 1: bottom-right (cornerIndex 2 -> UV(1, 1))
        // Vertex 2: top-right (cornerIndex 1 -> UV(1, 0))
        // Vertex 3: top-left (cornerIndex 0 -> UV(0, 0))
        int[][] spec;
        switch (dir) {
            case 0: // +X: right=-Z (a+w -> a), up=+Y (b -> b+h)
            case 5: // -Z: right=-X (a+w -> a), up=+Y (b -> b+h)
                spec = new int[][]{
                        {a + w, b,     1, 3},
                        {a,     b,     0, 2},
                        {a,     b + h, 3, 1},
                        {a + w, b + h, 2, 0}
                };
                break;
            case 2: // +Y: right=+X (a -> a+w), up=-Z (b+h -> b)
                spec = new int[][]{
                        {a,     b + h, 3, 3},
                        {a + w, b + h, 2, 2},
                        {a + w, b,     1, 1},
                        {a,     b,     0, 0}
                };
                break;
            case 1: // -X: right=+Z (a -> a+w), up=+Y (b -> b+h)
            case 3: // -Y: right=+X (a -> a+w), up=+Z (b -> b+h)
            case 4: // +Z: right=+X (a -> a+w), up=+Y (b -> b+h)
            default:
                spec = new int[][]{
                        {a,     b,     0, 3},
                        {a + w, b,     1, 2},
                        {a + w, b + h, 2, 1},
                        {a,     b + h, 3, 0}
                };
                break;
        }

        for (int v = 0; v < 4; v++) {
            int ca = spec[v][0];
            int cb = spec[v][1];
            int aoVal = ao[spec[v][2]];
            int cornerIdx = spec[v][3];

            int x, y, z;
            switch (dir) {
                case 0: x = slice + 1; y = cb; z = ca; break;
                case 1: x = slice;     y = cb; z = ca; break;
                case 2: x = ca; y = slice + 1; z = cb; break;
                case 3: x = ca; y = slice;     z = cb; break;
                case 4: x = ca; y = cb; z = slice + 1; break;
                case 5: x = ca; y = cb; z = slice;     break;
                default: x = 0; y = 0; z = 0;
            }

            long packed = VertexPacker.pack(
                    x, y, z,
                    dir, aoVal, info.blockLight(), info.skyLight(),
                    info.windWaving(), info.spriteId(), w, h, cornerIdx, info.materialMask()
            );
            vertices.add(packed);
        }
    }
    
    private static int[] computeQuadBounds(int dir, int slice, int a, int b, int w, int h) {
        int minX, minY, minZ, maxX, maxY, maxZ;
        switch (dir) {
            case 0: // +X (a=Z, b=Y)
                minX = slice + 1; maxX = slice + 1;
                minY = b; maxY = b + h;
                minZ = a; maxZ = a + w;
                break;
            case 1: // -X (a=Z, b=Y)
                minX = slice; maxX = slice;
                minY = b; maxY = b + h;
                minZ = a; maxZ = a + w;
                break;
            case 2: // +Y (a=X, b=Z)
                minX = a; maxX = a + w;
                minY = slice + 1; maxY = slice + 1;
                minZ = b; maxZ = b + h;
                break;
            case 3: // -Y (a=X, b=Z)
                minX = a; maxX = a + w;
                minY = slice; maxY = slice;
                minZ = b; maxZ = b + h;
                break;
            case 4: // +Z (a=X, b=Y)
                minX = a; maxX = a + w;
                minY = b; maxY = b + h;
                minZ = slice + 1; maxZ = slice + 1;
                break;
            case 5: // -Z (a=X, b=Y)
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
