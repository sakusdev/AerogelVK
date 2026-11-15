package org.sakus.aerogelvk.meshing;

import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import net.minecraft.client.Minecraft;
import net.minecraft.client.color.block.BlockColors;
import net.minecraft.client.renderer.block.model.BakedQuad;
import net.minecraft.client.renderer.texture.AbstractTexture;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.client.resources.model.BakedModel;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.tags.BlockTags;
import net.minecraft.util.RandomSource;
import net.minecraft.world.inventory.InventoryMenu;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.material.FluidState;
import net.minecraft.world.level.material.Fluids;
import org.lwjgl.opengl.GL11C;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Maps (BlockState, faceDirection) to 16-bit Sprite IDs and maintains a std430-layout
 * native ByteBuffer of atlas UV bounds + tint color (32 bytes per entry) for upload
 * to the Vulkan AtlasUVSSBO (Descriptor Set 0, Binding 6).
 */
public final class AtlasUVTable {

    public static final int MAX_SPRITES = 65536;
    public static final int ENTRY_BYTES = 32; // vec2 uvMin, vec2 uvMax, vec4 tint (std430)

    private static final Direction[] FACE_DIRS = {
            Direction.EAST,  // 0: +X
            Direction.WEST,  // 1: -X
            Direction.UP,    // 2: +Y
            Direction.DOWN,  // 3: -Y
            Direction.SOUTH, // 4: +Z
            Direction.NORTH  // 5: -Z
    };

    private static final AtlasUVTable INSTANCE = new AtlasUVTable();

    public static AtlasUVTable getInstance() {
        return INSTANCE;
    }

    private record SpriteKey(ResourceLocation name, int tintRgb) {}

    /** Immutable lookup snapshot published after rebuildFromBlockAtlas() for lock-free worker reads. */
    private record SpriteLookup(
            Map<BlockState, int[]> stateDirSpriteLookup,
            Object2IntOpenHashMap<BlockState> stateSpriteLookup,
            Object2IntOpenHashMap<BlockState> stateDirTintedMask
    ) {}

    /** (spriteName, tintRgb) -> spriteId */
    private final Map<SpriteKey, Integer> spriteIdLookup = new ConcurrentHashMap<>(1024);
    private final ConcurrentHashMap<Long, Integer> tintedVariantCache = new ConcurrentHashMap<>(512);
    private final AtomicInteger publishedSpriteCount = new AtomicInteger(0);
    private volatile SpriteLookup lookup = new SpriteLookup(Map.of(), new Object2IntOpenHashMap<>(), new Object2IntOpenHashMap<>());

    private ByteBuffer uvTableBuffer;
    private volatile boolean dirty = true;
    private int spriteCount = 0;

    private AtlasUVTable() {
        lookup.stateSpriteLookup().defaultReturnValue(-1);
        lookup.stateDirTintedMask().defaultReturnValue(0);
        uvTableBuffer = MemoryUtil.memAlloc(MAX_SPRITES * ENTRY_BYTES);
        uvTableBuffer.limit(0);
    }

    /**
     * Rebuilds the sprite table by iterating all registered blocks and their
     * possible states, resolving per-face sprites and default biome/block tints.
     */
    public synchronized void rebuildFromBlockAtlas() {
        Minecraft mc = Minecraft.getInstance();
        if (mc.getModelManager() == null || mc.getBlockRenderer() == null) {
            return; // Too early during client bootstrap
        }

        spriteIdLookup.clear();
        tintedVariantCache.clear();
        Map<BlockState, int[]> nextDirLookup = new HashMap<>(4096);
        Object2IntOpenHashMap<BlockState> nextStateLookup = new Object2IntOpenHashMap<>(4096);
        nextStateLookup.defaultReturnValue(-1);
        Object2IntOpenHashMap<BlockState> nextDirTintedMask = new Object2IntOpenHashMap<>(4096);
        nextDirTintedMask.defaultReturnValue(0);
        uvTableBuffer.clear();
        spriteCount = 0;

        // Entry 0: fallback (missingno or full-atlas, untinted white)
        registerFallbackEntry(mc);

        TextureAtlas blockAtlas = null;
        try {
            blockAtlas = mc.getModelManager().getAtlas(InventoryMenu.BLOCK_ATLAS);
        } catch (Exception ignored) {
        }

        TextureAtlasSprite waterSprite = null;
        if (blockAtlas != null) {
            try {
                waterSprite = blockAtlas.getSprite(ResourceLocation.fromNamespaceAndPath("minecraft", "block/water_still"));
            } catch (Exception ignored) {
            }
        }

        BlockColors blockColors = mc.getBlockColors();
        RandomSource random = RandomSource.create(42L);

        for (Block block : BuiltInRegistries.BLOCK) {
            for (BlockState state : block.getStateDefinition().getPossibleStates()) {
                if (state.isAir()) {
                    continue;
                }

                // Water / fluid states do not use BakedModel quads in vanilla
                if (state.is(Blocks.WATER) || state.is(Blocks.BUBBLE_COLUMN)) {
                    TextureAtlasSprite sprite = waterSprite != null
                            ? waterSprite
                            : safeGetParticleIcon(mc, state);
                    int id = (sprite != null) ? registerSprite(sprite, 0x3F76E4) : 0;
                    nextStateLookup.put(state, id);
                    nextDirLookup.put(state, new int[]{id, id, id, id, id, id});
                    nextDirTintedMask.put(state, 0x3F);
                    continue;
                }

                TextureAtlasSprite particleSprite = safeGetParticleIcon(mc, state);
                int fallbackTint = resolveFallbackTint(blockColors, state);
                int particleId = (particleSprite != null) ? registerSprite(particleSprite, fallbackTint) : 0;
                nextStateLookup.put(state, particleId);

                int[] dirIds = new int[6];
                int tintedMask = 0;
                BakedModel model = null;
                try {
                    model = mc.getBlockRenderer().getBlockModel(state);
                } catch (Exception ignored) {
                }

                for (int d = 0; d < 6; d++) {
                    if (model == null) {
                        dirIds[d] = particleId;
                        continue;
                    }
                    try {
                        List<BakedQuad> quads = model.getQuads(state, FACE_DIRS[d], random);
                        if (quads.isEmpty()) {
                            quads = model.getQuads(state, null, random);
                        }
                        if (quads.isEmpty()) {
                            dirIds[d] = particleId;
                            continue;
                        }
                        BakedQuad quad = quads.get(0);
                        TextureAtlasSprite faceSprite = quad.getSprite();
                        int tintRgb = 0xFFFFFF;
                        if (quad.isTinted()) {
                            tintRgb = resolveQuadTint(blockColors, state, quad.getTintIndex());
                            tintedMask |= (1 << d);
                        }
                        dirIds[d] = (faceSprite != null) ? registerSprite(faceSprite, tintRgb) : particleId;
                    } catch (Exception e) {
                        dirIds[d] = particleId;
                    }
                }
                nextDirLookup.put(state, dirIds);
                if (tintedMask != 0) {
                    nextDirTintedMask.put(state, tintedMask);
                }
            }
        }

        uvTableBuffer.flip();
        publishedSpriteCount.set(spriteCount);
        this.lookup = new SpriteLookup(nextDirLookup, nextStateLookup, nextDirTintedMask);
        dirty = true;
    }

    private TextureAtlasSprite safeGetParticleIcon(Minecraft mc, BlockState state) {
        try {
            return mc.getBlockRenderer().getBlockModelShaper().getParticleIcon(state);
        } catch (Exception e) {
            return null;
        }
    }

    private int resolveQuadTint(BlockColors blockColors, BlockState state, int tintIndex) {
        if (blockColors != null) {
            try {
                int c = blockColors.getColor(state, null, null, tintIndex);
                if (c != -1) {
                    return c & 0xFFFFFF;
                }
            } catch (Exception ignored) {
            }
        }
        return resolveFallbackTint(blockColors, state);
    }

    private int resolveFallbackTint(BlockColors blockColors, BlockState state) {
        if (state.is(Blocks.GRASS_BLOCK) || isGrassLike(state)) {
            return 0x7CBD52; // Plains grass green
        }
        if (state.is(BlockTags.LEAVES)) {
            if (state.is(Blocks.SPRUCE_LEAVES)) return 0x619961;
            if (state.is(Blocks.BIRCH_LEAVES)) return 0x80A755;
            if (state.is(Blocks.CHERRY_LEAVES)) return 0xFFFFFF;
            return 0x59AE30; // Default oak/jungle foliage green
        }
        if (blockColors != null) {
            try {
                int c = blockColors.getColor(state, null, null, 0);
                if (c != -1) {
                    return c & 0xFFFFFF;
                }
            } catch (Exception ignored) {
            }
        }
        return 0xFFFFFF;
    }

    private void registerFallbackEntry(Minecraft mc) {
        float[] uv;
        try {
            TextureAtlas atlas = mc.getModelManager().getAtlas(InventoryMenu.BLOCK_ATLAS);
            TextureAtlasSprite missing = atlas.getSprite(ResourceLocation.fromNamespaceAndPath("minecraft", "missingno"));
            uv = new float[]{missing.getU0(), missing.getV0(), missing.getU1(), missing.getV1()};
        } catch (Exception e) {
            uv = new float[]{0f, 0f, 1f, 1f};
        }
        spriteCount = 1;
        appendEntry(uv, 0xFFFFFF);
    }

    /** Registers a unique (sprite, tint) pair, returning its 16-bit ID. Reuses existing IDs. */
    private int registerSprite(TextureAtlasSprite sprite, int tintRgb) {
        ResourceLocation name = sprite.contents().name();
        SpriteKey key = new SpriteKey(name, tintRgb & 0xFFFFFF);
        Integer existing = spriteIdLookup.get(key);
        if (existing != null) {
            return existing;
        }
        if (spriteCount >= MAX_SPRITES) {
            return 0; // Table full; fall back
        }
        int id = spriteCount++;
        spriteIdLookup.put(key, id);
        appendEntry(new float[]{sprite.getU0(), sprite.getV0(), sprite.getU1(), sprite.getV1()}, tintRgb);
        return id;
    }

    private void appendEntry(float[] uv, int tintRgb) {
        ByteBuffer buf = uvTableBuffer;
        float r = ((tintRgb >>> 16) & 0xFF) / 255.0f;
        float g = ((tintRgb >>> 8) & 0xFF) / 255.0f;
        float b = (tintRgb & 0xFF) / 255.0f;
        buf.putFloat(uv[0]).putFloat(uv[1]).putFloat(uv[2]).putFloat(uv[3]);
        buf.putFloat(r).putFloat(g).putFloat(b).putFloat(1.0f);
    }

    /**
     * Reads the current OpenGL Block Texture Atlas into a newly allocated native RGBA8 ByteBuffer.
     * Returns null if the atlas is unavailable. Caller must free with MemoryUtil.memFree(pixels).
     */
    public record AtlasPixels(int width, int height, ByteBuffer rgbaPixels) {}

    public AtlasPixels captureBlockAtlasPixels() {
        Minecraft mc = Minecraft.getInstance();
        if (mc.getTextureManager() == null) {
            return null;
        }
        AbstractTexture tex = mc.getTextureManager().getTexture(InventoryMenu.BLOCK_ATLAS);
        if (tex == null) {
            return null;
        }
        int glTexId = tex.getId();
        if (glTexId <= 0) {
            return null;
        }

        int prevTex = GL11C.glGetInteger(GL11C.GL_TEXTURE_BINDING_2D);
        GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, glTexId);
        int width = GL11C.glGetTexLevelParameteri(GL11C.GL_TEXTURE_2D, 0, GL11C.GL_TEXTURE_WIDTH);
        int height = GL11C.glGetTexLevelParameteri(GL11C.GL_TEXTURE_2D, 0, GL11C.GL_TEXTURE_HEIGHT);

        if (width <= 0 || height <= 0) {
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, prevTex);
            return null;
        }

        ByteBuffer pixels = MemoryUtil.memAlloc(width * height * 4);
        GL11C.glGetTexImage(GL11C.GL_TEXTURE_2D, 0, GL11C.GL_RGBA, GL11C.GL_UNSIGNED_BYTE, pixels);
        GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, prevTex);
        pixels.position(0);
        return new AtlasPixels(width, height, pixels);
    }

    /**
     * Returns the 16-bit Sprite ID for the given state and face direction (0:+X..5:-Z).
     */
    public int getSpriteId(BlockState state, int direction) {
        if (state == null || state.isAir()) {
            return 0;
        }
        SpriteLookup snap = this.lookup;
        int[] dirIds = snap.stateDirSpriteLookup().get(state);
        if (dirIds != null && direction >= 0 && direction < 6) {
            return dirIds[direction] & 0xFFFF;
        }
        int id = snap.stateSpriteLookup().getInt(state);
        return id >= 0 ? (id & 0xFFFF) : 0;
    }

    public int getSpriteId(BlockState state, int direction, int biomeTintRgb) {
        if (state == null || state.isAir()) {
            return 0;
        }
        int baseId = getSpriteId(state, direction);
        if (biomeTintRgb < 0 || direction < 0 || direction >= 6) {
            return baseId;
        }
        SpriteLookup snap = this.lookup;
        int tintedMask = snap.stateDirTintedMask().getInt(state);
        if ((tintedMask & (1 << direction)) == 0) {
            return baseId;
        }
        int rgb = biomeTintRgb & 0xFFFFFF;
        long cacheKey = (((long) baseId) << 24) | rgb;
        Integer cached = tintedVariantCache.get(cacheKey);
        if (cached != null) {
            return cached;
        }
        return registerTintedVariant(baseId, rgb, cacheKey);
    }

    /**
     * Thread-safe lookup or runtime registration of a TextureAtlasSprite + tint combination
     * for BakedModel shape quads.
     */
    public int getOrRegisterSpriteId(TextureAtlasSprite sprite, int tintRgb) {
        if (sprite == null || sprite.contents() == null) {
            return 0;
        }
        int rgb = tintRgb & 0xFFFFFF;
        SpriteKey key = new SpriteKey(sprite.contents().name(), rgb);
        Integer existing = spriteIdLookup.get(key);
        if (existing != null) {
            return existing & 0xFFFF;
        }
        return registerRuntimeSprite(sprite, rgb, key);
    }

    private synchronized int registerRuntimeSprite(TextureAtlasSprite sprite, int rgb, SpriteKey key) {
        Integer existing = spriteIdLookup.get(key);
        if (existing != null) {
            return existing & 0xFFFF;
        }
        if (uvTableBuffer == null || spriteCount >= MAX_SPRITES - 1) {
            return 0;
        }
        int newId = spriteCount++;
        uvTableBuffer.limit(spriteCount * ENTRY_BYTES);
        uvTableBuffer.position(newId * ENTRY_BYTES);
        appendEntry(new float[]{sprite.getU0(), sprite.getV0(), sprite.getU1(), sprite.getV1()}, rgb);
        uvTableBuffer.position(0);
        spriteIdLookup.put(key, newId);
        publishedSpriteCount.set(spriteCount);
        dirty = true;
        return newId;
    }

    private synchronized int registerTintedVariant(int baseId, int rgb, long cacheKey) {
        Integer existing = tintedVariantCache.get(cacheKey);
        if (existing != null) {
            return existing;
        }
        if (uvTableBuffer == null || baseId < 0 || baseId >= spriteCount || spriteCount >= MAX_SPRITES - 1) {
            return baseId;
        }
        int baseOff = baseId * ENTRY_BYTES;
        float u0 = uvTableBuffer.getFloat(baseOff);
        float v0 = uvTableBuffer.getFloat(baseOff + 4);
        float u1 = uvTableBuffer.getFloat(baseOff + 8);
        float v1 = uvTableBuffer.getFloat(baseOff + 12);

        int newId = spriteCount++;
        uvTableBuffer.limit(spriteCount * ENTRY_BYTES);
        uvTableBuffer.position(newId * ENTRY_BYTES);
        appendEntry(new float[]{u0, v0, u1, v1}, rgb);
        uvTableBuffer.position(0);
        publishedSpriteCount.set(spriteCount);
        tintedVariantCache.put(cacheKey, newId);
        dirty = true;
        return newId;
    }

    public int getSpriteId(BlockState state) {
        if (state == null || state.isAir()) {
            return 0;
        }
        int id = this.lookup.stateSpriteLookup().getInt(state);
        return id >= 0 ? (id & 0xFFFF) : 0;
    }

    /**
     * Classifies the block into the 4-bit Material Mask:
     * 0: Matte, 1: Water, 2: Metallic/Reflective, 3: Emissive, 4: Foliage/SSS.
     */
    public int getMaterialMask(BlockState state) {
        if (state == null || state.isAir()) {
            return 0;
        }

        // 3: Emissive
        if (state.getLightEmission() > 0) {
            return 3;
        }

        // 1: Water
        FluidState fluid = state.getFluidState();
        if (!fluid.isEmpty()) {
            if (fluid.getType() == Fluids.WATER || fluid.getType() == Fluids.FLOWING_WATER) {
                return 1;
            }
        }
        if (state.is(Blocks.WATER) || state.is(Blocks.BUBBLE_COLUMN)) {
            return 1;
        }

        // 4: Foliage / SSS
        if (isFoliage(state)) {
            return 4;
        }

        // 2: Metallic / Reflective
        if (isMetallic(state)) {
            return 2;
        }

        return 0; // Matte
    }

    private boolean isFoliage(BlockState state) {
        return state.is(BlockTags.LEAVES)
                || state.is(BlockTags.FLOWERS)
                || state.is(BlockTags.SAPLINGS)
                || state.is(BlockTags.CROPS)
                || state.is(BlockTags.REPLACEABLE_BY_TREES)
                || state.is(BlockTags.SMALL_FLOWERS)
                || state.is(BlockTags.TALL_FLOWERS)
                || state.is(BlockTags.WART_BLOCKS) == false && isGrassLike(state);
    }

    private boolean isGrassLike(BlockState state) {
        return state.is(Blocks.TALL_GRASS) || state.is(Blocks.FERN)
                || state.is(Blocks.LARGE_FERN) || state.is(Blocks.SHORT_GRASS)
                || state.is(Blocks.SEAGRASS) || state.is(Blocks.TALL_SEAGRASS)
                || state.is(Blocks.KELP) || state.is(Blocks.KELP_PLANT)
                || state.is(Blocks.SWEET_BERRY_BUSH) || state.is(Blocks.DEAD_BUSH)
                || state.is(Blocks.VINE) || state.is(Blocks.GLOW_LICHEN)
                || state.is(Blocks.CACTUS) || state.is(Blocks.SUGAR_CANE)
                || state.is(Blocks.BAMBOO) || state.is(Blocks.BAMBOO_SAPLING);
    }

    private boolean isMetallic(BlockState state) {
        if (state.is(BlockTags.BEACON_BASE_BLOCKS)) {
            return true; // iron, gold, diamond, emerald, netherite
        }
        return state.is(Blocks.COPPER_BLOCK) || state.is(Blocks.EXPOSED_COPPER)
                || state.is(Blocks.WEATHERED_COPPER) || state.is(Blocks.OXIDIZED_COPPER)
                || state.is(Blocks.CUT_COPPER) || state.is(Blocks.EXPOSED_CUT_COPPER)
                || state.is(Blocks.WEATHERED_CUT_COPPER) || state.is(Blocks.OXIDIZED_CUT_COPPER)
                || state.is(Blocks.RAW_IRON_BLOCK) || state.is(Blocks.RAW_COPPER_BLOCK)
                || state.is(Blocks.RAW_GOLD_BLOCK)
                || state.is(Blocks.ANVIL) || state.is(Blocks.CHIPPED_ANVIL) || state.is(Blocks.DAMAGED_ANVIL)
                || state.is(Blocks.LIGHTNING_ROD) || state.is(Blocks.CHAIN)
                || state.is(Blocks.IRON_BARS) || state.is(Blocks.IRON_DOOR) || state.is(Blocks.IRON_TRAPDOOR)
                || state.is(Blocks.CAULDRON) || state.is(Blocks.WATER_CAULDRON)
                || state.is(Blocks.LAVA_CAULDRON) || state.is(Blocks.POWDER_SNOW_CAULDRON);
    }

    /**
     * Returns true for blocks that should receive wind waving animation
     * (leaves, tall grass, flowers, and water).
     */
    public boolean isWindWaving(BlockState state) {
        if (state == null || state.isAir()) {
            return false;
        }
        if (getMaterialMask(state) == 1) {
            return true; // Water
        }
        return state.is(BlockTags.LEAVES)
                || state.is(BlockTags.SAPLINGS)
                || state.is(BlockTags.FLOWERS)
                || state.is(BlockTags.SMALL_FLOWERS)
                || state.is(BlockTags.TALL_FLOWERS)
                || isGrassLike(state);
    }

    /**
     * Returns the native std430-layout ByteBuffer ready for Vulkan SSBO upload.
     * The buffer's position/limit cover exactly {@link #getSpriteCount()} entries.
     */
    public synchronized ByteBuffer getUvTableBuffer() {
        ByteBuffer buf = uvTableBuffer;
        if (buf == null) {
            return null;
        }
        return buf.duplicate().order(buf.order()).position(0);
    }

    public int getSpriteCount() {
        return publishedSpriteCount.get();
    }

    public boolean isDirty() {
        return dirty;
    }

    public synchronized void clearDirty() {
        dirty = false;
    }

    /** Frees the native buffer. Call on client shutdown. */
    public synchronized void destroy() {
        if (uvTableBuffer != null) {
            MemoryUtil.memFree(uvTableBuffer);
            uvTableBuffer = null;
        }
    }
}
