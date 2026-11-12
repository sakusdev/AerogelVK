package com.example.valkyrie.meshing;

import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.client.resources.model.Material;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.tags.BlockTags;
import net.minecraft.tags.TagKey;
import net.minecraft.world.inventory.InventoryMenu;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.material.FluidState;
import net.minecraft.world.level.material.Fluids;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Maps BlockStates to 16-bit Sprite IDs and maintains a std430-layout native
 * ByteBuffer of atlas UV bounds (u0, v0, u1, v1 per entry) for upload to the
 * Vulkan AtlasUVSSBO (Descriptor Set 0, Binding 6).
 */
public final class AtlasUVTable {

    public static final int MAX_SPRITES = 65536;
    public static final int ENTRY_BYTES = 16; // 4 floats

    private static final AtlasUVTable INSTANCE = new AtlasUVTable();

    public static AtlasUVTable getInstance() {
        return INSTANCE;
    }

    /** spriteId -> [u0, v0, u1, v1] */
    private final List<float[]> uvEntries = new ArrayList<>(1024);
    /** ResourceLocation (atlas sprite id) -> spriteId */
    private final Object2IntOpenHashMap<ResourceLocation> spriteIdLookup = new Object2IntOpenHashMap<>(1024);
    /** BlockState -> spriteId */
    private final Object2IntOpenHashMap<BlockState> stateSpriteLookup = new Object2IntOpenHashMap<>(4096);

    private ByteBuffer uvTableBuffer;
    private volatile boolean dirty = true;
    private int spriteCount = 0;

    private AtlasUVTable() {
        spriteIdLookup.defaultReturnValue(-1);
        stateSpriteLookup.defaultReturnValue(-1);
        uvTableBuffer = MemoryUtil.memAlloc(MAX_SPRITES * ENTRY_BYTES);
        uvTableBuffer.limit(0);
    }

    /**
     * Rebuilds the sprite table by iterating all registered blocks and their
     * possible states, resolving the particle icon for each state.
     */
    public synchronized void rebuildFromBlockAtlas() {
        Minecraft mc = Minecraft.getInstance();
        if (mc.getModelManager() == null || mc.getBlockRenderer() == null) {
            return; // Too early during client bootstrap
        }

        uvEntries.clear();
        spriteIdLookup.clear();
        stateSpriteLookup.clear();
        uvTableBuffer.clear();
        uvTableBuffer.limit(0);
        spriteCount = 0;

        // Reserve entry 0 as the fallback (full-atlas / white)
        registerFallbackEntry(mc);

        for (Block block : BuiltInRegistries.BLOCK) {
            for (BlockState state : block.getStateDefinition().getPossibleStates()) {
                try {
                    TextureAtlasSprite sprite = mc.getBlockRenderer().getBlockModelShaper().getParticleIcon(state);
                    if (sprite == null) {
                        stateSpriteLookup.put(state, 0);
                        continue;
                    }
                    int id = registerSprite(sprite);
                    stateSpriteLookup.put(state, id);
                } catch (Exception e) {
                    stateSpriteLookup.put(state, 0);
                }
            }
        }

        dirty = true;
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
        appendEntry(uv);
    }

    /** Registers a unique sprite, returning its 16-bit ID. Reuses existing IDs. */
    private int registerSprite(TextureAtlasSprite sprite) {
        ResourceLocation name = sprite.contents().name();
        int existing = spriteIdLookup.getInt(name);
        if (existing >= 0) {
            return existing;
        }
        if (spriteCount >= MAX_SPRITES) {
            return 0; // Table full; fall back
        }
        int id = spriteCount++;
        spriteIdLookup.put(name, id);
        appendEntry(new float[]{sprite.getU0(), sprite.getV0(), sprite.getU1(), sprite.getV1()});
        return id;
    }

    private void appendEntry(float[] uv) {
        ByteBuffer buf = uvTableBuffer;
        int pos = buf.position();
        buf.putFloat(uv[0]).putFloat(uv[1]).putFloat(uv[2]).putFloat(uv[3]);
        buf.limit(Math.max(buf.limit(), pos + ENTRY_BYTES));
    }

    /**
     * Returns the 16-bit Sprite ID for the given state (0 fallback for air/unknown).
     */
    public int getSpriteId(BlockState state, int direction) {
        return getSpriteId(state);
    }

    public int getSpriteId(BlockState state) {
        if (state == null || state.isAir()) {
            return 0;
        }
        int id = stateSpriteLookup.getInt(state);
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
    public ByteBuffer getUvTableBuffer() {
        ByteBuffer buf = uvTableBuffer;
        buf.position(0);
        return buf;
    }

    public int getSpriteCount() {
        return spriteCount;
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
