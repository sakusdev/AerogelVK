package com.example.valkyrie.mixin;

import com.example.valkyrie.ValkyrieRenderer;
import net.minecraft.client.multiplayer.ClientChunkCache;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.network.FriendlyByteBuf;
import net.minecraft.network.protocol.game.ClientboundLevelChunkPacketData;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.function.Consumer;

@Mixin(ClientChunkCache.class)
public class MixinClientChunkCache {

    @Shadow
    @Final
    ClientLevel level;

    @Inject(method = "replaceWithPacketData", at = @At("RETURN"), require = 0)
    private void onReplaceWithPacketData(
            int chunkX,
            int chunkZ,
            FriendlyByteBuf buffer,
            CompoundTag heightmaps,
            Consumer<ClientboundLevelChunkPacketData.BlockEntityTagOutput> blockEntities,
            CallbackInfoReturnable<LevelChunk> cir
    ) {
        LevelChunk chunk = cir.getReturnValue();
        if (chunk == null || level == null) {
            return;
        }
        int minSection = level.getMinSection();
        LevelChunkSection[] sections = chunk.getSections();
        for (int i = 0; i < sections.length; i++) {
            LevelChunkSection section = sections[i];
            if (section != null && !section.hasOnlyAir()) {
                ValkyrieRenderer.getInstance().scheduleSectionRebuild(chunkX, minSection + i, chunkZ);
            }
        }
    }

    @Inject(method = "drop", at = @At("HEAD"), require = 0)
    private void onDropChunk(ChunkPos chunkPos, CallbackInfo ci) {
        if (level != null && chunkPos != null) {
            ValkyrieRenderer.getInstance().unloadChunkColumn(
                    chunkPos.x,
                    chunkPos.z,
                    level.getMinSection(),
                    level.getMaxSection()
            );
        }
    }
}
