package com.example.valkyrie.mixin;

import com.example.valkyrie.ValkyrieRenderer;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ClientLevel.class)
public class MixinChunkRebuild {

    @Inject(method = "setBlocksDirty", at = @At("HEAD"), require = 0)
    private void onSetBlocksDirty(BlockPos pos, BlockState oldState, BlockState newState, CallbackInfo ci) {
        int sectionX = SectionPos.blockToSectionCoord(pos.getX());
        int sectionY = SectionPos.blockToSectionCoord(pos.getY());
        int sectionZ = SectionPos.blockToSectionCoord(pos.getZ());
        ValkyrieRenderer.getInstance().scheduleSectionRebuild(sectionX, sectionY, sectionZ);
    }

    @Inject(method = "setSectionDirtyWithNeighbors", at = @At("HEAD"), require = 0)
    private void onSetSectionDirtyWithNeighbors(int sectionX, int sectionY, int sectionZ, CallbackInfo ci) {
        ValkyrieRenderer.getInstance().scheduleSectionRebuild(sectionX, sectionY, sectionZ);
    }

    @Inject(method = "onChunkLoaded", at = @At("RETURN"), require = 0)
    private void onChunkLoaded(ChunkPos chunkPos, CallbackInfo ci) {
        ClientLevel level = (ClientLevel) (Object) this;
        LevelChunk chunk = level.getChunkSource().getChunkNow(chunkPos.x, chunkPos.z);
        int minSection = level.getMinSection();
        if (chunk != null) {
            LevelChunkSection[] sections = chunk.getSections();
            for (int i = 0; i < sections.length; i++) {
                LevelChunkSection section = sections[i];
                if (section != null && !section.hasOnlyAir()) {
                    ValkyrieRenderer.getInstance().scheduleSectionRebuild(chunkPos.x, minSection + i, chunkPos.z);
                }
            }
            return;
        }
        int maxSection = level.getMaxSection();
        for (int sectionY = minSection; sectionY < maxSection; sectionY++) {
            ValkyrieRenderer.getInstance().scheduleSectionRebuild(chunkPos.x, sectionY, chunkPos.z);
        }
    }
}
