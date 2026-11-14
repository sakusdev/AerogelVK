package com.example.valkyrie.mixin.compat.sodium;

import com.example.valkyrie.ValkyrieRenderer;
import com.example.valkyrie.compat.CompatManager;
import com.llamalad7.mixinextras.sugar.ref.LocalBooleanRef;
import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.SheetedDecalTextureGenerator;
import com.mojang.blaze3d.vertex.VertexConsumer;
import com.mojang.blaze3d.vertex.VertexMultiConsumer;
import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import net.caffeinemc.mods.sodium.client.render.SodiumWorldRenderer;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.client.renderer.MultiBufferSource;
import net.minecraft.client.renderer.RenderBuffers;
import net.minecraft.client.renderer.RenderType;
import net.minecraft.client.renderer.blockentity.BlockEntityRenderDispatcher;
import net.minecraft.client.resources.model.ModelBakery;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;
import net.minecraft.server.level.BlockDestructionProgress;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.chunk.LevelChunk;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.SortedSet;
import java.util.function.Consumer;

/**
 * Coexistence bridge when Sodium is installed alongside ValkyrieRender:
 * - Suppresses Sodium's GL terrain draw pass so Valkyrie owns chunk rendering
 * - Forwards Sodium rebuild notifications to Valkyrie's GreedyMesher
 * - Prevents entity over-culling and restores block-entity rendering when Sodium meshing is suppressed
 */
@Mixin(value = SodiumWorldRenderer.class, remap = false)
public class MixinSodiumWorldRenderer {

    @Shadow
    private Minecraft client;

    @Shadow
    private ClientLevel level;

    @Shadow
    private int renderDistance;

    @Inject(method = "drawChunkLayer", at = @At("HEAD"), cancellable = true)
    private void valkyrie$cancelSodiumTerrainDraw(
            RenderType renderType,
            ChunkRenderMatrices matrices,
            double x,
            double y,
            double z,
            CallbackInfo ci
    ) {
        if (CompatManager.isSodiumTerrainBypassed()) {
            ci.cancel();
        }
    }

    @Inject(method = "scheduleRebuildForChunk", at = @At("HEAD"))
    private void valkyrie$forwardSectionRebuild(int x, int y, int z, boolean important, CallbackInfo ci) {
        if (CompatManager.isSodiumLoaded()) {
            ValkyrieRenderer.getInstance().scheduleSectionRebuild(x, y, z);
        }
    }

    @Inject(method = "isEntityVisible", at = @At("HEAD"), cancellable = true)
    private void valkyrie$bypassSodiumEntityOcclusion(Entity entity, CallbackInfoReturnable<Boolean> cir) {
        if (CompatManager.isSodiumTerrainBypassed()) {
            cir.setReturnValue(true);
        }
    }

    @Inject(method = "isBoxVisible", at = @At("HEAD"), cancellable = true)
    private void valkyrie$bypassSodiumBoxOcclusion(
            double minX, double minY, double minZ,
            double maxX, double maxY, double maxZ,
            CallbackInfoReturnable<Boolean> cir
    ) {
        if (CompatManager.isSodiumTerrainBypassed()) {
            cir.setReturnValue(true);
        }
    }

    @Inject(
            method = "renderBlockEntities(Lcom/mojang/blaze3d/vertex/PoseStack;Lnet/minecraft/client/renderer/RenderBuffers;Lit/unimi/dsi/fastutil/longs/Long2ObjectMap;FLnet/minecraft/client/renderer/MultiBufferSource$BufferSource;DDDLnet/minecraft/client/renderer/blockentity/BlockEntityRenderDispatcher;Lnet/minecraft/client/player/LocalPlayer;Lcom/llamalad7/mixinextras/sugar/ref/LocalBooleanRef;)V",
            at = @At("HEAD"),
            cancellable = true
    )
    private void valkyrie$renderLoadedChunkBlockEntities(
            PoseStack poseStack,
            RenderBuffers renderBuffers,
            Long2ObjectMap<SortedSet<BlockDestructionProgress>> blockBreakingProgressions,
            float tickDelta,
            MultiBufferSource.BufferSource bufferSource,
            double camX,
            double camY,
            double camZ,
            BlockEntityRenderDispatcher blockEntityRenderer,
            LocalPlayer player,
            LocalBooleanRef isGlowing,
            CallbackInfo ci
    ) {
        if (!CompatManager.isSodiumTerrainBypassed() || this.level == null) {
            return;
        }

        int camChunkX = SectionPos.blockToSectionCoord(camX);
        int camChunkZ = SectionPos.blockToSectionCoord(camZ);
        int dist = Math.max(2, this.renderDistance);

        for (int cz = camChunkZ - dist; cz <= camChunkZ + dist; cz++) {
            for (int cx = camChunkX - dist; cx <= camChunkX + dist; cx++) {
                LevelChunk chunk = this.level.getChunkSource().getChunkNow(cx, cz);
                if (chunk == null || chunk.getBlockEntities().isEmpty()) {
                    continue;
                }
                for (BlockEntity be : chunk.getBlockEntities().values()) {
                    BlockPos pos = be.getBlockPos();
                    MultiBufferSource consumer = bufferSource;
                    poseStack.pushPose();
                    poseStack.translate(pos.getX() - camX, pos.getY() - camY, pos.getZ() - camZ);

                    SortedSet<BlockDestructionProgress> progressSet = blockBreakingProgressions.get(pos.asLong());
                    if (progressSet != null && !progressSet.isEmpty()) {
                        int stage = progressSet.last().getProgress();
                        if (stage >= 0 && stage < ModelBakery.DESTROY_TYPES.size()) {
                            PoseStack.Pose pose = poseStack.last();
                            VertexConsumer overlay = new SheetedDecalTextureGenerator(
                                    renderBuffers.crumblingBufferSource().getBuffer(ModelBakery.DESTROY_TYPES.get(stage)),
                                    pose,
                                    1.0f
                            );
                            consumer = renderType -> {
                                VertexConsumer base = bufferSource.getBuffer(renderType);
                                return renderType.affectsCrumbling() ? VertexMultiConsumer.create(overlay, base) : base;
                            };
                        }
                    }

                    blockEntityRenderer.render(be, tickDelta, poseStack, consumer);
                    poseStack.popPose();
                }
            }
        }
        ci.cancel();
    }

    @Inject(method = "iterateVisibleBlockEntities", at = @At("HEAD"), cancellable = true)
    private void valkyrie$iterateLoadedBlockEntities(Consumer<BlockEntity> consumer, CallbackInfo ci) {
        if (!CompatManager.isSodiumTerrainBypassed() || this.level == null || this.client == null || this.client.cameraEntity == null) {
            return;
        }
        int camChunkX = SectionPos.blockToSectionCoord(this.client.cameraEntity.getX());
        int camChunkZ = SectionPos.blockToSectionCoord(this.client.cameraEntity.getZ());
        int dist = Math.max(2, this.renderDistance);
        for (int cz = camChunkZ - dist; cz <= camChunkZ + dist; cz++) {
            for (int cx = camChunkX - dist; cx <= camChunkX + dist; cx++) {
                LevelChunk chunk = this.level.getChunkSource().getChunkNow(cx, cz);
                if (chunk != null) {
                    for (BlockEntity be : chunk.getBlockEntities().values()) {
                        consumer.accept(be);
                    }
                }
            }
        }
        ci.cancel();
    }
}
