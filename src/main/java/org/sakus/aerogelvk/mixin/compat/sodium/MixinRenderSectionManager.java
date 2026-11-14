package org.sakus.aerogelvk.mixin.compat.sodium;

import org.sakus.aerogelvk.compat.CompatManager;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.caffeinemc.mods.sodium.client.render.chunk.RenderSectionManager;
import net.caffeinemc.mods.sodium.client.render.chunk.terrain.TerrainRenderPass;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Prevents Sodium from running CPU chunk meshing threads or uploading/drawing GL terrain buffers
 * while ValkyrieRender handles terrain meshing and Vulkan rendering.
 */
@Mixin(value = RenderSectionManager.class, remap = false)
public class MixinRenderSectionManager {

    @Inject(method = "updateChunks", at = @At("HEAD"), cancellable = true)
    private void valkyrie$skipSodiumChunkMeshing(boolean updateImmediately, CallbackInfo ci) {
        if (CompatManager.isSodiumTerrainBypassed()) {
            ci.cancel();
        }
    }

    @Inject(method = "uploadChunks", at = @At("HEAD"), cancellable = true)
    private void valkyrie$skipSodiumChunkUploads(CallbackInfo ci) {
        if (CompatManager.isSodiumTerrainBypassed()) {
            ci.cancel();
        }
    }

    @Inject(method = "renderLayer", at = @At("HEAD"), cancellable = true)
    private void valkyrie$skipSodiumRenderLayer(
            ChunkRenderMatrices matrices,
            TerrainRenderPass pass,
            double x,
            double y,
            double z,
            CallbackInfo ci
    ) {
        if (CompatManager.isSodiumTerrainBypassed()) {
            ci.cancel();
        }
    }
}
