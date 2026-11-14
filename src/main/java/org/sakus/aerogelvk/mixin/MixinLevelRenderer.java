package org.sakus.aerogelvk.mixin;

import org.sakus.aerogelvk.ValkyrieRenderer;
import org.sakus.aerogelvk.compat.CompatManager;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.LightTexture;
import net.minecraft.client.renderer.RenderType;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(value = LevelRenderer.class, priority = 1100)
public class MixinLevelRenderer {

    @Inject(method = "renderSectionLayer", at = @At("HEAD"), cancellable = true)
    private void onRenderSectionLayer(RenderType renderType, double camX, double camY, double camZ, 
                                      Matrix4f frustumMatrix, Matrix4f projectionMatrix, CallbackInfo ci) {
        if (!CompatManager.shouldYieldToExternalShaderPipeline()
                && ValkyrieRenderer.getInstance().isInitialized()) {
            // Cancel vanilla chunk rendering (Solid, CutoutMipped, Cutout, Translucent)
            ci.cancel();
        }
    }

    @Inject(method = "renderLevel", at = @At(
        value = "INVOKE", 
        target = "Lnet/minecraft/client/renderer/LevelRenderer;renderSectionLayer(Lnet/minecraft/client/renderer/RenderType;DDDLorg/joml/Matrix4f;Lorg/joml/Matrix4f;)V",
        ordinal = 0
    ))
    private void onBeforeTerrainRender(DeltaTracker deltaTracker, boolean renderBlockOutline, Camera camera, 
                                       GameRenderer gameRenderer, LightTexture lightTexture, 
                                       Matrix4f frustumMatrix, Matrix4f projectionMatrix, CallbackInfo ci) {
        if (CompatManager.shouldYieldToExternalShaderPipeline()) {
            return;
        }
        // 1. Execute Vulkan Culling, G-Buffer, Shadow, and Deferred Lighting
        ValkyrieRenderer.getInstance().renderWorld(camera, frustumMatrix, projectionMatrix, deltaTracker);
        
        // 2. Blit Vulkan Color & Depth into Minecraft's Main RenderTarget
        ValkyrieRenderer.getInstance().compositeToOpenGL(Minecraft.getInstance().getMainRenderTarget());
    }

    @Inject(method = "allChanged", at = @At("RETURN"))
    private void onAllChanged(CallbackInfo ci) {
        ValkyrieRenderer.getInstance().onWorldReload();
    }

    @Inject(method = "setSectionDirty(III)V", at = @At("HEAD"), require = 0)
    private void onSetSectionDirty(int sectionX, int sectionY, int sectionZ, CallbackInfo ci) {
        ValkyrieRenderer.getInstance().scheduleSectionRebuild(sectionX, sectionY, sectionZ);
    }
}
