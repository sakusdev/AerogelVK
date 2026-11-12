package com.example.valkyrie.mixin;

import com.example.valkyrie.ValkyrieRenderer;
import com.mojang.blaze3d.platform.Window;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Window.class)
public class MixinWindow {

    @Inject(method = "onFramebufferResize", at = @At("RETURN"), require = 0)
    private void onFramebufferResize(long window, int width, int height, CallbackInfo ci) {
        ValkyrieRenderer.getInstance().onResize(width, height);
    }

    @Inject(method = "close", at = @At("HEAD"), require = 0)
    private void onClose(CallbackInfo ci) {
        ValkyrieRenderer.getInstance().shutdown();
    }
}
