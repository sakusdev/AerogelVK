package org.sakus.aerogelvk.mixin;

import com.mojang.blaze3d.platform.IconSet;
import com.mojang.blaze3d.platform.Window;
import net.minecraft.client.Minecraft;
import net.minecraft.server.packs.PackResources;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWImage;
import org.lwjgl.stb.STBImage;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.sakus.aerogelvk.ValkyrieRenderer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

@Mixin(Window.class)
public class MixinWindow {

    @Unique
    private static final Logger AEROGELVK_LOGGER = LoggerFactory.getLogger("AerogelVK-Window");

    @Shadow
    @Final
    private long window;

    @Inject(method = "setIcon", at = @At("HEAD"), cancellable = true, require = 0)
    private void onSetIcon(PackResources packResources, IconSet iconSet, CallbackInfo ci) {
        if (Minecraft.ON_OSX || this.window == 0L) {
            return;
        }
        try (InputStream is = MixinWindow.class.getResourceAsStream("/assets/aerogelvk/icon.png")) {
            if (is == null) {
                return;
            }
            byte[] bytes = is.readAllBytes();
            ByteBuffer raw = MemoryUtil.memAlloc(bytes.length);
            ByteBuffer decoded = null;
            ByteBuffer resized = null;
            try (MemoryStack stack = MemoryStack.stackPush()) {
                raw.put(bytes).flip();
                IntBuffer w = stack.mallocInt(1);
                IntBuffer h = stack.mallocInt(1);
                IntBuffer comp = stack.mallocInt(1);
                decoded = STBImage.stbi_load_from_memory(raw, w, h, comp, 4);
                if (decoded == null) {
                    return;
                }
                int srcW = w.get(0);
                int srcH = h.get(0);
                int dstW = srcW;
                int dstH = srcH;
                ByteBuffer iconPixels = decoded;

                // Downsample large icons (e.g. 1024x1024) to <= 256x256 for X11/GLFW compatibility
                if (srcW > 256 || srcH > 256) {
                    dstW = 256;
                    dstH = 256;
                    resized = aerogelvk$downsampleRgba(decoded, srcW, srcH, dstW, dstH);
                    iconPixels = resized;
                }

                GLFWImage.Buffer icons = GLFWImage.malloc(1, stack);
                icons.position(0);
                icons.width(dstW);
                icons.height(dstH);
                icons.pixels(iconPixels);
                GLFW.glfwSetWindowIcon(this.window, icons);
                ci.cancel();
            } finally {
                if (resized != null) {
                    MemoryUtil.memFree(resized);
                }
                if (decoded != null) {
                    STBImage.stbi_image_free(decoded);
                }
                MemoryUtil.memFree(raw);
            }
        } catch (Throwable t) {
            AEROGELVK_LOGGER.warn("[Valkyrie] Failed to apply AerogelVK window icon, using default.", t);
        }
    }

    @Unique
    private static ByteBuffer aerogelvk$downsampleRgba(ByteBuffer src, int srcW, int srcH, int dstW, int dstH) {
        ByteBuffer dst = MemoryUtil.memAlloc(dstW * dstH * 4);
        int scaleX = Math.max(1, srcW / dstW);
        int scaleY = Math.max(1, srcH / dstH);
        int samples = scaleX * scaleY;
        for (int y = 0; y < dstH; y++) {
            int sy0 = (y * srcH) / dstH;
            for (int x = 0; x < dstW; x++) {
                int sx0 = (x * srcW) / dstW;
                int r = 0, g = 0, b = 0, a = 0;
                for (int dy = 0; dy < scaleY; dy++) {
                    int sy = Math.min(srcH - 1, sy0 + dy);
                    for (int dx = 0; dx < scaleX; dx++) {
                        int sx = Math.min(srcW - 1, sx0 + dx);
                        int idx = (sy * srcW + sx) * 4;
                        r += src.get(idx) & 0xFF;
                        g += src.get(idx + 1) & 0xFF;
                        b += src.get(idx + 2) & 0xFF;
                        a += src.get(idx + 3) & 0xFF;
                    }
                }
                int outIdx = (y * dstW + x) * 4;
                dst.put(outIdx, (byte) (r / samples));
                dst.put(outIdx + 1, (byte) (g / samples));
                dst.put(outIdx + 2, (byte) (b / samples));
                dst.put(outIdx + 3, (byte) (a / samples));
            }
        }
        return dst;
    }

    @Inject(method = "onFramebufferResize", at = @At("RETURN"), require = 0)
    private void onFramebufferResize(long window, int width, int height, CallbackInfo ci) {
        ValkyrieRenderer.getInstance().onResize(width, height);
    }

    @Inject(method = "close", at = @At("HEAD"), require = 0)
    private void onClose(CallbackInfo ci) {
        ValkyrieRenderer.getInstance().shutdown();
    }
}
