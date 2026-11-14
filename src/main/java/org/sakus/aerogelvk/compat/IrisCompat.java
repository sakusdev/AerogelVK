package org.sakus.aerogelvk.compat;

import net.irisshaders.iris.api.v0.IrisApi;

/**
 * Isolated Iris Shaders integration via the official IrisApi.
 * Loaded only when Iris is present at runtime.
 */
public final class IrisCompat {

    private IrisCompat() {
    }

    /**
     * Returns true when the user currently has an Iris OpenGL shaderpack enabled,
     * signaling Valkyrie to yield world rendering to Iris/Sodium for that frame.
     */
    public static boolean isShaderPackInUse() {
        try {
            return IrisApi.getInstance().isShaderPackInUse();
        } catch (Throwable ignored) {
            return false;
        }
    }
}
