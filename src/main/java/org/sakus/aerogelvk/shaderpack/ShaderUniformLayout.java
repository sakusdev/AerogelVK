package org.sakus.aerogelvk.shaderpack;

import org.joml.Matrix4f;
import org.joml.Vector2i;
import org.joml.Vector3f;
import org.joml.Vector4f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Defines the {@code std140} Vulkan uniform buffer layout ({@code ShaderPackUBO}) and
 * provides a deterministic Java {@link ByteBuffer} packing helper for OptiFine/Iris uniforms.
 */
public final class ShaderUniformLayout {

    /** Total byte size reserved for the host-coherent ShaderPack UBO. */
    public static final int UBO_SIZE_BYTES = 1024;

    private static final int ACTIVE_BYTES = 928;

    private static final int OFF_GBUFFER_MV = 0;
    private static final int OFF_GBUFFER_MV_INV = 64;
    private static final int OFF_GBUFFER_PREV_MV = 128;
    private static final int OFF_GBUFFER_PROJ = 192;
    private static final int OFF_GBUFFER_PROJ_INV = 256;
    private static final int OFF_GBUFFER_PREV_PROJ = 320;
    private static final int OFF_SHADOW_MV = 384;
    private static final int OFF_SHADOW_MV_INV = 448;
    private static final int OFF_SHADOW_PROJ = 512;
    private static final int OFF_SHADOW_PROJ_INV = 576;

    private static final int OFF_CAMERA_POS = 640;
    private static final int OFF_PREV_CAMERA_POS = 656;
    private static final int OFF_SUN_POS = 672;
    private static final int OFF_MOON_POS = 688;
    private static final int OFF_SHADOW_LIGHT_POS = 704;
    private static final int OFF_UP_POS = 720;
    private static final int OFF_SKY_COLOR = 736;
    private static final int OFF_FOG_COLOR = 752;
    private static final int OFF_ENTITY_COLOR = 768;

    private static final int OFF_VIEW_WIDTH = 784;
    private static final int OFF_VIEW_HEIGHT = 788;
    private static final int OFF_ASPECT_RATIO = 792;
    private static final int OFF_NEAR = 796;
    private static final int OFF_FAR = 800;
    private static final int OFF_FRAME_TIME = 804;
    private static final int OFF_FRAME_TIME_COUNTER = 808;
    private static final int OFF_WORLD_TIME = 812;
    private static final int OFF_SUN_ANGLE = 816;
    private static final int OFF_RAIN_STRENGTH = 820;
    private static final int OFF_WETNESS = 824;
    private static final int OFF_IS_EYE_IN_WATER = 828;
    private static final int OFF_FRAME_COUNTER = 832;
    private static final int OFF_NIGHT_VISION = 836;
    private static final int OFF_EYE_BRIGHTNESS = 840;
    private static final int OFF_EYE_BRIGHTNESS_SMOOTH = 848;
    private static final int OFF_BLINDNESS = 856;
    private static final int OFF_DARKNESS_FACTOR = 860;
    private static final int OFF_HELD_BLOCK_LIGHT = 864;
    private static final int OFF_HELD_ITEM_ID = 868;
    private static final int OFF_ALPHA_TEST_REF = 872;
    private static final int OFF_CENTER_DEPTH_SMOOTH = 876;
    private static final int OFF_SHADOW_ANGLE = 880;
    private static final int OFF_SCREEN_BRIGHTNESS = 884;
    private static final int OFF_FOG_START = 888;
    private static final int OFF_FOG_END = 892;
    private static final int OFF_FOG_DENSITY = 896;
    private static final int OFF_WORLD_DAY = 900;
    private static final int OFF_MOON_PHASE = 904;
    private static final int OFF_HELD_BLOCK_LIGHT_2 = 908;
    private static final int OFF_HELD_ITEM_ID_2 = 912;

    /** GLSL 460 {@code std140} uniform block matching {@link #pack(ByteBuffer, UniformData)}. */
    public static final String GLSL_UBO_BLOCK =
            "layout(std140, set = " + ShaderDescriptorBindings.SET_GLOBAL
                    + ", binding = " + ShaderDescriptorBindings.BINDING_SHADER_PACK_UBO + ") uniform ShaderPackUBO {\n"
                    + "    mat4 gbufferModelView;\n"
                    + "    mat4 gbufferModelViewInverse;\n"
                    + "    mat4 gbufferPreviousModelView;\n"
                    + "    mat4 gbufferProjection;\n"
                    + "    mat4 gbufferProjectionInverse;\n"
                    + "    mat4 gbufferPreviousProjection;\n"
                    + "    mat4 shadowModelView;\n"
                    + "    mat4 shadowModelViewInverse;\n"
                    + "    mat4 shadowProjection;\n"
                    + "    mat4 shadowProjectionInverse;\n"
                    + "    vec3 cameraPosition; float _pad0;\n"
                    + "    vec3 previousCameraPosition; float _pad1;\n"
                    + "    vec3 sunPosition; float _pad2;\n"
                    + "    vec3 moonPosition; float _pad3;\n"
                    + "    vec3 shadowLightPosition; float _pad4;\n"
                    + "    vec3 upPosition; float _pad5;\n"
                    + "    vec3 skyColor; float _pad6;\n"
                    + "    vec3 fogColor; float _pad7;\n"
                    + "    vec4 entityColor;\n"
                    + "    float viewWidth;\n"
                    + "    float viewHeight;\n"
                    + "    float aspectRatio;\n"
                    + "    float near;\n"
                    + "    float far;\n"
                    + "    float frameTime;\n"
                    + "    float frameTimeCounter;\n"
                    + "    int worldTime;\n"
                    + "    float sunAngle;\n"
                    + "    float rainStrength;\n"
                    + "    float wetness;\n"
                    + "    int isEyeInWater;\n"
                    + "    int frameCounter;\n"
                    + "    float nightVision;\n"
                    + "    ivec2 eyeBrightness;\n"
                    + "    ivec2 eyeBrightnessSmooth;\n"
                    + "    float blindness;\n"
                    + "    float darknessFactor;\n"
                    + "    int heldBlockLightValue;\n"
                    + "    int heldItemId;\n"
                    + "    float alphaTestRef;\n"
                    + "    float centerDepthSmooth;\n"
                    + "    float shadowAngle;\n"
                    + "    float screenBrightness;\n"
                    + "    float fogStart;\n"
                    + "    float fogEnd;\n"
                    + "    float fogDensity;\n"
                    + "    int worldDay;\n"
                    + "    int moonPhase;\n"
                    + "    int heldBlockLightValue2;\n"
                    + "    int heldItemId2;\n"
                    + "    float _padEnd0;\n"
                    + "    vec2 _padEnd1;\n"
                    + "} vk_ubo;\n";

    /** All GLSL uniform member names exposed inside {@code vk_ubo}. */
    public static final Set<String> UBO_FIELD_NAMES;

    static {
        Set<String> fields = new LinkedHashSet<>();
        Collections.addAll(fields,
                "gbufferModelView", "gbufferModelViewInverse", "gbufferPreviousModelView",
                "gbufferProjection", "gbufferProjectionInverse", "gbufferPreviousProjection",
                "shadowModelView", "shadowModelViewInverse",
                "shadowProjection", "shadowProjectionInverse",
                "cameraPosition", "previousCameraPosition",
                "sunPosition", "moonPosition", "shadowLightPosition", "upPosition",
                "skyColor", "fogColor", "entityColor",
                "viewWidth", "viewHeight", "aspectRatio", "near", "far",
                "frameTime", "frameTimeCounter", "worldTime", "sunAngle",
                "rainStrength", "wetness", "isEyeInWater", "frameCounter",
                "nightVision", "eyeBrightness", "eyeBrightnessSmooth",
                "blindness", "darknessFactor", "heldBlockLightValue", "heldItemId",
                "alphaTestRef", "centerDepthSmooth", "shadowAngle", "screenBrightness",
                "fogStart", "fogEnd", "fogDensity", "worldDay", "moonPhase",
                "heldBlockLightValue2", "heldItemId2"
        );
        UBO_FIELD_NAMES = Collections.unmodifiableSet(fields);
    }

    private ShaderUniformLayout() {
    }

    /**
     * Mutable container for a frame's OptiFine/Iris shaderpack uniforms.
     * Any null inverse matrix is automatically computed from its forward matrix during {@link #pack}.
     */
    public static final class UniformData {
        public final Matrix4f gbufferModelView = new Matrix4f();
        public Matrix4f gbufferModelViewInverse;
        public final Matrix4f gbufferPreviousModelView = new Matrix4f();
        public final Matrix4f gbufferProjection = new Matrix4f();
        public Matrix4f gbufferProjectionInverse;
        public final Matrix4f gbufferPreviousProjection = new Matrix4f();

        public final Matrix4f shadowModelView = new Matrix4f();
        public Matrix4f shadowModelViewInverse;
        public final Matrix4f shadowProjection = new Matrix4f();
        public Matrix4f shadowProjectionInverse;

        public final Vector3f cameraPosition = new Vector3f();
        public final Vector3f previousCameraPosition = new Vector3f();
        public final Vector3f sunPosition = new Vector3f(0.0f, 100.0f, 0.0f);
        public final Vector3f moonPosition = new Vector3f(0.0f, -100.0f, 0.0f);
        public final Vector3f shadowLightPosition = new Vector3f(0.0f, 100.0f, 0.0f);
        public final Vector3f upPosition = new Vector3f(0.0f, 1.0f, 0.0f);
        public final Vector3f skyColor = new Vector3f(0.45f, 0.68f, 1.0f);
        public final Vector3f fogColor = new Vector3f(0.68f, 0.80f, 0.96f);
        public final Vector4f entityColor = new Vector4f(0.0f, 0.0f, 0.0f, 0.0f);

        public float viewWidth = 1280.0f;
        public float viewHeight = 720.0f;
        public float aspectRatio = 1280.0f / 720.0f;
        public float near = 0.05f;
        public float far = 256.0f;
        public float frameTime = 1.0f / 60.0f;
        public float frameTimeCounter = 0.0f;
        public int worldTime = 6000;
        public float sunAngle = 0.25f;
        public float rainStrength = 0.0f;
        public float wetness = 0.0f;
        public int isEyeInWater = 0;
        public int frameCounter = 0;
        public float nightVision = 0.0f;
        public final Vector2i eyeBrightness = new Vector2i(240, 240);
        public final Vector2i eyeBrightnessSmooth = new Vector2i(240, 240);
        public float blindness = 0.0f;
        public float darknessFactor = 0.0f;
        public int heldBlockLightValue = 0;
        public int heldItemId = 0;

        public float alphaTestRef = 0.1f;
        public float centerDepthSmooth = 0.5f;
        public float shadowAngle = 0.25f;
        public float screenBrightness = 0.5f;
        public float fogStart = 64.0f;
        public float fogEnd = 256.0f;
        public float fogDensity = 0.0f;
        public int worldDay = 0;
        public int moonPhase = 0;
        public int heldBlockLightValue2 = 0;
        public int heldItemId2 = 0;
    }

    /**
     * Packs {@code data} into {@code dst} starting at offset 0 using little-endian {@code std140} layout.
     *
     * @param dst  target buffer with at least {@link #UBO_SIZE_BYTES} capacity from offset 0
     * @param data frame uniform values
     */
    public static void pack(ByteBuffer dst, UniformData data) {
        pack(dst, 0, data);
    }

    /**
     * Packs {@code data} into {@code dst} starting at {@code baseOffset} using little-endian {@code std140} layout.
     *
     * @param dst        target buffer with at least {@code baseOffset + UBO_SIZE_BYTES} capacity
     * @param baseOffset byte offset within {@code dst}
     * @param data       frame uniform values
     */
    public static void pack(ByteBuffer dst, int baseOffset, UniformData data) {
        if (dst == null) {
            throw new IllegalArgumentException("dst must not be null");
        }
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        if (baseOffset < 0 || baseOffset + UBO_SIZE_BYTES > dst.capacity()) {
            throw new IllegalArgumentException(
                    "Buffer capacity " + dst.capacity() + " too small for offset " + baseOffset + " + " + UBO_SIZE_BYTES);
        }

        dst.order(ByteOrder.LITTLE_ENDIAN);

        Matrix4f gbufferMvInv = data.gbufferModelViewInverse != null
                ? data.gbufferModelViewInverse
                : safeInvert(data.gbufferModelView);
        Matrix4f gbufferProjInv = data.gbufferProjectionInverse != null
                ? data.gbufferProjectionInverse
                : safeInvert(data.gbufferProjection);
        Matrix4f shadowMvInv = data.shadowModelViewInverse != null
                ? data.shadowModelViewInverse
                : safeInvert(data.shadowModelView);
        Matrix4f shadowProjInv = data.shadowProjectionInverse != null
                ? data.shadowProjectionInverse
                : safeInvert(data.shadowProjection);

        data.gbufferModelView.get(baseOffset + OFF_GBUFFER_MV, dst);
        gbufferMvInv.get(baseOffset + OFF_GBUFFER_MV_INV, dst);
        data.gbufferPreviousModelView.get(baseOffset + OFF_GBUFFER_PREV_MV, dst);
        data.gbufferProjection.get(baseOffset + OFF_GBUFFER_PROJ, dst);
        gbufferProjInv.get(baseOffset + OFF_GBUFFER_PROJ_INV, dst);
        data.gbufferPreviousProjection.get(baseOffset + OFF_GBUFFER_PREV_PROJ, dst);
        data.shadowModelView.get(baseOffset + OFF_SHADOW_MV, dst);
        shadowMvInv.get(baseOffset + OFF_SHADOW_MV_INV, dst);
        data.shadowProjection.get(baseOffset + OFF_SHADOW_PROJ, dst);
        shadowProjInv.get(baseOffset + OFF_SHADOW_PROJ_INV, dst);

        putVec3Padded(dst, baseOffset + OFF_CAMERA_POS, data.cameraPosition);
        putVec3Padded(dst, baseOffset + OFF_PREV_CAMERA_POS, data.previousCameraPosition);
        putVec3Padded(dst, baseOffset + OFF_SUN_POS, data.sunPosition);
        putVec3Padded(dst, baseOffset + OFF_MOON_POS, data.moonPosition);
        putVec3Padded(dst, baseOffset + OFF_SHADOW_LIGHT_POS, data.shadowLightPosition);
        putVec3Padded(dst, baseOffset + OFF_UP_POS, data.upPosition);
        putVec3Padded(dst, baseOffset + OFF_SKY_COLOR, data.skyColor);
        putVec3Padded(dst, baseOffset + OFF_FOG_COLOR, data.fogColor);

        dst.putFloat(baseOffset + OFF_ENTITY_COLOR, data.entityColor.x);
        dst.putFloat(baseOffset + OFF_ENTITY_COLOR + 4, data.entityColor.y);
        dst.putFloat(baseOffset + OFF_ENTITY_COLOR + 8, data.entityColor.z);
        dst.putFloat(baseOffset + OFF_ENTITY_COLOR + 12, data.entityColor.w);

        float safeHeight = Math.max(1.0f, data.viewHeight);
        float aspect = data.aspectRatio > 0.0f ? data.aspectRatio : (Math.max(1.0f, data.viewWidth) / safeHeight);

        dst.putFloat(baseOffset + OFF_VIEW_WIDTH, data.viewWidth);
        dst.putFloat(baseOffset + OFF_VIEW_HEIGHT, data.viewHeight);
        dst.putFloat(baseOffset + OFF_ASPECT_RATIO, aspect);
        dst.putFloat(baseOffset + OFF_NEAR, data.near);
        dst.putFloat(baseOffset + OFF_FAR, data.far);
        dst.putFloat(baseOffset + OFF_FRAME_TIME, data.frameTime);
        dst.putFloat(baseOffset + OFF_FRAME_TIME_COUNTER, data.frameTimeCounter);
        dst.putInt(baseOffset + OFF_WORLD_TIME, data.worldTime);
        dst.putFloat(baseOffset + OFF_SUN_ANGLE, data.sunAngle);
        dst.putFloat(baseOffset + OFF_RAIN_STRENGTH, data.rainStrength);
        dst.putFloat(baseOffset + OFF_WETNESS, data.wetness);
        dst.putInt(baseOffset + OFF_IS_EYE_IN_WATER, data.isEyeInWater);
        dst.putInt(baseOffset + OFF_FRAME_COUNTER, data.frameCounter);
        dst.putFloat(baseOffset + OFF_NIGHT_VISION, data.nightVision);

        dst.putInt(baseOffset + OFF_EYE_BRIGHTNESS, data.eyeBrightness.x);
        dst.putInt(baseOffset + OFF_EYE_BRIGHTNESS + 4, data.eyeBrightness.y);
        dst.putInt(baseOffset + OFF_EYE_BRIGHTNESS_SMOOTH, data.eyeBrightnessSmooth.x);
        dst.putInt(baseOffset + OFF_EYE_BRIGHTNESS_SMOOTH + 4, data.eyeBrightnessSmooth.y);

        dst.putFloat(baseOffset + OFF_BLINDNESS, data.blindness);
        dst.putFloat(baseOffset + OFF_DARKNESS_FACTOR, data.darknessFactor);
        dst.putInt(baseOffset + OFF_HELD_BLOCK_LIGHT, data.heldBlockLightValue);
        dst.putInt(baseOffset + OFF_HELD_ITEM_ID, data.heldItemId);
        dst.putFloat(baseOffset + OFF_ALPHA_TEST_REF, data.alphaTestRef);
        dst.putFloat(baseOffset + OFF_CENTER_DEPTH_SMOOTH, data.centerDepthSmooth);
        dst.putFloat(baseOffset + OFF_SHADOW_ANGLE, data.shadowAngle);
        dst.putFloat(baseOffset + OFF_SCREEN_BRIGHTNESS, data.screenBrightness);
        dst.putFloat(baseOffset + OFF_FOG_START, data.fogStart);
        dst.putFloat(baseOffset + OFF_FOG_END, data.fogEnd);
        dst.putFloat(baseOffset + OFF_FOG_DENSITY, data.fogDensity);
        dst.putInt(baseOffset + OFF_WORLD_DAY, data.worldDay);
        dst.putInt(baseOffset + OFF_MOON_PHASE, data.moonPhase);
        dst.putInt(baseOffset + OFF_HELD_BLOCK_LIGHT_2, data.heldBlockLightValue2);
        dst.putInt(baseOffset + OFF_HELD_ITEM_ID_2, data.heldItemId2);

        // Zero trailing padding bytes deterministically
        for (int off = OFF_HELD_ITEM_ID_2 + 4; off < UBO_SIZE_BYTES; off += 4) {
            dst.putInt(baseOffset + off, 0);
        }
    }

    private static void putVec3Padded(ByteBuffer dst, int offset, Vector3f v) {
        dst.putFloat(offset, v.x);
        dst.putFloat(offset + 4, v.y);
        dst.putFloat(offset + 8, v.z);
        dst.putFloat(offset + 12, 0.0f);
    }

    private static Matrix4f safeInvert(Matrix4f m) {
        if (Math.abs(m.determinant()) < 1e-12f) {
            return new Matrix4f().identity();
        }
        return new Matrix4f(m).invert();
    }
}
