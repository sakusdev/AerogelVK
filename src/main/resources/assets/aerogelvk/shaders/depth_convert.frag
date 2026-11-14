#version 460

// ValkyrieRender - Converts internal Reversed-Z G-Buffer depth (1=near, 0=far)
// into Standard-Z OpenGL-compatible depth (0=near, 1=far) in the shared interop depth buffer.

layout(set = 1, binding = 2) uniform sampler2D uDepthBuffer;

void main() {
    float revZ = texelFetch(uDepthBuffer, ivec2(gl_FragCoord.xy), 0).r;
    gl_FragDepth = clamp(1.0 - revZ, 0.0, 1.0);
}
