#version 460

// ValkyrieRender - Phase 4 G-Buffer Fragment Shader
// Outputs RT0 (Albedo + Packed Material/Roughness) and RT1 (Octahedral Normal + Packed Light/AO/Emissive).

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vWorldNormal;
layout(location = 2) in vec2 vTiledUV;
layout(location = 3) flat in uint vSpriteId;
layout(location = 4) flat in uint vMaterialMask;
layout(location = 5) in vec4 vLightAO; // x: blockLight, y: skyLight, z: aoLevel, w: windWaving

// Set 0, Binding 6: Texture Atlas UV Entry Table
struct AtlasUVEntry {
    vec2 uvMin; // (u0, v0)
    vec2 uvMax; // (u1, v1)
    vec4 tint;  // (r, g, b, 1.0) in sRGB
};

layout(std430, set = 0, binding = 6) readonly buffer AtlasUVSSBO {
    AtlasUVEntry uvTable[];
};

// Set 0, Binding 7: Block Atlas Combined Image Sampler
layout(set = 0, binding = 7) uniform sampler2D uBlockAtlas;

// Render Targets
layout(location = 0) out vec4 outRT0;  // R8G8B8A8_SRGB: RGB = Albedo, A = packed (materialMask << 4 | roughness)
layout(location = 1) out uvec2 outRT1; // R16G16_UINT: R = 16-bit Oct Normal, G = (blockLight | skyLight<<4 | ao<<8 | emissive<<12)

// Helper: Octahedral normal vector sign
vec2 octSign(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

// Encode 3D unit normal into 16-bit octahedral representation
uint encodeOctNormal16(vec3 n) {
    n = normalize(n);
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * octSign(n.xy);
    oct = oct * 0.5 + 0.5; // Remap [-1, 1] to [0, 1]

    uint r = uint(clamp(oct.x * 255.0 + 0.5, 0.0, 255.0));
    uint g = uint(clamp(oct.y * 255.0 + 0.5, 0.0, 255.0));
    return (r & 0xFFu) | ((g & 0xFFu) << 8);
}

void main() {
    AtlasUVEntry entry = uvTable[vSpriteId];
    vec2 localFrac = fract(vTiledUV);
    vec2 spriteSpan = entry.uvMax - entry.uvMin;

    // Half-texel inward clamp to strictly eliminate atlas bleeding (guarded when spriteSpan < 1 texel)
    vec2 atlasSize = vec2(textureSize(uBlockAtlas, 0));
    vec2 halfTexel = min(0.5 / max(atlasSize, vec2(1.0)), max(spriteSpan, vec2(0.0)) * 0.49);
    vec2 atlasUV = mix(entry.uvMin, entry.uvMax, localFrac);
    vec2 clampedUV = clamp(atlasUV, entry.uvMin + halfTexel, entry.uvMax - halfTexel);

    // Continuous derivatives to eliminate mipmap seams across greedy quad block repeats
    vec2 dUVdx = dFdx(vTiledUV) * spriteSpan;
    vec2 dUVdy = dFdy(vTiledUV) * spriteSpan;

    vec4 albedo = textureGrad(uBlockAtlas, clampedUV, dUVdx, dUVdy);
    if (albedo.a < 0.1) {
        discard;
    }

    // Apply per-sprite/biome tint (converting sRGB tint to linear space)
    albedo.rgb *= pow(max(entry.tint.rgb, vec3(0.0)), vec3(2.2));

    // Material roughness selection (4 bits: 0..15)
    // 0: Matte, 1: Water, 2: Metallic, 3: Emissive, 4: Foliage
    uint roughness4;
    switch (vMaterialMask) {
        case 1u: roughness4 = 1u;  break; // Water (very smooth)
        case 2u: roughness4 = 2u;  break; // Metallic (smooth)
        case 3u: roughness4 = 8u;  break; // Emissive
        case 4u: roughness4 = 11u; break; // Foliage
        case 0u:
        default: roughness4 = 13u; break; // Matte
    }

    // Pack material mask (4 bits) and roughness (4 bits) into 8-bit alpha channel
    uint packedMatRough = ((vMaterialMask & 0xFu) << 4) | (roughness4 & 0xFu);
    float alphaOut = float(packedMatRough) / 255.0;
    outRT0 = vec4(albedo.rgb, alphaOut);

    // Normal encoding (16-bit Octahedral)
    uint packedNormal = encodeOctNormal16(vWorldNormal);

    // Pack light levels: blockLight(4), skyLight(4), ao(4), emissive(4)
    uint bl = uint(clamp(vLightAO.x + 0.5, 0.0, 15.0));
    uint sl = uint(clamp(vLightAO.y + 0.5, 0.0, 15.0));
    uint ao = uint(clamp(vLightAO.z + 0.5, 0.0, 3.0));
    uint emissive = (vMaterialMask == 3u) ? 15u : 0u;

    uint packedLight = (bl & 0xFu)
                     | ((sl & 0xFu) << 4)
                     | ((ao & 0xFu) << 8)
                     | ((emissive & 0xFu) << 12);

    outRT1 = uvec2(packedNormal, packedLight);
}
