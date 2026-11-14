#version 460

// ValkyrieRender - Phase 5 Cascaded Shadow Map (CSM) Vertex Shader
// Projects chunk geometry into light space using FrameUBO.csmMatrices[cascadeIndex].

// Set 0, Binding 0: Frame Uniform Buffer
layout(std140, set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invViewProj;
    mat4 prevViewProj;
    mat4 csmMatrices[3];
    vec4 sunDir;
    vec4 cameraPos;
    vec2 viewportSize;
    uint frameIndex;
    float time;
    uint renderFlags;
    uint activeCascades;
    float csmFarDist;
    float reserved0;
} FrameUBO;

// Set 0, Binding 1: Mega-Vertex Arena SSBO (64-bit / uvec2 per vertex)
layout(std430, set = 0, binding = 1) readonly buffer MegaVertexSSBO {
    uvec2 vertices[];
};

// Set 0, Binding 2: Chunk Metadata SSBO
struct ChunkDrawMeta {
    ivec3 chunkPos;
    uint vertexOffset;
    uint quadCount;
    uint aabbPacked;
    uint flags;
    uint padding;
};

layout(std430, set = 0, binding = 2) readonly buffer ChunkMetaSSBO {
    ChunkDrawMeta chunks[];
};

// Set 0, Binding 5: Instance Index to Chunk Metadata Index mapping
layout(std430, set = 0, binding = 5) readonly buffer VisibleChunkIdSSBO {
    uint instanceToChunkIdx[];
};

// Push constant for selecting the active shadow cascade and culling out-of-range sections
layout(push_constant) uniform ShadowPushConstants {
    uint cascadeIndex;
    float maxCascadeDistSq;
} pc;

void main() {
    // Lookup chunk section position first for early cascade range rejection
    uint chunkIdx = instanceToChunkIdx[gl_InstanceIndex];
    ChunkDrawMeta meta = chunks[chunkIdx];
    vec3 sectionOrigin = vec3(meta.chunkPos) * 16.0;

    if (pc.maxCascadeDistSq > 0.0) {
        vec3 delta = (sectionOrigin + vec3(8.0)) - FrameUBO.cameraPos.xyz;
        if (dot(delta, delta) > pc.maxCascadeDistSq) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            return;
        }
    }

    uvec2 packedVertex = vertices[gl_VertexIndex];
    uint dataLow = packedVertex.x;

    // Unpack vertex coordinates and wind flag
    uint posX       = dataLow & 0x3Fu;
    uint posY       = (dataLow >> 6) & 0x3Fu;
    uint posZ       = (dataLow >> 12) & 0x3Fu;
    uint windWaving = (dataLow >> 31) & 0x1u;

    vec3 worldPos = sectionOrigin + vec3(float(posX), float(posY), float(posZ));

    // Dynamic wind sway (consistent with G-Buffer pass)
    if (windWaving == 1u) {
        float wavePhase = FrameUBO.time * 2.5 + worldPos.x * 0.75 + worldPos.z * 0.75;
        float wave = sin(wavePhase);
        worldPos.x += wave * 0.06;
        worldPos.z += wave * 0.06;
    }

    // Project into light space for current cascade
    gl_Position = FrameUBO.csmMatrices[pc.cascadeIndex] * vec4(worldPos, 1.0);
}
