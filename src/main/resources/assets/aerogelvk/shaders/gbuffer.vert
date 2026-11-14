#version 460

// ValkyrieRender - Phase 4 64-bit Vertex Pulling G-Buffer Vertex Shader
// Pulls 8-byte packed vertices from MegaVertexSSBO and projects to screen.

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

// Vertex Outputs to Fragment Shader
layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec2 vTiledUV;
layout(location = 3) flat out uint vSpriteId;
layout(location = 4) flat out uint vMaterialMask;
layout(location = 5) out vec4 vLightAO; // x: blockLight, y: skyLight, z: aoLevel, w: windWaving

// Quad Corner UV lookup table
const vec2 CORNER_UVS[4] = vec2[4](
    vec2(0.0, 0.0), // Corner 0: (0, 0)
    vec2(1.0, 0.0), // Corner 1: (1, 0)
    vec2(1.0, 1.0), // Corner 2: (1, 1)
    vec2(0.0, 1.0)  // Corner 3: (0, 1)
);

// Cube Face Normal Enum lookup table (0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z)
const vec3 FACE_NORMALS[6] = vec3[6](
    vec3( 1.0,  0.0,  0.0), // 0: +X
    vec3(-1.0,  0.0,  0.0), // 1: -X
    vec3( 0.0,  1.0,  0.0), // 2: +Y
    vec3( 0.0, -1.0,  0.0), // 3: -Y
    vec3( 0.0,  0.0,  1.0), // 4: +Z
    vec3( 0.0,  0.0, -1.0)  // 5: -Z
);

void main() {
    // Vertex pulling from MegaVertexSSBO
    // gl_VertexIndex already includes vertexOffset specified in VkDrawIndexedIndirectCommand
    uvec2 packedVertex = vertices[gl_VertexIndex];
    uint dataLow = packedVertex.x;
    uint dataHigh = packedVertex.y;

    // Unpack dataLow (32 bits)
    uint posX        = dataLow & 0x3Fu;
    uint posY        = (dataLow >> 6) & 0x3Fu;
    uint posZ        = (dataLow >> 12) & 0x3Fu;
    uint normalEnum  = (dataLow >> 18) & 0x7u;
    uint aoLevel     = (dataLow >> 21) & 0x3u;
    uint blockLight  = (dataLow >> 23) & 0xFu;
    uint skyLight    = (dataLow >> 27) & 0xFu;
    uint windWaving  = (dataLow >> 31) & 0x1u;

    // Unpack dataHigh (32 bits)
    uint spriteId     = dataHigh & 0xFFFFu;
    uint quadWidth    = (dataHigh >> 16) & 0x1Fu;
    uint quadHeight   = (dataHigh >> 21) & 0x1Fu;
    uint cornerIndex  = (dataHigh >> 26) & 0x3u;
    uint materialMask = (dataHigh >> 28) & 0xFu;

    // Lookup chunk metadata from instance index
    uint chunkIdx = instanceToChunkIdx[gl_InstanceIndex];
    ChunkDrawMeta meta = chunks[chunkIdx];

    // Compute world-space position (half-scale 0.5 for non-full VoxelShape sub-meshes)
    float posScale = ((meta.flags & 4u) != 0u) ? 0.5 : 1.0;
    vec3 sectionOrigin = vec3(meta.chunkPos) * 16.0;
    vec3 worldPos = sectionOrigin + vec3(float(posX), float(posY), float(posZ)) * posScale;

    // Subtle wind displacement for foliage/leaves/water
    if (windWaving == 1u) {
        float wavePhase = FrameUBO.time * 2.5 + worldPos.x * 0.75 + worldPos.z * 0.75;
        float wave = sin(wavePhase);
        float waveY = cos(wavePhase * 0.8);
        worldPos.x += wave * 0.06;
        worldPos.z += wave * 0.06;
        worldPos.y += waveY * 0.02;
    }

    // World normal
    vec3 worldNormal = (normalEnum < 6u) ? FACE_NORMALS[normalEnum] : vec3(0.0, 1.0, 0.0);

    // Compute tiled local UV (cornerU * quadWidth, cornerV * quadHeight)
    vec2 cornerUV = CORNER_UVS[cornerIndex];
    vec2 tiledUV = vec2(cornerUV.x * float(quadWidth), cornerUV.y * float(quadHeight));

    // Output interpolants
    vWorldPos = worldPos;
    vWorldNormal = worldNormal;
    vTiledUV = tiledUV;
    vSpriteId = spriteId;
    vMaterialMask = materialMask;
    vLightAO = vec4(float(blockLight), float(skyLight), float(aoLevel), float(windWaving));

    gl_Position = FrameUBO.viewProj * vec4(worldPos, 1.0);
}
