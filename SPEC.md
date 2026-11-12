# Project Specification: "ValkyrieRender" - Vulkan GPU-Driven Voxel Deferred Renderer for Minecraft 1.21.1

## 1. Overview & Architecture Goals
Minecraft Java Edition (1.21.1) のチャンク描画パイプライン（OpenGL）を完全にバイパスし、LWJGL 3 の Vulkan バインディングを用いた独自の **GPU-Driven Deferred Rendering Pipeline** に置き換える。
バニラおよび他ModのUI・エンティティ描画との互換性を維持するため、地形とシェーディング計算のみを Vulkan で高速処理し、`EXT_external_objects` (`VK_KHR_external_memory`) を介して OpenGL のフレームバッファへ最終合成（Interop）するハイブリッドアーキテクチャを採用する。

### Target Specifications
- **Minecraft Version**: 1.21.1 (NeoForge or Fabric, Java 21)
- **Graphics API**: Vulkan 1.3 (via `org.lwjgl.vulkan`) + OpenGL 4.6 Interop
- **Core Features**:
  - Greedy Meshing + 64-bit (8-byte) 頂点パッキングによるVRAM帯域削減
  - Compute Shader による Frustum / Hi-Z Occlusion Culling と `vkCmdDrawIndexedIndirectCount`
  - 省メモリ帯域 G-Buffer 設計（統合GPU / APU環境でも高FPSを維持）
  - カスケードシャドウマップ（フレーム間キャッシュ機構付き） + Screen Space Reflections (SSR) + 大気散乱

---

## 2. System Architecture & Pipeline Flow

```text
[Minecraft Client Thread]
  ├─ Mixin: Hook Chunk Load / Block Update ──> [Async Meshing Worker Pool (Java Virtual Threads)]
  │                                               └─ Extract PalettedContainer & Light Data
  │                                               └─ Greedy Meshing & 64-bit Vertex Packing
  │                                               └─ Upload to Vulkan Staging Buffer -> Device Local SSBO
  │
  └─ Mixin: Hook LevelRenderer.renderLevel()
       ├─ Cancel Vanilla Terrain Rendering (Solid, Cutout, Translucent)
       ├─ [Vulkan Render Pass]
       │    ├─ 1. Compute Pass: Chunk Culling (Frustum + Hi-Z) -> Generate Indirect Draw Buffer
       │    ├─ 2. Shadow Pass: Render CSM (Skip static cascades if sun angle delta < threshold)
       │    ├─ 3. G-Buffer Pass: `vkCmdDrawIndexedIndirectCount` (Single Draw Call per Region)
       │    ├─ 4. Lighting & SSR Compute Pass: Evaluate PBR, CSM, Godrays, Water SSR -> HDR Color Image
       │    └─ 5. Post-Process Pass: TAA / FXAA + ACES Tonemapping -> Shared Vulkan-GL Color/Depth Texture
       │
       └─ [OpenGL Interop & Composite]
            ├─ Wait on Vulkan Semaphore (`glWaitSemaphoreEXT`)
            ├─ Blit Shared Color & Depth Texture to Vanilla `RenderTarget` (`MainTarget`)
            ├─ Resume Vanilla Rendering (Entities, BlockEntities, Particles, Weather, GUI)
            └─ Signal Vulkan Semaphore on Frame End (`glSignalSemaphoreEXT`)
```

---

## 3. Data Structures & Memory Layout (CRITICAL)

### 3.1. 64-bit Packed Vertex Format (8 Bytes per Quad Vertex)

頂点バッファによるメモリ帯域ボトルネックを防ぐため、通常の頂点属性（32〜48バイト）を廃止し、**1頂点あたり `uint64_t`（または `uvec2` = 8バイト）** に完全パッキングする。各チャンクセクションは `16x16x16` を基準とする。

```glsl
// GLSL Unpacking Reference (Vertex Shader)
// struct PackedVertex { uint dataLow; uint dataHigh; };

// dataLow (32 bits):
// [0..5]   (6 bits) : X position inside chunk section (0..16, fixed point or integer)
// [6..11]  (6 bits) : Y position inside chunk section (0..16)
// [12..17] (6 bits) : Z position inside chunk section (0..16)
// [18..20] (3 bits) : Face Normal Enum (0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z)
// [21..22] (2 bits) : Ambient Occlusion (AO) level (0..3)
// [23..26] (4 bits) : Block Light level (0..15)
// [27..30] (4 bits) : Sky Light level (0..15)
// [31]     (1 bit)  : Wind Waving Flag (Leaves, Grass, Water)

// dataHigh (32 bits):
// [0..15]  (16 bits): Texture Atlas Sprite ID (Index into Uniform/SSBO UV Table)
// [16..20] (5 bits) : Greedy Quad Width (1..16)
// [21..25] (5 bits) : Greedy Quad Height (1..16)
// [26..27] (2 bits) : Corner Index (0..3 for UV interpolation)
// [28..31] (4 bits) : Material Mask (0: Matte, 1: Water, 2: Metallic/Reflective, 3: Emissive, 4: Foliage/SSS)
```

### 3.2. Chunk Metadata SSBO (Per Chunk Section)

```glsl
struct ChunkDrawMeta {
    ivec3 chunkPos;       // World Chunk Section Coordinate (x, y, z)
    uint vertexOffset;    // Offset into Global Mega-Vertex Buffer
    uint quadCount;       // Number of quads in this section
    uint aabbPacked;      // Tight AABB inside section for fine culling
    uint flags;           // Bit 0: HasTranslucent, Bit 1: DirtyShadow
    uint padding;
};
```

### 3.3. Bandwidth-Optimized G-Buffer Layout

統合GPUやミドルレンジGPUのメモリ帯域を圧迫しないよう、G-Bufferは極限までビット数を削る。

| Attachment | Format | Channels & Content | Size (1080p) |
| --- | --- | --- | --- |
| **RT0 (Albedo + Mat)** | `VK_FORMAT_R8G8B8A8_SRGB` | `RGB`: Albedo, `A`: Material ID & Roughness Packed | 8.3 MB |
| **RT1 (Normal + Light)** | `VK_FORMAT_R16G16_UINT` | `R`: Octahedral Encoded Normal (16-bit), `G`: BlockLight(4) + SkyLight(4) + AO(4) + Emissive(4) | 8.3 MB |
| **Depth** | `VK_FORMAT_D32_SFLOAT` | Reversed-Z Hardware Depth Buffer (Shared with OpenGL) | 8.3 MB |

*※ワールド座標（World Position）はG-Bufferに保存せず、Depth値と `InverseViewProjectionMatrix` からコンピュートシェーダー内で復元する。*

---

## 4. Key Implementation Modules (Directory Structure)

```text
src/main/java/com/example/valkyrie/
├── ValkyrieMod.java                  // Mod Entrypoint
├── mixin/
│   ├── MixinLevelRenderer.java       // Cancel terrain pass, inject Vulkan render & composite
│   ├── MixinChunkRebuild.java        // Intercept chunk data updates
│   └── MixinWindow.java              // Manage Vulkan Surface / Interop lifecycle
├── vulkan/
│   ├── VkContext.java                // Instance, PhysicalDevice, Device, Queues, VMA Allocator
│   ├── VkInteropManager.java         // EXT_external_objects, Shared Textures & Semaphores
│   ├── VkMegaBuffer.java             // Persistent Mapped SSBO + Sub-allocator for Chunks
│   ├── VkPipelineManager.java        // Compute, G-Buffer, Shadow, Lighting Pipelines
│   └── VkShaderCompiler.java         // Runtime GLSL -> SPIR-V compilation via Shaderc
└── meshing/
    ├── GreedyMesher.java             // 6-direction Greedy Meshing + AO & Light smoothing
    ├── VertexPacker.java             // Pack vertex attributes into long (64-bit)
    └── AtlasUVTable.java             // Extract Minecraft Block Atlas UV bounds into SSBO
```

---

## 5. Crucial Technical Implementation Details for AI Agent

### 5.1. Mixin Hook Strategy (`MixinLevelRenderer.java`)

バニラの描画を壊さずに地形だけを置き換えるため、`LevelRenderer#renderLevel` 内のチャンク層描画（`renderSectionLayer`）を無効化し、直後にVulkanの出力結果をOpenGLのフレームバッファに書き戻す。

```java
@Mixin(LevelRenderer.class)
public class MixinLevelRenderer {

    @Inject(method = "renderSectionLayer", at = @At("HEAD"), cancellable = true)
    private void onRenderSectionLayer(RenderType renderType, double camX, double camY, double camZ, 
                                      Matrix4f frustumMatrix, Matrix4f projectionMatrix, CallbackInfo ci) {
        // Cancel vanilla chunk rendering (Solid, CutoutMipped, Cutout, Translucent)
        ci.cancel();
    }

    @Inject(method = "renderLevel", at = @At(
        value = "INVOKE", 
        target = "Lnet/minecraft/client/renderer/LevelRenderer;renderSectionLayer(Lnet/minecraft/client/renderer/RenderType;DDDLorg/joml/Matrix4f;Lorg/joml/Matrix4f;)V",
        ordinal = 0
    ))
    private void onBeforeTerrainRender(DeltaTracker deltaTracker, boolean renderBlockOutline, Camera camera, 
                                       GameRenderer gameRenderer, LightTexture lightTexture, 
                                       Matrix4f frustumMatrix, Matrix4f projectionMatrix, CallbackInfo ci) {
        // 1. Execute Vulkan Culling, G-Buffer, Shadow, and Deferred Lighting
        ValkyrieRenderer.getInstance().renderWorld(camera, frustumMatrix, projectionMatrix, deltaTracker);
        
        // 2. Blit Vulkan Color & Depth into Minecraft's Main RenderTarget
        ValkyrieRenderer.getInstance().compositeToOpenGL(Minecraft.getInstance().getMainRenderTarget());
    }
}
```

### 5.2. OpenGL - Vulkan Zero-Copy Interop (`VkInteropManager.java`)

CPUを介したピクセル転送（`glReadPixels` / `vkCmdCopyImageToBuffer`）は絶対に避けること。必ず以下の拡張を使用してGPUメモリ上で直接テクスチャとセマフォを共有する。

1. **Vulkan側拡張**:
* `VK_KHR_external_memory` + Windowsなら `VK_KHR_external_memory_win32`（Linuxなら `VK_KHR_external_memory_fd`）
* `VK_KHR_external_semaphore` + `VK_KHR_external_semaphore_win32`

2. **OpenGL側拡張**:
* `GL_EXT_memory_object` / `GL_EXT_memory_object_win32`
* `GL_EXT_semaphore` / `GL_EXT_semaphore_win32`

3. **Depth Buffer の整合性**:
* Vulkan の最終パスで、シェーディング済みカラーと共にハードウェア Depth を共有テクスチャへ出力する。
* OpenGL側で `glBlitNamedFramebuffer`（`GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT`）を実行し、バニラの `MainTarget` に深度値を転写する。これにより、後続のバニラ処理で描画されるエンティティ（Mob・チェスト・プレイヤーの手）やパーティクルが、Vulkanで描画した地形と完璧に前後判定（隠面消去）される。

### 5.3. Shadow Map Caching (Static / Dynamic Split)

シェーダー負荷の最大の要因であるシャドウマップ生成を最小化するため、2層キャッシュ構造を実装する。

* **Cascade 0 (Near: 0..32 blocks)**: 毎フレーム更新（近くのブロック破壊や葉の揺れを即座に反映）。
* **Cascade 1 & 2 (Mid/Far: 32..192 blocks)**: **4〜8フレームに1回のみ更新**、またはチャンク更新イベント発生時のみ再描画。これにより遠景シャドウのGPU負荷を最大85%削減する。

---

## 6. Step-by-Step Implementation Milestones for Coding Agent

AIエージェントは以下のフェーズ順に実装・ビルド検証を行うこと。一気に全ファイルを生成せず、各フェーズのコンパイルと動作を確認してから次へ進むこと。

* **Phase 1: Build Setup & Vulkan Context Initialization**
* `build.gradle` に `lwjgl-vulkan`, `lwjgl-vma`, `lwjgl-shaderc` の依存関係（およびNatives）を追加。
* ゲーム起動時にヘッドレス（Offscreen）で Vulkan 1.3 インスタンス、Device、VMA（Vulkan Memory Allocator）を初期化する `VkContext` を実装。

* **Phase 2: Vulkan-OpenGL Interop Verification**
* 共有カラーテクスチャと共有セマフォを作成。
* Vulkan の Compute Shader で単色グラデーションを描画し、Minecraft のワールド背景として `MainTarget` に Blit 合成できるか検証。

* **Phase 3: Chunk Extraction & Greedy Mesher**
* チャンクのブロックデータとブロックアトラスのUV座標を抽出。
* Greedy Meshing と 64-bit 頂点パッキングを実装し、Vulkan の Mega-Buffer（SSBO）へ転送。

* **Phase 4: GPU-Driven G-Buffer & Depth Composite**
* Compute Culling + `vkCmdDrawIndexedIndirectCount` で地形をG-Bufferに描画。
* Depth を OpenGL に転写し、バニラのエンティティが Vulkan 地形の上に正しくオクルージョンされて描画されることを確認。

* **Phase 5: Deferred Lighting, CSM, and Water SSR**
* G-Buffer を読み取るライティング Compute Shader、カスケードシャドウマップ、水面のレイマーチング反射（SSR）、大気フォグを実装。
