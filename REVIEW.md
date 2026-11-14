# AerogelVK / ValkyrieRender レビュー報告書（2026-11-14、更新版）

対象: https://github.com/sakusdev/AerogelVK （HEAD `4857f4a`）
手法: 全 Java 30ファイル＋GLSL 10本＋テストの精読、実ビルド検証（Temurin JDK 21.0.13、sakusorg）、実シェーダーパック14種での変換・ライブ描画の実測

---

## 総評

骨組みではない。Vulkan 初期化、EXT_external_objects による GL/VK ゼロコピー共有（FD/Win32 両対応、デバイス UUID 照合つき）、フリーリスト付き 256MB 頂点アリーナ、GPU カリング、CSM、Deferred Lighting、Hi-Z まで実コードで存在し、変換器も正規表現パッチではなく字句解析＋スコープ追跡つきのトークンレベル変換器（2,377行）。土台は「AI が書いた雛形」の水準を明確に超えている。

弱点は3つに集約される。
1. **描画の正しさ**：非キューブ形状の無視、半透明パスの欠落、シェーダーパックの一部パス未実行
2. **同期の厳密さ**：フレーム間バッファ再利用のレース、フェンス処理、GL/VK セマフォ条件
3. **検証の甘さ**：後述の通り、シェーダーパック検証スイートは「実パックが真っ黒」でも合格してしまう作りになっており、実機の問題を検出できていなかった

パフォーマンスと Vulkan の恩恵については末尾に別節を設けた。結論だけ先に言うと、現時点で「バニラより何倍速い」と言える実測は存在せず（ベンチは全て llvmpipe のソフトウェア描画）、設計上の利点は「GPU-driven 化」のものであって、Vulkan 固有の並列性（マルチスレッド記録、非同期オーバーラップ）はまだ回収できていない。

---

## ビルド検証（実測）

- `compileJava` / `jar` / `remapJar`: **成功**。MOD jar はビルドできる
- `./gradlew build`: **失敗**。`:test` が「テストソースはあるが実行可能なテストが見つからない」で落ちる。2つの検証スイートは JUnit ではなく main クラス方式で、Gradle の test タスクに拾われない。通常ビルドがデフォルトで赤いのが実態で、`verifyValkyrie` を明示実行したときだけ検証が走る。是正すること（verifyValkyrie を check に組み込むか、failOnNoDiscoveredTests を切る）
- `verifyValkyrie`: **合格**（BUILD SUCCESSFUL）。llvmpipe 上で Vulkan 1.3 + VMA + 256MB メガバッファを初期化し、合成シェーダーパックを実際にオフスクリーン描画してピクセル読み戻しで検証（4096/4096 非黒、相違色 1025、深度 0.9933）。1280×720・30フレームのベンチで平均 45.56 FPS（ソフトウェア描画での値。GPU 性能の指標にはならない）
- バージョン表記: レポート側の「Gradle 9.4.0 + Loom 1.16.3」に対し実物は **Gradle 9.5.0 + Loom 1.17.21**。これはテストビルド時に落ちたため組み合わせを変更した結果であり、コード側の問題ではない（報告文の更新漏れ）。また `runProdCompatSmokeClient` 内の fabric-api が `0.116.11+1.21.1` ハードコードで、gradle.properties の `0.105.0+1.21.1` と不一致

---

## 重大な指摘

### B1. シェーダーパック記録中の例外フォールバックが、記録中の同一コマンドバッファへ継続記録する
`ValkyrieRenderer.java:425-445` — `recordFrame(...)` が例外を投げると catch で `unloadPack()`（vkDeviceWaitIdle＋パイプライン破棄）を呼んだ直後、まだ記録中（vkEndCommandBuffer 前）の同じコマンドバッファへ `recordGpuDrivenTerrainPasses(...)` を続けて記録する。動的レンダリング開始後に失敗した場合、レンダリングスコープが開いたまま別パス記録が始まり、検証エラー／デバイス喪失になりうる。破棄済みパイプラインを参照する記録済みコマンドが submit される問題もある。フォールバックするならコマンドバッファをリセットして最初から記録し直すべき

### B2. メッシュ再アップロード時、旧頂点領域を即時解放→即再利用（フレーム間レース）
`vulkan/VkMegaBuffer.java:283-288` — `uploadSectionMesh` は旧割当を即座にフリーリストへ戻し、同じ領域が別セクションへ再割当されうる。前フレーム（FRAMES_IN_FLIGHT=2 の別スロット）が同じ領域をまだ描画している可能性があり、待機なしで上書きされる。フレーム単位の遅延解放（retirement queue）が標準解法

### B3. フェンス待ちのタイムアウト結果を無視してリセット
`ValkyrieRenderer.java:277-279` — `vkWaitForFences(..., 1_000_000_000L)` の戻り値を確認せず無条件で `vkResetFences`。GPU 処理が1秒を超えると未完了フェンスのリセット＝仕様違反で、その後の submit が未定義動作

### B4. GL/VK セマフォ同期が「前フレームで composite が実行された場合のみ」成立
`ValkyrieRenderer.java:370-381` — Vulkan submit は `consumeGlSemaphoreSignaled()` が true のときだけ GL 完了セマフォを待つ。composite がスキップ／失敗したフレームや初回フレームでは、GL が共有イメージを読んでいる最中に Vulkan が上書きしうる。加えて `VkInteropManager.java:858` の `glWaitSemaphoreEXT` は実レイアウトと無関係に `GL_LAYOUT_TRANSFER_SRC_EXT` 固定申告、`:954` の signal 側も固定。ドライバ依存で壊れる余地あり

### B5. 非フルブロック形状をすべてフルキューブとしてメッシュ化し、ブロックエンティティを二重描画
`meshing/GreedyMesher.java:242` 付近 — スキップ条件は `RenderShape.INVISIBLE` のみで `VoxelShape` は未使用。階段・ハーフブロック・フェンス・作物はフルキューブ面になる。チェスト／ベッド／看板などの `ENTITYBLOCK_ANIMATED` もキューブ化され、バニラの正規モデル描画と重なってゴーストキューブが出る。最も目立つ正確性の欠陥

### B6. 半透明パスが存在せず、`flags` の HasTranslucent は死にデータ
ChunkDrawMeta の `flags` は全シェーダーで宣言のみで読まれない（`chunk_cull.comp:33`、`csm_shadow.vert:36`、`gbuffer.vert:36`）。描画は不透明 G-Buffer 単一パスのみ。水・ガラス・氷・ステンドグラスはすべて不透明描画。SPEC §3.2 が設計した flags の用途自体が未実装

### B7. シェーダーパックの `gbuffers_water` はコンパイルされるが一度も実行されない
`shaderpack/VkShaderPackPipeline.java` — compilePack は water パスをコンパイルするが recordFrame には明示コメント「we do not re-draw all opaque quads with gbuffersWaterPass on top.」があり実行呼び出しなし。BSL 等の水専用処理（屈折・波法線・専用フォグ）は効かない。さらに `gbuffers_skybasic` / `gbuffers_entities` / `gbuffers_hand` / `gbuffers_blockentities` は未対応（該当文字列自体が存在しない）。「BSL 系パックがそのまま動く」は部分的にのみ正しい

---

## 中程度の指摘

- **C1. CSM と G-Buffer がカメラ視錐カリング結果を共有**（`ValkyrieRenderer.java:618-690`）— 視錐外だがシャドウ範囲内にある太陽側の遮蔽物が影マップに入らず、視錐端や背後で影が欠落する
- **C2. FrameUBO が単一バッファ**（`VkPipelineManager.createFrameUbo`）— 2フレーム in-flight と共有され、前フレームが読んでいる間に行列を上書き。フレームスロットごとのリング化が定石。後述「Vulkan の恩恵」とも直結する
- **C3. 共有テクスチャ破棄に GPU 待機なし**（`VkInteropManager.java:1238`）— リサイズ経路から呼ばれ、使用中イメージ破棄で検証エラー／クラッシュの原因になる
- **C4. per-section DirtyShadow フラグが恒常的に false**（`GreedyMesher.java:204`）— 一度も true にならず `:343` の分岐は到達不能。SPEC §3.2 の設計が死に機能（CSM 間引き自体はレンダラ側フラグで別途動作）
- **C5. 同種ブロック間の面を常にカリング**（`GreedyMesher.java:261`）— 葉ブロック同士も対象で、バニラのファンシー設定（葉の内面描画）と見た目が変わる
- **C6. バイオーム tint が平原色フォールバック固定**（`AtlasUVTable.java:189-211`）— `BlockColors.getColor(state, null, null, ...)` で草 `0x7CBD52`、葉 `0x59AE30` に落ちる。沼地・雪原・砂漠で色が異なる。溶岩の特殊扱いなし
- **C7. STAGING_READBACK フォールバックは毎フレーム全画面 CPU リードバック**（`VkInteropManager.java:861-896`）— interop 不可環境では全画面を CPU 経由転送。SPEC §5.2「CPU 転送は絶対に避ける」と矛盾するだけでなく、その環境ではバニラより遅くなる
- **C8. ステージングリング超過時にフレーム途中で GPU 全待機**（`VkMegaBuffer.java:382-390`）— 32MB リング溢れで単発送信＋完全待機。大量チャンク読み込み時にフレームスパイク
- **C9. Sodium mixin が特定バージョンの内部シグネチャに強結合**（`MixinSodiumWorldRenderer`）— ゲートは「sodium が存在するか」だけでバージョンを見ず、`defaultRequire: 1` のため Sodium 更新で起動時 InjectionError になりうる
- **C10. Vulkan 初期化失敗時も地形キャンセルが継続**（`MixinLevelRenderer.java:22-28`）— デバイス確保に失敗すると黒い世界になる。フォールバック経路なし
- **C11. トランスパイラの「黙ってゼロ化」範囲が広い**（`GlslToVulkanTranspiler`）— 未知 uniform はゼロ定数化、未知の頂点属性もゼロ初期化。パック固有 uniform を使うシェーダーはコンパイルは通るが見た目が静かに壊れる。vsh/fsh 以外のステージは対象外
- **C12. 既定カスケード数が SPEC と異なる**（`ValkyrieConfig.java:35` の `csmCascades = 2`）— SPEC §5.3 とシェーダー側の前提は3。既定では遠景カスケードが丸ごと存在しない

### 良かった点

- **頂点パッキングは完全一致**。VertexPacker.java、SPEC §3.1、gbuffer.vert、csm_shadow.vert、変換器の頂点ラッパー生成コードの5者で全フィールドのシフト・マスクが一致。packTightAabb の 5bit×6 も chunk_cull.comp と一致、ChunkDrawMeta 32バイトも Java／GLSL std430／SPEC で一致。G-Buffer フォーマットも SPEC §3.3 通り
- Hi-Z ミップチェーンの同期は丁寧（mip ごとにイメージバリア）。ただし set1 binding3 の記述子申告と構築中 mip のレイアウトに検証レイヤーで指摘されうる箇所あり（実害は薄い）
- テストは本番コードを直接実行している点は評価できる（VertexPacker ラウンドトリップ、本物の GreedyMesher の検証、実デバイスでのカリング読み戻し、実 GL での interop 合成）。ただし次節の問題がある
- llvmpipe で外部セマフォ拡張が無いことを検出し「GL semaphore sync disabled」と警告して劣化動作する作りは正直で良い
- ライトマップ座標の変換は標準通りで正しい（頂点ラッパーの `vk_MultiTexCoord1 = (blockLight×16, skyLight×16)` と `VK_LIGHTMAP_MATRIX` = 1/256 スケール + 8/256 オフセットが整合）

---

## シェーダーパック検証の実測（2026-11-14 追加検証）

ユーザー環境で「パックによっては真っ黒」と報告されたため、検証スイートに実パック14種（BSL、Complementary 両版、MakeUp、Mellow、Potato、Sildur's 2種、Simplicissimus、superDuperVanilla、Tea、VanillaPlus、YoFPS、miniature）を食わせ、変換とライブ描画を実測した。

### ライブ描画の結果（64×64、テスト用2クアッド、llvmpipe）

| パック | 結果 |
|---|---|
| BSL_v10.1.8 | **完全な黒** 非黒 0/4096、相違色1、中心 (0,0,0,255) |
| VanillaPlus_v3.3 | **完全な黒** 0/4096、中心 (0,0,0,0) |
| YoFPS (Chocapic edit) | **完全な黒** 0/4096 |
| Complementary Reimagined r5.9.3 | **ほぼ黒** 中心 (2,2,2)、相違色54 |
| Complementary Unbound r5.9.3 | **ほぼ黒** 中心 (1,1,1)、相違色40 |
| MakeUp-UltraFast-9.6 | 描画されるが中心 (2,2,2) と真っ暗寄り |
| Mellow v3.5 | 描画されるが中心 (1,1,1)、相違色7のみ |
| Tea / Simplicissimus / miniature | 正常（相違色 4043〜4081、明るい） |
| PotatoShaders | 明るいが alpha=0・中心深度 1.0 と別種の異常あり |

### なぜスイートが合格してしまったか（最重要）

`ShaderPackVerificationTest.testLiveVulkanShaderPackPipelineExecution` の失敗条件は「**全部のパックが黒いときだけ**」（`nonBlackPacks == 0` で初めて AssertionError）。個別パックが完全な黒でも `ALL VERIFICATION TESTS PASSED` が出る。実機の真っ黒を検出できない構造が、「テストは通るのに実機で黒い」の直接の原因。パックごとに nonBlack 下限・相違色下限で落とすよう是正すること。

加えて `SCRATCH_SHADERPACKS_DIR`（`ShaderPackVerificationTest.java:77-79`）が開発機の絶対パスにハードコードされており、そのパスが存在しない環境では実パック検証が**無言で合成パック1種に縮退**する。実パック検証はパスを引数／環境変数で受け取るか、リポジトリ内の fixture 配置に変えること。

### Complementary について

変換は 17/17 パスとも shaderc エラーなし。ライブ実行でも shadow / terrain / water / deferred×1 / composite×7 / final の12パスが実際に走る。つまりパス欠落ではなく、**チェーンの途中で光量がほぼ 0 に潰れている**。現ハーネスは最終画像しか読み戻さないため、どのパスで潰れるかは追跡できない。切り分けにはパスごとの colortex 読み戻し（terrain 直後、deferred 直後、composite 各段）を行う計装が必要。有力候補は deferred が読む G-buffer データ／uniform の受け渡し（C11 のゼロ化範囲を含む）。

なお以下は容疑から外れた（確認済み）:
- lmcoord のスケーリングは正しい（上記「良かった点」参照）
- `flip.` ディレクティブは ShaderPackProperties が parse するだけで pipeline 側は未使用（死に機能ではある）が、BSL v10.1.8 は flip 宣言自体を持たないため、今回の黒の主因ではない

### 確定した変換器バグ: Sildur's 2種が変換で死に、スイート全体を止める

Sildur's Enhanced Default / Vibrant Shaders は `gbuffers_textured.vsh` の変換で shaderc エラー（`'[]' : scalar integer expression required`）になり、例外でスイート全体が中断していた（ライブ描画フェーズに到達しない）。変換後ソースで発生箇所を特定済み:

- 元コード（TAA ジッター）: `gl_Position.xy += offsets[framemod8] * gl_Position.w * texelSize;` — `framemod8` は OptiFine カスタム uniform（float 型、`frameCounter % 8` の式で定義）
- 変換後（Vibrant で第450行）: `gl_Position.xy += offsets[float(vk_ubo.frameCounter % 8)] * gl_Position.w * texelSize;`
- 原因: DERIVED_UNIFORMS の置換（`framemod8 -> float(vk_ubo.frameCounter % 8)`）が**配列添字の中にまで float キャスト式をテキスト置換**し、GLSL 460 では不正な float 添字を生成する
- 修正方針: 添字位置に現れる式は int 化する。具体的には、derived uniform を実変数として宣言する方式に変えるか、添字書き換え時に既知の float 派生 uniform を `int(...)` で包む。`framemod2 / framemod4 / framemod600 / frameMod` も同系統なので一括で対処すること

---

## パフォーマンス評価（vs バニラ）

まず前提として、**実 GPU・実ゲームでの同一条件ベンチマークは存在しない**。既存の数値は全て llvmpipe（CPU ソフトウェア描画）であり、GPU 性能比較には使えない。以下は設計レベルの評価である。

### 設計上の利点（本物）

- **頂点帯域**: バニラの地形頂点は32バイト／頂点。本実装は8バイト／頂点＋共有インデックスバッファで、帯域は約 1/4〜1/6
- **ドローコール**: バニラはセクションごとに CPU から大量発行。本実装は GPU カリング（視錐＋Hi-Z 遮蔽）→ `vkCmdDrawIndexedIndirectCount` でパスあたり数回まで圧縮
- **軽い G-Buffer**: 2アタッチメント約17MB（1080p）。統合 GPU／APU の帯域に優しい
- 効果が出る条件: 描画距離が長い、遮蔽が多い、メモリ帯域が細い環境

### 割り引くべき点

- **現状は描く仕事がバニラより少ない**。半透明なし（B6）、非キューブ形状なし（B5）。機能を揃えるまでは「速い」と主張できない
- **固定コスト**: カリング用コンピュート、CSM 再描画（C1 の共有問題含む）、GL/VK interop の毎フレーム同期。描画距離の短い軽いシーンではバニラが速い可能性すらある
- **本当の競合は Sodium**。ドローコール削減と頂点圧縮は Sodium が GL のまま既に実装している。本実装の上乗せは GPU 遮蔽カリングと deferred 化の分だけ、が実態に近い
- 結論を出すには実 GPU で vanilla / Sodium / 本実装を同一ワールド・同一視点・同一描画距離で測り、FPS ではなくフレームタイム分布（1% low）で比較する必要がある

## Vulkan の恩恵について

「Vulkan 化の恩恵を受けられているか」という問いへの評価。

### 受けられている恩恵

- GPU-driven カリング＋indirect draw によるドローコール圧縮
- VMA とサブアロケーションによる明示的なメモリ管理（256MB メガアリーナ）
- ただしこれらは正確には「Vulkan の恩恵」より「GPU-driven 化の恩恵」であり、GL 4.6 の compute＋indirect でも近い設計は可能

### 受けられていない恩恵

- **マルチスレッドでのコマンド記録**: Vulkan 最大の売りだが、実装はレンダースレッドでの単一記録のまま
- **非同期オーバーラップ**: FrameUBO 単一バッファ共有（C2）、フェンス処理の甘さ（B3）、ステージング溢れ時の全待機（C8）など、並列に走らせる設計になっていない
- **本体が GL のまま**: 地形だけ Vulkan で描き、毎フレーム共有テクスチャ＋セマフォで GL に合流する構造上、その合流点が直列化ポイントになり、エンティティ／UI／パーティクルの GL ドライバ負荷は一切減らない。interop が使えない環境では C7 の毎フレーム全画面 CPU 読み戻しとなり、バニラより遅くなる

まとめると、現状は「Vulkan で書かれた GPU-driven レンダラー」であり、「Vulkan の並列性を活かしたレンダラー」にはまだなっていない。土台の方向は合っているが、恩恵の回収はこれからである。

---

## SPEC と実装の乖離

| SPEC の主張 | 実態 |
|---|---|
| §3.1 パッキング／§3.2 メタ／§3.3 G-Buffer | 完全準拠。ただし §3.2 flags の意味（半透明・DirtyShadow）は未使用 |
| §2 Post: TAA / FXAA | FXAA のみ。TAA は存在しない |
| §2 Lighting: Godrays | 実体はフォグの太陽散乱項のみ。体積ゴッドレイなし |
| §2 Async Meshing Worker Pool (Virtual Threads) | 固定スレッドプール（2〜4本） |
| §5.1 Mixin 戦略 | SPEC のサンプルコード通りに実装。乖離なし |
| §5.2 CPU 転送は絶対に避ける | ゼロコピーが本線だが STAGING_READBACK は毎フレーム全画面 CPU 転送 |
| §5.2 glBlitNamedFramebuffer で転写 | 実装はフルスクリーンシェーダー＋gl_FragDepth 書き込み。結果は等価だが方式は異なる |
| §5.3 CSM キャッシュ | 間引きは実装（間隔6）。ただし既定カスケード数2、カリング共有の問題あり |
| BSL 系シェーダーパック対応 | 変換器・パイプラインは実在し deferred/composite/final は実行されるが、water 未実行・sky/entities/hand 未対応・未知 uniform ゼロ化により部分的対応 |

---

## 修正の優先順位（提案）

1. **検証スイートの是正**（パックごとの合否判定、SCRATCH パスのハードコード解消、`:test` タスク）— これが直らないと以後の修正を検証できない
2. **変換器の添字 int 化**（Sildur's で確定したバグ。framemod 系一括）
3. **パス別 colortex 読み戻しの計装**を入れ、BSL／Complementary がどのパスで黒に潰れるか特定する
4. **B1**（例外時フォールバックの記録破壊）
5. **B2 / B3 / B4**（同期の3点セット）
6. **B5**（形状メッシュ化）と **B6**（半透明）— 見た目に直結
7. パフォーマンス主張をするなら、実 GPU で Sodium を含めた同一条件ベンチを取ること

---

## 追記2（2026-11-14）: Sodium との真っ向比較における評価の修正

HEAD `4857f4a` 時点の現在地として「実戦での完成度・総合力では Sodium が圧倒的に強い」「設計上限では特定条件で AerogelVK が Sodium の構造を超える武器を持つ」という結論自体は維持する。ただし、その根拠として挙げられていた主張には以下の「盛りすぎ」があったため、ここに修正を記録する。今後の評価・実装判断はこの修正版を正とすること。

### 修正1: Hi-Z 遮蔽カリングの射程と未検証性

- Hi-Z が効くのは「山・建築物・地下などの密集した遮蔽シーン」に限定される。描画距離を伸ばすこと自体のボトルネック（CPU 側のメッシング、アップロード、頂点スループット）を Hi-Z が消すわけではなく、「距離に比例して圧倒的アドバンテージ」という表現は誤り
- 現状、Hi-Z ミップチェーンとカリングパスは存在するが、「実シーンで実際に何%のチャンクが間引けたか」「カリングパスの所要時間に対して GPU タイムの収支がプラスか」の実測データはゼロ。武器として数えるには実測が前提

### 修正2: 17MB G-Buffer の適用範囲

- 約17MB の軽量 G-Buffer が武器になるのは、AerogelVK 自前のライティング経路（シェーダーパックを使わない軽量モード）のときだけ
- 外部シェーダーパック（BSL / Complementary 等）の実行時は、パック側が要求する colortex 群（RGBA16F / RGBA32F 等）の帯域をそのまま食うため、メモリ帯域面での Iris に対する優位性は消える

### 修正3: iGPU（Radeon 780M 等）における帯域の奪い合い

- 8バイト頂点圧縮でプラスを作っても、同じ統合メモリ帯域を「Vulkan 側（地形・シャドウ・ポスト）」「OpenGL 側（エンティティ・UI・パーティクル）」「GL/VK interop の同期・合流」が奪い合う構図になる
- 収支がプラスになるかは、実機プロファイリング（Radeon GPU Profiler 等）なしには断言できない。「帯域が細い環境で強力」という主張は実測待ち

### 修正4: Sodium のドローコール最適化は既に極限域

- Sodium は GL 4.5 / MDI（glMultiDrawElementsIndirect）ベースで CPU ドローコール問題をすでに極限まで潰している。単純な「ドローコール削減」はもはや差別化要因にならない
- 純粋な差別化要因は2点に絞られる: (1) GPU-driven 遮蔽カリング（Hi-Z）、(2) Vulkan の非同期コンピュート・マルチスレッドコマンド記録の回収（現状は未回収）

### 収束したロードマップ

1. B1〜B6 を潰して土俵に上がる（見た目と同期を正しくする）
2. 実機（Radeon 780M 想定）で「Hi-Z カリングの間引き率とパス所要時間」と「interop 合流のオーバーヘッド」を実測する。カリング前後の drawCount は既存の読み戻し基盤で取得でき、パス時間は Vulkan タイムスタンプクエリで取得できる
3. その数字を根拠に、Vulkan 固有の恩恵（マルチスレッド記録、非同期コンピュート）を回収する

根拠のない強気を排し、「今日の実戦は Sodium」「上限の勝負は実測してから」が正確な現在地である。
