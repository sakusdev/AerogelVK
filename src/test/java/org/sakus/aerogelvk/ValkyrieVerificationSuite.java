package org.sakus.aerogelvk;

import org.sakus.aerogelvk.meshing.AtlasUVTable;
import org.sakus.aerogelvk.meshing.GreedyMesher;
import org.sakus.aerogelvk.meshing.VertexPacker;
import org.sakus.aerogelvk.vulkan.VkContext;
import org.sakus.aerogelvk.vulkan.VkInteropManager;
import org.sakus.aerogelvk.vulkan.VkMegaBuffer;
import org.sakus.aerogelvk.vulkan.VkPipelineManager;
import org.sakus.aerogelvk.vulkan.VkShaderCompiler;
import net.minecraft.SharedConstants;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GL11C;
import org.lwjgl.opengl.GL30C;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.*;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static org.lwjgl.vulkan.VK10.*;

public class ValkyrieVerificationSuite {

    static {
        if (System.getProperty("org.lwjgl.system.stackSize") == null) {
            System.setProperty("org.lwjgl.system.stackSize", "512");
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== [1/7] Testing 64-bit VertexPacker & Tight AABB Bit Layout ===");
        testVertexPacker();

        System.out.println("=== [2/7] Testing Runtime GLSL 460 -> Vulkan 1.3 SPIR-V Compilation (10 Shaders) ===");
        testAllShaders();

        System.out.println("=== [3/7] Testing Reversed-Z Matrix Conversion & Standard-Z Equivalence ===");
        testReversedZProjectionMath();

        System.out.println("=== [4/7] Testing GreedyMesher Winding Order, Bounds & UV Alignment ===");
        testGreedyMesherWindingAndBounds();

        System.out.println("=== [5/7] Testing Live Headless Vulkan 1.3 Context, MegaBuffer, Pipelines & Dispatch ===");
        testHeadlessVulkanPipeline();

        System.out.println("=== [6/7] Testing Live OpenGL + Vulkan Interop (FBO Blit, Resize & Fallback) ===");
        testLiveOpenGLVulkanInterop();

        System.out.println("=== [7/7] Testing Native Vulkan 1.3 OptiFine/Iris Shaderpack Pipeline (14 Real Packs) ===");
        ShaderPackVerificationTest.runAll();

        System.out.println("=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY ===");
    }

    private static void testVertexPacker() {
        int x = 15, y = 8, z = 16;
        int normal = VertexPacker.NORM_POS_Y;
        int ao = 3;
        int blockLight = 14;
        int skyLight = 15;
        boolean wind = true;
        int spriteId = 43210;
        int width = 16;
        int height = 12;
        int corner = 2;
        int matMask = VertexPacker.MAT_WATER;

        long packed = VertexPacker.pack(
                x, y, z, normal, ao, blockLight, skyLight, wind,
                spriteId, width, height, corner, matMask
        );

        assertEq("posX", x, VertexPacker.unpackX(packed));
        assertEq("posY", y, VertexPacker.unpackY(packed));
        assertEq("posZ", z, VertexPacker.unpackZ(packed));
        assertEq("normal", normal, VertexPacker.unpackNormal(packed));
        assertEq("ao", ao, VertexPacker.unpackAo(packed));
        assertEq("blockLight", blockLight, VertexPacker.unpackBlockLight(packed));
        assertEq("skyLight", skyLight, VertexPacker.unpackSkyLight(packed));
        assertEq("windWaving", 1, VertexPacker.unpackWindWaving(packed) ? 1 : 0);
        assertEq("spriteId", spriteId, VertexPacker.unpackSpriteId(packed));
        assertEq("width", width, VertexPacker.unpackWidth(packed));
        assertEq("height", height, VertexPacker.unpackHeight(packed));
        assertEq("cornerIndex", corner, VertexPacker.unpackCornerIndex(packed));
        assertEq("materialMask", matMask, VertexPacker.unpackMaterialMask(packed));

        int aabb = VertexPacker.packTightAabb(1, 2, 3, 14, 15, 16);
        assertEq("aabb.minX", 1, aabb & 0x1F);
        assertEq("aabb.minY", 2, (aabb >>> 5) & 0x1F);
        assertEq("aabb.minZ", 3, (aabb >>> 10) & 0x1F);
        assertEq("aabb.maxX", 14, (aabb >>> 15) & 0x1F);
        assertEq("aabb.maxY", 15, (aabb >>> 20) & 0x1F);
        assertEq("aabb.maxZ", 16, (aabb >>> 25) & 0x1F);
        System.out.println("  -> 64-bit VertexPacker bit-field round-trip verified (8 bytes/vertex).");
    }

    private static void testAllShaders() throws Exception {
        String[][] shaders = {
                {"/assets/aerogelvk/shaders/test_gradient.comp", "COMPUTE"},
                {"/assets/aerogelvk/shaders/chunk_cull.comp", "COMPUTE"},
                {"/assets/aerogelvk/shaders/hiz_downsample.comp", "COMPUTE"},
                {"/assets/aerogelvk/shaders/gbuffer.vert", "VERTEX"},
                {"/assets/aerogelvk/shaders/gbuffer.frag", "FRAGMENT"},
                {"/assets/aerogelvk/shaders/csm_shadow.vert", "VERTEX"},
                {"/assets/aerogelvk/shaders/depth_convert.vert", "VERTEX"},
                {"/assets/aerogelvk/shaders/depth_convert.frag", "FRAGMENT"},
                {"/assets/aerogelvk/shaders/deferred_lighting.comp", "COMPUTE"},
                {"/assets/aerogelvk/shaders/post_composite.comp", "COMPUTE"}
        };

        for (String[] entry : shaders) {
            String path = entry[0];
            int stage = switch (entry[1]) {
                case "VERTEX" -> VkShaderCompiler.STAGE_VERTEX;
                case "FRAGMENT" -> VkShaderCompiler.STAGE_FRAGMENT;
                case "COMPUTE" -> VkShaderCompiler.STAGE_COMPUTE;
                default -> throw new IllegalArgumentException();
            };
            try (InputStream in = ValkyrieVerificationSuite.class.getResourceAsStream(path)) {
                if (in == null) {
                    throw new IllegalStateException("Missing shader resource: " + path);
                }
                String source = new String(in.readAllBytes(), StandardCharsets.UTF_8);
                ByteBuffer spirv = VkShaderCompiler.compileGlslToSpirv(path, source, stage);
                int byteSize = spirv.remaining();
                if (byteSize <= 20 || spirv.getInt(0) != 0x07230203) {
                    throw new IllegalStateException("Invalid SPIR-V magic number for " + path);
                }
                MemoryUtil.memFree(spirv);
                System.out.printf("  -> Compiled %-48s [SPIR-V: %5d bytes, Magic: 0x07230203]%n", path, byteSize);
            }
        }
    }

    private static void testReversedZProjectionMath() {
        float near = 0.05f;
        float far = 512.0f;
        Matrix4f glProj = new Matrix4f().setPerspective((float) Math.toRadians(70.0), 16f / 9f, near, far);
        Matrix4f vkProj = ValkyrieRenderer.toVulkanReversedZ(glProj);

        float[] testDepths = {near, 1.0f, 10.0f, 100.0f, far};
        for (float z : testDepths) {
            Vector4f pGl = new Vector4f(0f, 0f, -z, 1f).mul(glProj);
            float glWinZ = 0.5f * (pGl.z / pGl.w) + 0.5f;

            Vector4f pVk = new Vector4f(0f, 0f, -z, 1f).mul(vkProj);
            float vkNdcZ = pVk.z / pVk.w;

            if (Math.abs((1.0f - vkNdcZ) - glWinZ) > 1e-4f) {
                throw new AssertionError("Depth mismatch at z=" + z + ": 1-vkNdcZ=" + (1.0f - vkNdcZ) + " vs glWinZ=" + glWinZ);
            }
        }
        System.out.println("  -> Reversed-Z projection & Standard-Z depth conversion math verified.");
    }

    private static void testGreedyMesherWindingAndBounds() {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();

        int halo = GreedyMesher.HALO_SIZE;
        BlockState[] blocks = new BlockState[halo * halo * halo];
        Arrays.fill(blocks, Blocks.AIR.defaultBlockState());
        byte[] bl = new byte[halo * halo * halo];
        byte[] sl = new byte[halo * halo * halo];
        Arrays.fill(sl, (byte) 15);

        // Place a 2x1x2 stone slab at y=4, x=5..6, z=7..8
        for (int x = 5; x <= 6; x++) {
            for (int z = 7; z <= 8; z++) {
                int idx = ((4 + 1) * halo + (z + 1)) * halo + (x + 1);
                blocks[idx] = Blocks.STONE.defaultBlockState();
            }
        }

        GreedyMesher.SectionSnapshot snap = new GreedyMesher.SectionSnapshot(0, 0, 0, blocks, bl, sl);
        GreedyMesher.MeshedSectionResult res = GreedyMesher.mesh(snap);

        // Greedy merged 2x1x2 box has exactly 6 quads (one per face)!
        assertEq("quadCount", 6, res.quadCount());
        assertEq("vertexCount", 24, res.packedVertices().length);

        int[] minAabb = VertexPacker.unpackAabbMin(res.aabbPacked());
        int[] maxAabb = VertexPacker.unpackAabbMax(res.aabbPacked());
        assertEq("aabb.minX", 5, minAabb[0]);
        assertEq("aabb.minY", 4, minAabb[1]);
        assertEq("aabb.minZ", 7, minAabb[2]);
        assertEq("aabb.maxX", 7, maxAabb[0]);
        assertEq("aabb.maxY", 5, maxAabb[1]);
        assertEq("aabb.maxZ", 9, maxAabb[2]);

        Vector3f[] expectedNormals = {
                new Vector3f(1, 0, 0), new Vector3f(-1, 0, 0),
                new Vector3f(0, 1, 0), new Vector3f(0, -1, 0),
                new Vector3f(0, 0, 1), new Vector3f(0, 0, -1)
        };

        long[] verts = res.packedVertices();
        for (int q = 0; q < res.quadCount(); q++) {
            int base = q * 4;
            int norm = VertexPacker.unpackNormal(verts[base]);
            Vector3f[] p = new Vector3f[4];
            for (int i = 0; i < 4; i++) {
                long v = verts[base + i];
                p[i] = new Vector3f(VertexPacker.unpackX(v), VertexPacker.unpackY(v), VertexPacker.unpackZ(v));
            }
            // Check Triangle 1 (0, 1, 2) and Triangle 2 (0, 2, 3) CCW cross products match outward normal
            Vector3f n1 = new Vector3f(p[1]).sub(p[0]).cross(new Vector3f(p[2]).sub(p[0])).normalize();
            Vector3f n2 = new Vector3f(p[2]).sub(p[0]).cross(new Vector3f(p[3]).sub(p[0])).normalize();
            if (n1.dot(expectedNormals[norm]) < 0.99f || n2.dot(expectedNormals[norm]) < 0.99f) {
                throw new AssertionError("Winding mismatch for face dir=" + norm + ": n1=" + n1 + ", n2=" + n2);
            }
            // Check corner indices are [3, 2, 1, 0] for upright UV mapping
            assertEq("corner[0]", 3, VertexPacker.unpackCornerIndex(verts[base + 0]));
            assertEq("corner[1]", 2, VertexPacker.unpackCornerIndex(verts[base + 1]));
            assertEq("corner[2]", 1, VertexPacker.unpackCornerIndex(verts[base + 2]));
            assertEq("corner[3]", 0, VertexPacker.unpackCornerIndex(verts[base + 3]));
        }

        // Invisible non-fluid blocks (e.g. BARRIER) must emit zero quads
        Arrays.fill(blocks, Blocks.AIR.defaultBlockState());
        blocks[((2 + 1) * halo + (2 + 1)) * halo + (2 + 1)] = Blocks.BARRIER.defaultBlockState();
        GreedyMesher.MeshedSectionResult barrierRes = GreedyMesher.mesh(new GreedyMesher.SectionSnapshot(0, 0, 0, blocks, bl, sl));
        assertEq("barrierQuadCount", 0, barrierRes.quadCount());

        // Adjacent same-fluid blocks cull shared internal face & greedy merge to 6 translucent quads (B6 split)
        Arrays.fill(blocks, Blocks.AIR.defaultBlockState());
        blocks[((2 + 1) * halo + (2 + 1)) * halo + (2 + 1)] = Blocks.WATER.defaultBlockState();
        blocks[((2 + 1) * halo + (2 + 1)) * halo + (3 + 1)] = Blocks.WATER.defaultBlockState();
        GreedyMesher.MeshedSectionResult waterPoolRes = GreedyMesher.mesh(new GreedyMesher.SectionSnapshot(0, 0, 0, blocks, bl, sl));
        assertEq("waterPoolOpaqueQuadCount", 0, waterPoolRes.quadCount());
        assertEq("waterPoolTranslucentQuadCount", 6, waterPoolRes.translucentQuadCount());

        // Stone adjacent to translucent water: stone keeps 6 opaque faces, water culls stone-facing face (5 translucent faces)
        Arrays.fill(blocks, Blocks.AIR.defaultBlockState());
        blocks[((2 + 1) * halo + (2 + 1)) * halo + (2 + 1)] = Blocks.STONE.defaultBlockState();
        blocks[((2 + 1) * halo + (2 + 1)) * halo + (3 + 1)] = Blocks.WATER.defaultBlockState();
        GreedyMesher.MeshedSectionResult stoneWaterRes = GreedyMesher.mesh(new GreedyMesher.SectionSnapshot(0, 0, 0, blocks, bl, sl));
        assertEq("stoneWaterOpaqueQuadCount", 6, stoneWaterRes.quadCount());
        assertEq("stoneWaterTranslucentQuadCount", 5, stoneWaterRes.translucentQuadCount());

        // Non-full VoxelShape block (e.g. OAK_SLAB) emits half-scale opaque quads (B5)
        Arrays.fill(blocks, Blocks.AIR.defaultBlockState());
        blocks[((2 + 1) * halo + (2 + 1)) * halo + (2 + 1)] = Blocks.OAK_SLAB.defaultBlockState();
        GreedyMesher.MeshedSectionResult slabRes = GreedyMesher.mesh(new GreedyMesher.SectionSnapshot(0, 0, 0, blocks, bl, sl));
        assertEq("slabHalfScaleOpaqueQuadCount", 6, slabRes.halfScaleOpaqueQuadCount());

        System.out.println("  -> GreedyMesher 6-face CCW winding, tight AABB, upright UVs, invisible skip & fluid culling verified.");
    }

    private static void testHeadlessVulkanPipeline() {
        VkContext ctx = VkContext.getInstance();
        ctx.init();

        VkMegaBuffer mega = VkMegaBuffer.getInstance();
        mega.init();

        // Test section upload, sparse slot dispatch limit, and unload
        long[] dummyQuad = new long[]{
                VertexPacker.pack(0, 0, 0, 2, 3, 0, 15, false, 1, 1, 1, 3, 0),
                VertexPacker.pack(1, 0, 0, 2, 3, 0, 15, false, 1, 1, 1, 2, 0),
                VertexPacker.pack(1, 0, 1, 2, 3, 0, 15, false, 1, 1, 1, 1, 0),
                VertexPacker.pack(0, 0, 1, 2, 3, 0, 15, false, 1, 1, 1, 0, 0)
        };
        int slot0 = mega.uploadSectionMesh(0, 0, 0, dummyQuad, 1, VertexPacker.packTightAabb(0, 0, 0, 1, 1, 1), 0);
        int slot1 = mega.uploadSectionMesh(1, 0, 0, dummyQuad, 1, VertexPacker.packTightAabb(0, 0, 0, 1, 1, 1), 0);
        assertEq("slot0", 0, slot0);
        assertEq("slot1", 1, slot1);
        assertEq("dispatchLimitBeforeUnload", 2, mega.getSlotDispatchLimit());
        mega.unloadSection(0, 0, 0);
        assertEq("dispatchLimitWithSparseSlot", 2, mega.getSlotDispatchLimit());
        mega.unloadSection(1, 0, 0);
        assertEq("dispatchLimitAfterAllUnloaded", 0, mega.getSlotDispatchLimit());

        // Create temporary offscreen color & depth images for VkPipelineManager initialization
        VkDevice device = ctx.getVkDevice();
        long offscreenColorImg = 0, offscreenColorMem = 0, offscreenColorView = 0;
        long offscreenDepthImg = 0, offscreenDepthMem = 0, offscreenDepthView = 0;
        long readbackBuf = 0, readbackMem = 0;

        try (MemoryStack stack = MemoryStack.stackPush()) {
            long[] colorRes = createOffscreenImage(ctx, device, 64, 64, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, stack);
            offscreenColorImg = colorRes[0];
            offscreenColorMem = colorRes[1];
            offscreenColorView = colorRes[2];

            long[] depthRes = createOffscreenImage(ctx, device, 64, 64, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, stack);
            offscreenDepthImg = depthRes[0];
            offscreenDepthMem = depthRes[1];
            offscreenDepthView = depthRes[2];

            // Host-visible readback buffer: [0..4): drawCount, [16..16399]: 64x64 RGBA8, [16400..32783]: 64x64 D32_SFLOAT
            int colorOffset = 16;
            int colorBytes = 64 * 64 * 4;
            int depthOffset = colorOffset + colorBytes;
            int totalReadbackBytes = depthOffset + colorBytes;

            VkBufferCreateInfo bufInfo = VkBufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(totalReadbackBytes)
                    .usage(VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer pBuf = stack.mallocLong(1);
            if (vkCreateBuffer(device, bufInfo, null, pBuf) != VK_SUCCESS) throw new RuntimeException("vkCreateBuffer failed");
            readbackBuf = pBuf.get(0);

            VkMemoryRequirements memReqs = VkMemoryRequirements.malloc(stack);
            vkGetBufferMemoryRequirements(device, readbackBuf, memReqs);
            VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                    .allocationSize(memReqs.size())
                    .memoryTypeIndex(ctx.findMemoryType(memReqs.memoryTypeBits(),
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
            LongBuffer pMem = stack.mallocLong(1);
            if (vkAllocateMemory(device, allocInfo, null, pMem) != VK_SUCCESS) throw new RuntimeException("vkAllocateMemory failed");
            readbackMem = pMem.get(0);
            vkBindBufferMemory(device, readbackBuf, readbackMem, 0);

            VkPipelineManager pipe = VkPipelineManager.getInstance();
            pipe.init(64, 64, offscreenColorView, offscreenDepthView);

            pipe.updateDescriptorSet0(
                    mega.getMegaVertexBuffer().buffer,
                    mega.getChunkMetaBuffer().buffer,
                    mega.getIndirectCmdBuffer().buffer,
                    mega.getDrawCountBuffer().buffer,
                    mega.getVisibleChunkIdBuffer().buffer,
                    mega.getAtlasUvBuffer().buffer,
                    pipe.getBlockAtlasView()
            );

            // Dispatch test_gradient.comp on GPU and wait for completion
            VkCommandBuffer gradCmd = ctx.beginSingleTimeCommands();
            VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                    .oldLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                    .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(offscreenColorImg)
                    .srcAccessMask(0)
                    .dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT);
            barrier.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
            vkCmdPipelineBarrier(gradCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, null, null, barrier);

            vkCmdBindPipeline(gradCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.getTestGradientPipeline());
            LongBuffer pSets = stack.longs(pipe.getDescriptorSet0(), pipe.getDescriptorSet1());
            vkCmdBindDescriptorSets(gradCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.getPipelineLayout(), 0, pSets, null);
            ByteBuffer pc = stack.malloc(4);
            pc.putFloat(0, 1.0f);
            vkCmdPushConstants(gradCmd, pipe.getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
            vkCmdDispatch(gradCmd, 4, 4, 1);
            ctx.endSingleTimeCommands(gradCmd);

            // Upload default fallback UV table (entry 0: uv=[0..1], tint=white)
            mega.uploadAtlasUvTable(AtlasUVTable.getInstance().getUvTableBuffer());

            // Upload a 16x16 +Z-facing terrain quad at z=1 facing camera at (8, 8, 8)
            long[] wallQuad = new long[]{
                    VertexPacker.pack(0,  0,  1, 4, 3, 15, 15, false, 0, 1, 1, 3, 0),
                    VertexPacker.pack(16, 0,  1, 4, 3, 15, 15, false, 0, 1, 1, 2, 0),
                    VertexPacker.pack(16, 16, 1, 4, 3, 15, 15, false, 0, 1, 1, 1, 0),
                    VertexPacker.pack(0,  16, 1, 4, 3, 15, 15, false, 0, 1, 1, 0, 0)
            };
            mega.uploadSectionMesh(0, 0, 0, wallQuad, 1, VertexPacker.packTightAabb(0, 0, 1, 16, 16, 2), 0);

            Vector3f camPos = new Vector3f(8f, 8f, 8f);
            Vector3f target = new Vector3f(8f, 8f, 1f);
            Matrix4f glProj = new Matrix4f().perspective((float) Math.toRadians(70.0), 1.0f, 0.05f, 256.0f);

            ValkyrieRenderer renderer = ValkyrieRenderer.getInstance();
            // Run Frame 1 (builds Hi-Z pyramid) & Frame 2 (culls against Frame 1 Hi-Z pyramid)
            for (int frame = 0; frame < 2; frame++) {
                VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
                renderer.recordHeadlessGpuDrivenFrame(
                        cmd, 64, 64, camPos, target, glProj,
                        offscreenColorImg, offscreenDepthImg, offscreenDepthView, frame == 0
                );
                if (frame == 1) {
                    VkBufferCopy.Buffer countCopy = VkBufferCopy.calloc(1, stack).srcOffset(0).dstOffset(0).size(4);
                    vkCmdCopyBuffer(cmd, mega.getDrawCountBuffer().buffer, readbackBuf, countCopy);

                    VkBufferImageCopy.Buffer colorCopy = VkBufferImageCopy.calloc(1, stack)
                            .bufferOffset(colorOffset).bufferRowLength(0).bufferImageHeight(0);
                    colorCopy.imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
                    colorCopy.imageExtent().set(64, 64, 1);
                    vkCmdCopyImageToBuffer(cmd, offscreenColorImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuf, colorCopy);

                    VkBufferImageCopy.Buffer depthCopy = VkBufferImageCopy.calloc(1, stack)
                            .bufferOffset(depthOffset).bufferRowLength(0).bufferImageHeight(0);
                    depthCopy.imageSubresource().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).layerCount(1);
                    depthCopy.imageExtent().set(64, 64, 1);
                    vkCmdCopyImageToBuffer(cmd, offscreenDepthImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuf, depthCopy);
                }
                ctx.endSingleTimeCommands(cmd);
            }

            PointerBuffer ppMapped = stack.mallocPointer(1);
            vkMapMemory(device, readbackMem, 0, totalReadbackBytes, 0, ppMapped);
            ByteBuffer mapped = MemoryUtil.memByteBuffer(ppMapped.get(0), totalReadbackBytes);
            int drawCount = mapped.getInt(0);
            int centerIdx = (32 * 64 + 32) * 4;
            int r = mapped.get(colorOffset + centerIdx) & 0xFF;
            int g = mapped.get(colorOffset + centerIdx + 1) & 0xFF;
            int b = mapped.get(colorOffset + centerIdx + 2) & 0xFF;
            int a = mapped.get(colorOffset + centerIdx + 3) & 0xFF;
            float centerDepth = mapped.getFloat(depthOffset + centerIdx);
            vkUnmapMemory(device, readbackMem);

            assertEq("gpuDrawCountAfterHiZ", 1, drawCount);
            if (r == 0 && g == 0 && b == 0) {
                throw new AssertionError("Center pixel RGBA is black (0,0,0," + a + "), expected lit terrain quad!");
            }
            if (!(centerDepth > 0.5f && centerDepth < 0.999f)) {
                throw new AssertionError("Converted Standard-Z depth out of range at center pixel: " + centerDepth);
            }
            System.out.printf("  -> Headless GPU-Driven Pipeline verified (drawCount=%d, centerRGBA=(%d,%d,%d,%d), centerStdZ=%.4f).%n",
                    drawCount, r, g, b, a, centerDepth);

            benchmarkBuiltInVulkan1280x720(ctx, device, pipe, renderer, camPos, target, stack);
        } finally {
            vkDeviceWaitIdle(device);
            if (readbackBuf != 0) vkDestroyBuffer(device, readbackBuf, null);
            if (readbackMem != 0) vkFreeMemory(device, readbackMem, null);
            if (offscreenColorView != 0) vkDestroyImageView(device, offscreenColorView, null);
            if (offscreenColorImg != 0) vkDestroyImage(device, offscreenColorImg, null);
            if (offscreenColorMem != 0) vkFreeMemory(device, offscreenColorMem, null);
            if (offscreenDepthView != 0) vkDestroyImageView(device, offscreenDepthView, null);
            if (offscreenDepthImg != 0) vkDestroyImage(device, offscreenDepthImg, null);
            if (offscreenDepthMem != 0) vkFreeMemory(device, offscreenDepthMem, null);
        }
    }

    private static void benchmarkBuiltInVulkan1280x720(
            VkContext ctx,
            VkDevice device,
            VkPipelineManager pipe,
            ValkyrieRenderer renderer,
            Vector3f camPos,
            Vector3f target,
            MemoryStack stack
    ) {
        final int bw = 1280;
        final int bh = 720;
        final int warmupFrames = 4;
        final int benchFrames = 30;
        Matrix4f benchProj = new Matrix4f().perspective((float) Math.toRadians(70.0), (float) bw / (float) bh, 0.05f, 256.0f);

        long benchColorImg = 0, benchColorMem = 0, benchColorView = 0;
        long benchDepthImg = 0, benchDepthMem = 0, benchDepthView = 0;
        try {
            long[] colorRes = createOffscreenImage(ctx, device, bw, bh, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, stack);
            benchColorImg = colorRes[0];
            benchColorMem = colorRes[1];
            benchColorView = colorRes[2];

            long[] depthRes = createOffscreenImage(ctx, device, bw, bh, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, stack);
            benchDepthImg = depthRes[0];
            benchDepthMem = depthRes[1];
            benchDepthView = depthRes[2];

            pipe.recreateSizeDependentResources(bw, bh, benchColorView, benchDepthView);
            long startNs = 0L;
            for (int f = 0; f < warmupFrames + benchFrames; f++) {
                if (f == warmupFrames) {
                    vkDeviceWaitIdle(device);
                    startNs = System.nanoTime();
                }
                VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
                renderer.recordHeadlessGpuDrivenFrame(
                        cmd, bw, bh, camPos, target, benchProj,
                        benchColorImg, benchDepthImg, benchDepthView, f == 0
                );
                ctx.endSingleTimeCommands(cmd);
            }
            vkDeviceWaitIdle(device);
            double elapsedSec = (System.nanoTime() - startNs) / 1_000_000_000.0;
            double avgFps = benchFrames / Math.max(elapsedSec, 1e-6);
            double avgMs = (elapsedSec * 1000.0) / benchFrames;
            System.out.printf("  -> [1280x720 Vulkan GPU Benchmark] %-42s [frames=%d, avgFps=%.2f, avgFrameMs=%.2f]%n",
                    "Built-in Vulkan (VkPipelineManager)", benchFrames, avgFps, avgMs);
        } finally {
            vkDeviceWaitIdle(device);
            if (benchColorView != 0) vkDestroyImageView(device, benchColorView, null);
            if (benchColorImg != 0) vkDestroyImage(device, benchColorImg, null);
            if (benchColorMem != 0) vkFreeMemory(device, benchColorMem, null);
            if (benchDepthView != 0) vkDestroyImageView(device, benchDepthView, null);
            if (benchDepthImg != 0) vkDestroyImage(device, benchDepthImg, null);
            if (benchDepthMem != 0) vkFreeMemory(device, benchDepthMem, null);
        }
    }

    private static void testLiveOpenGLVulkanInterop() {
        String display = System.getenv("DISPLAY");
        String wayland = System.getenv("WAYLAND_DISPLAY");
        if ((display == null || display.isBlank()) && (wayland == null || wayland.isBlank())) {
            System.out.println("  -> [SKIP-NO-DISPLAY] No X11/Wayland display available; skipping live GLFW+OpenGL interop check.");
            VkPipelineManager.getInstance().close();
            VkMegaBuffer.getInstance().close();
            VkContext.getInstance().close();
            return;
        }

        if (!GLFW.glfwInit()) {
            throw new AssertionError("glfwInit() failed despite display environment (" + display + ")");
        }

        long win = 0L;
        int targetFbo = 0, targetColorTex = 0, targetDepthTex = 0;
        VkContext ctx = VkContext.getInstance();
        VkMegaBuffer mega = VkMegaBuffer.getInstance();
        VkInteropManager interop = VkInteropManager.getInstance();
        VkPipelineManager pipe = VkPipelineManager.getInstance();

        try {
            GLFW.glfwDefaultWindowHints();
            GLFW.glfwWindowHint(GLFW.GLFW_VISIBLE, GLFW.GLFW_FALSE);
            GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MAJOR, 4);
            GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MINOR, 5);
            GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_PROFILE, GLFW.GLFW_OPENGL_CORE_PROFILE);

            win = GLFW.glfwCreateWindow(64, 64, "ValkyrieInteropTest", 0L, 0L);
            if (win == 0L) {
                throw new AssertionError("Failed to create offscreen GLFW OpenGL 4.5 context under display=" + display);
            }
            GLFW.glfwMakeContextCurrent(win);
            GL.createCapabilities();

            ctx.init();
            mega.init();

            // Verify initial creation and resize path (64x64 -> 32x32 -> 64x64)
            interop.recreate(64, 64);
            interop.recreate(32, 32);
            interop.recreate(64, 64);
            if (!interop.isInitialized() || interop.getSharedColorImage() == 0 || interop.getSharedDepthImage() == 0) {
                throw new AssertionError("VkInteropManager failed to initialize shared color/depth targets");
            }

            // Create target OpenGL FBO matching 64x64 RGBA8 + DEPTH32F
            targetColorTex = GL11C.glGenTextures();
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, targetColorTex);
            GL11C.glTexImage2D(GL11C.GL_TEXTURE_2D, 0, GL11C.GL_RGBA8, 64, 64, 0, GL11C.GL_RGBA, GL11C.GL_UNSIGNED_BYTE, 0L);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MIN_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MAG_FILTER, GL11C.GL_NEAREST);

            targetDepthTex = GL11C.glGenTextures();
            GL11C.glBindTexture(GL11C.GL_TEXTURE_2D, targetDepthTex);
            GL11C.glTexImage2D(GL11C.GL_TEXTURE_2D, 0, GL30C.GL_DEPTH_COMPONENT32F, 64, 64, 0, GL11C.GL_DEPTH_COMPONENT, GL11C.GL_FLOAT, 0L);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MIN_FILTER, GL11C.GL_NEAREST);
            GL11C.glTexParameteri(GL11C.GL_TEXTURE_2D, GL11C.GL_TEXTURE_MAG_FILTER, GL11C.GL_NEAREST);

            targetFbo = GL30C.glGenFramebuffers();
            GL30C.glBindFramebuffer(GL30C.GL_FRAMEBUFFER, targetFbo);
            GL30C.glFramebufferTexture2D(GL30C.GL_FRAMEBUFFER, GL30C.GL_COLOR_ATTACHMENT0, GL11C.GL_TEXTURE_2D, targetColorTex, 0);
            GL30C.glFramebufferTexture2D(GL30C.GL_FRAMEBUFFER, GL30C.GL_DEPTH_ATTACHMENT, GL11C.GL_TEXTURE_2D, targetDepthTex, 0);
            int fboStatus = GL30C.glCheckFramebufferStatus(GL30C.GL_FRAMEBUFFER);
            if (fboStatus != GL30C.GL_FRAMEBUFFER_COMPLETE) {
                throw new AssertionError("Target GL FBO incomplete: 0x" + Integer.toHexString(fboStatus));
            }
            GL11C.glClearColor(0f, 0f, 0f, 0f);
            GL11C.glClearDepth(1.0);
            GL11C.glClear(GL11C.GL_COLOR_BUFFER_BIT | GL11C.GL_DEPTH_BUFFER_BIT);
            GL30C.glBindFramebuffer(GL30C.GL_FRAMEBUFFER, 0);

            if (!pipe.isInitialized()) {
                pipe.init(64, 64, interop.getSharedColorImageView(), interop.getSharedDepthImageView());
            } else {
                pipe.recreateSizeDependentResources(64, 64, interop.getSharedColorImageView(), interop.getSharedDepthImageView());
            }
            pipe.updateDescriptorSet0(
                    mega.getMegaVertexBuffer().buffer,
                    mega.getChunkMetaBuffer().buffer,
                    mega.getIndirectCmdBuffer().buffer,
                    mega.getDrawCountBuffer().buffer,
                    mega.getVisibleChunkIdBuffer().buffer,
                    mega.getAtlasUvBuffer().buffer,
                    pipe.getBlockAtlasView()
            );
            mega.uploadAtlasUvTable(AtlasUVTable.getInstance().getUvTableBuffer());

            long[] wallQuad = new long[]{
                    VertexPacker.pack(0,  0,  1, 4, 3, 15, 15, false, 0, 1, 1, 3, 0),
                    VertexPacker.pack(16, 0,  1, 4, 3, 15, 15, false, 0, 1, 1, 2, 0),
                    VertexPacker.pack(16, 16, 1, 4, 3, 15, 15, false, 0, 1, 1, 1, 0),
                    VertexPacker.pack(0,  16, 1, 4, 3, 15, 15, false, 0, 1, 1, 0, 0)
            };
            mega.uploadSectionMesh(0, 0, 0, wallQuad, 1, VertexPacker.packTightAabb(0, 0, 1, 16, 16, 2), 0);

            Vector3f camPos = new Vector3f(8f, 8f, 8f);
            Vector3f target = new Vector3f(8f, 8f, 1f);
            Matrix4f glProj = new Matrix4f().perspective((float) Math.toRadians(70.0), 1.0f, 0.05f, 256.0f);
            ValkyrieRenderer renderer = ValkyrieRenderer.getInstance();

            for (int frame = 0; frame < 2; frame++) {
                VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
                renderer.recordHeadlessGpuDrivenFrame(
                        cmd, 64, 64, camPos, target, glProj,
                        interop.getSharedColorImage(), interop.getSharedDepthImage(), interop.getSharedDepthImageView(),
                        frame == 0
                );
                if (frame == 1 && interop.usesExternalSemaphores()) {
                    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                        throw new RuntimeException("vkEndCommandBuffer failed");
                    }
                    try (MemoryStack stack = MemoryStack.stackPush()) {
                        PointerBuffer pCmd = stack.pointers(cmd);
                        LongBuffer pSignal = stack.longs(interop.getVkRenderCompleteSemaphore());
                        VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                                .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                                .pCommandBuffers(pCmd)
                                .pSignalSemaphores(pSignal);
                        if (vkQueueSubmit(ctx.getGraphicsQueue(), submit, VK_NULL_HANDLE) != VK_SUCCESS) {
                            throw new RuntimeException("vkQueueSubmit failed");
                        }
                        vkQueueWaitIdle(ctx.getGraphicsQueue());
                        vkFreeCommandBuffers(ctx.getVkDevice(), ctx.getCommandPool(), cmd);
                    }
                } else {
                    ctx.endSingleTimeCommands(cmd);
                }
            }

            interop.compositeToMainTarget(targetFbo, 64, 64);
            GL11C.glFinish();

            int glErr = GL11C.glGetError();
            if (glErr != GL11C.GL_NO_ERROR) {
                throw new AssertionError("OpenGL error during compositeToMainTarget: 0x" + Integer.toHexString(glErr));
            }

            GL30C.glBindFramebuffer(GL30C.GL_READ_FRAMEBUFFER, targetFbo);
            ByteBuffer rgba = MemoryUtil.memAlloc(4);
            FloatBuffer depthBuf = MemoryUtil.memAllocFloat(1);
            int r, g, b, a;
            float glDepth;
            try {
                GL11C.glReadPixels(32, 32, 1, 1, GL11C.GL_RGBA, GL11C.GL_UNSIGNED_BYTE, rgba);
                GL11C.glReadPixels(32, 32, 1, 1, GL11C.GL_DEPTH_COMPONENT, GL11C.GL_FLOAT, depthBuf);
                r = rgba.get(0) & 0xFF;
                g = rgba.get(1) & 0xFF;
                b = rgba.get(2) & 0xFF;
                a = rgba.get(3) & 0xFF;
                glDepth = depthBuf.get(0);
            } finally {
                MemoryUtil.memFree(rgba);
                MemoryUtil.memFree(depthBuf);
            }

            if (r == 0 && g == 0 && b == 0) {
                throw new AssertionError("GL target FBO center pixel is black (0,0,0," + a + "), interop blit failed! mode=" + interop.getInteropMode());
            }
            if (!(glDepth > 0.5f && glDepth < 0.999f)) {
                throw new AssertionError("GL target FBO center depth out of range: " + glDepth + ", mode=" + interop.getInteropMode());
            }

            System.out.printf("  -> Live OpenGL+Vulkan Interop verified (mode=%s, glRGBA=(%d,%d,%d,%d), glDepth=%.4f).%n",
                    interop.getInteropMode(), r, g, b, a, glDepth);
        } finally {
            pipe.close();
            interop.close();
            if (targetFbo != 0) GL30C.glDeleteFramebuffers(targetFbo);
            if (targetColorTex != 0) GL11C.glDeleteTextures(targetColorTex);
            if (targetDepthTex != 0) GL11C.glDeleteTextures(targetDepthTex);
            mega.close();
            ctx.close();
            if (win != 0L) GLFW.glfwDestroyWindow(win);
            GLFW.glfwTerminate();
        }
    }

    private static long[] createOffscreenImage(VkContext ctx, VkDevice device, int w, int h, int format, int usage, int aspect, MemoryStack stack) {
        VkImageCreateInfo imgInfo = VkImageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                .imageType(VK_IMAGE_TYPE_2D)
                .format(format)
                .extent(e -> e.width(w).height(h).depth(1))
                .mipLevels(1)
                .arrayLayers(1)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                .usage(usage)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        LongBuffer pImg = stack.mallocLong(1);
        if (vkCreateImage(device, imgInfo, null, pImg) != VK_SUCCESS) throw new RuntimeException("vkCreateImage failed");

        VkMemoryRequirements reqs = VkMemoryRequirements.malloc(stack);
        vkGetImageMemoryRequirements(device, pImg.get(0), reqs);
        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(reqs.size())
                .memoryTypeIndex(ctx.findMemoryType(reqs.memoryTypeBits(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        LongBuffer pMem = stack.mallocLong(1);
        if (vkAllocateMemory(device, allocInfo, null, pMem) != VK_SUCCESS) throw new RuntimeException("vkAllocateMemory failed");
        vkBindImageMemory(device, pImg.get(0), pMem.get(0), 0);

        VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                .image(pImg.get(0))
                .viewType(VK_IMAGE_VIEW_TYPE_2D)
                .format(format)
                .subresourceRange(r -> r.aspectMask(aspect).levelCount(1).layerCount(1));
        LongBuffer pView = stack.mallocLong(1);
        if (vkCreateImageView(device, viewInfo, null, pView) != VK_SUCCESS) throw new RuntimeException("vkCreateImageView failed");

        return new long[]{pImg.get(0), pMem.get(0), pView.get(0)};
    }

    private static void assertEq(String field, int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError("Field " + field + " mismatch: expected " + expected + ", got " + actual);
        }
    }
}
