package com.example.valkyrie;

import com.example.valkyrie.meshing.VertexPacker;
import com.example.valkyrie.vulkan.VkShaderCompiler;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.lwjgl.system.MemoryUtil;

public class ValkyrieVerificationSuite {

    public static void main(String[] args) throws Exception {
        System.out.println("=== [1/2] Testing 64-bit VertexPacker & Tight AABB Bit Layout ===");
        testVertexPacker();

        System.out.println("=== [2/2] Testing Runtime GLSL 460 -> Vulkan 1.3 SPIR-V Compilation ===");
        testAllShaders();

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
                {"/assets/valkyrie/shaders/test_gradient.comp", "COMPUTE"},
                {"/assets/valkyrie/shaders/chunk_cull.comp", "COMPUTE"},
                {"/assets/valkyrie/shaders/gbuffer.vert", "VERTEX"},
                {"/assets/valkyrie/shaders/gbuffer.frag", "FRAGMENT"},
                {"/assets/valkyrie/shaders/csm_shadow.vert", "VERTEX"},
                {"/assets/valkyrie/shaders/deferred_lighting.comp", "COMPUTE"},
                {"/assets/valkyrie/shaders/post_composite.comp", "COMPUTE"}
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

    private static void assertEq(String field, int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError("Field " + field + " mismatch: expected " + expected + ", got " + actual);
        }
    }
}
