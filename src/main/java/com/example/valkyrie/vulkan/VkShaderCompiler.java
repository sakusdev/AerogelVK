package com.example.valkyrie.vulkan;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.shaderc.Shaderc;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;

import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkCreateShaderModule;

/**
 * Runtime GLSL 460 -> SPIR-V compiler and VkShaderModule factory.
 *
 * <p>Uses LWJGL 3's Shaderc bindings to compile GLSL source strings (or classpath
 * resources) into SPIR-V bytecode targeting the Vulkan 1.3 environment, then
 * wraps the resulting bytecode in a {@code VkShaderModule}.</p>
 *
 * <p>All native resources (shaderc compiler, options, result, and the SPIR-V
 * ByteBuffer) are released deterministically. The returned {@link ByteBuffer}
 * from {@link #compileGlslToSpirv(String, String, int)} is allocated with
 * {@link MemoryUtil#memAlloc(long)} and must be freed by the caller with
 * {@link MemoryUtil#memFree(ByteBuffer)}.</p>
 */
public final class VkShaderCompiler {

    /** Shaderc stage constant for GLSL vertex shaders. */
    public static final int STAGE_VERTEX = Shaderc.shaderc_glsl_vertex_shader;

    /** Shaderc stage constant for GLSL fragment shaders. */
    public static final int STAGE_FRAGMENT = Shaderc.shaderc_glsl_fragment_shader;

    /** Shaderc stage constant for GLSL compute shaders. */
    public static final int STAGE_COMPUTE = Shaderc.shaderc_glsl_compute_shader;

    private static final String DEFAULT_ENTRY_POINT = "main";

    private VkShaderCompiler() {
        // Utility class.
    }

    /**
     * Compiles a GLSL source string into SPIR-V bytecode.
     *
     * @param shaderName   logical name used for diagnostics (e.g. resource path)
     * @param glslSource   GLSL 460 source text
     * @param shadercStage one of {@link #STAGE_VERTEX}, {@link #STAGE_FRAGMENT}, {@link #STAGE_COMPUTE}
     * @return a native {@link ByteBuffer} (allocated via {@link MemoryUtil#memAlloc(long)})
     *         containing the SPIR-V bytecode, positioned at 0 and limited to its length.
     *         The caller owns this buffer and must free it with {@link MemoryUtil#memFree(ByteBuffer)}.
     * @throws RuntimeException if compilation fails or shaderc returns an error
     */
    public static ByteBuffer compileGlslToSpirv(String shaderName, String glslSource, int shadercStage) {
        if (shaderName == null) {
            shaderName = "<unnamed>";
        }
        if (glslSource == null) {
            throw new IllegalArgumentException("glslSource must not be null for shader: " + shaderName);
        }

        long compiler = MemoryUtil.NULL;
        long options = MemoryUtil.NULL;
        long result = MemoryUtil.NULL;
        ByteBuffer srcUtf8 = null;
        ByteBuffer nameUtf8 = null;
        ByteBuffer entryUtf8 = null;

        try {
            compiler = Shaderc.shaderc_compiler_initialize();
            if (compiler == MemoryUtil.NULL) {
                throw new RuntimeException("Failed to initialize shaderc compiler for shader: " + shaderName);
            }

            options = Shaderc.shaderc_compile_options_initialize();
            if (options == MemoryUtil.NULL) {
                throw new RuntimeException("Failed to initialize shaderc compile options for shader: " + shaderName);
            }

            Shaderc.shaderc_compile_options_set_target_env(
                    options,
                    Shaderc.shaderc_target_env_vulkan,
                    Shaderc.shaderc_env_version_vulkan_1_3);

            Shaderc.shaderc_compile_options_set_optimization_level(
                    options,
                    Shaderc.shaderc_optimization_level_performance);

            // Emit debug info so validation layers / RenderDoc can map SPIR-V back to GLSL.
            Shaderc.shaderc_compile_options_set_generate_debug_info(options);

            // Allocate UTF-8 buffers on the heap via MemoryUtil instead of MemoryStack so large (>64 KB)
            // shaderpack sources never overflow LWJGL's 64 KB thread stack.
            srcUtf8 = MemoryUtil.memUTF8(glslSource, false);
            nameUtf8 = MemoryUtil.memUTF8(shaderName, true);
            entryUtf8 = MemoryUtil.memUTF8(DEFAULT_ENTRY_POINT, true);

            result = Shaderc.nshaderc_compile_into_spv(
                    compiler,
                    MemoryUtil.memAddress(srcUtf8),
                    srcUtf8.remaining(),
                    shadercStage,
                    MemoryUtil.memAddress(nameUtf8),
                    MemoryUtil.memAddress(entryUtf8),
                    options);

            if (result == MemoryUtil.NULL) {
                throw new RuntimeException(
                        "shaderc_compile_into_spv returned NULL for shader: " + shaderName);
            }

            int status = Shaderc.shaderc_result_get_compilation_status(result);
            if (status != Shaderc.shaderc_compilation_status_success) {
                String errorMessage = Shaderc.shaderc_result_get_error_message(result);
                if (errorMessage == null) {
                    errorMessage = "<no error message>";
                }
                throw new RuntimeException(
                        "Shader compilation failed [" + shaderName + "] (status=" + status + "):\n"
                                + errorMessage);
            }

            long length = Shaderc.shaderc_result_get_length(result);
            if (length <= 0L) {
                throw new RuntimeException(
                        "Shader compilation produced empty SPIR-V [" + shaderName + "]");
            }

            ByteBuffer spirvView = Shaderc.shaderc_result_get_bytes(result);
            if (spirvView == null) {
                throw new RuntimeException(
                        "shaderc_result_get_bytes returned NULL for shader: " + shaderName);
            }

            // Copy out of shaderc-owned memory into a caller-owned native buffer.
            ByteBuffer out = MemoryUtil.memAlloc((int) length);
            try {
                ByteBuffer src = spirvView.duplicate();
                src.position(0);
                src.limit((int) length);
                out.put(src);
                out.flip();
            } catch (Throwable t) {
                MemoryUtil.memFree(out);
                throw t;
            }
            return out;
        } finally {
            if (entryUtf8 != null) {
                MemoryUtil.memFree(entryUtf8);
            }
            if (nameUtf8 != null) {
                MemoryUtil.memFree(nameUtf8);
            }
            if (srcUtf8 != null) {
                MemoryUtil.memFree(srcUtf8);
            }
            if (result != MemoryUtil.NULL) {
                Shaderc.shaderc_result_release(result);
            }
            if (options != MemoryUtil.NULL) {
                Shaderc.shaderc_compile_options_release(options);
            }
            if (compiler != MemoryUtil.NULL) {
                Shaderc.shaderc_compiler_release(compiler);
            }
        }
    }

    /**
     * Creates a {@code VkShaderModule} from SPIR-V bytecode.
     *
     * @param device     the logical Vulkan device
     * @param spirvCode  SPIR-V bytecode (position/limit must describe the code region)
     * @return the raw {@code VkShaderModule} handle
     * @throws RuntimeException if {@code vkCreateShaderModule} fails
     */
    public static long createShaderModule(VkDevice device, ByteBuffer spirvCode) {
        if (device == null) {
            throw new IllegalArgumentException("device must not be null");
        }
        if (spirvCode == null) {
            throw new IllegalArgumentException("spirvCode must not be null");
        }
        if (spirvCode.remaining() <= 0) {
            throw new IllegalArgumentException("spirvCode is empty");
        }
        if ((spirvCode.remaining() & 3) != 0) {
            throw new IllegalArgumentException(
                    "SPIR-V bytecode length must be a multiple of 4 bytes, got: " + spirvCode.remaining());
        }

        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkShaderModuleCreateInfo createInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
                    .pCode(spirvCode);

            LongBuffer pShaderModule = stack.mallocLong(1);
            int err = vkCreateShaderModule(device, createInfo, null, pShaderModule);
            if (err != VK_SUCCESS) {
                throw new RuntimeException("vkCreateShaderModule failed with VkResult: " + err);
            }
            return pShaderModule.get(0);
        }
    }

    /**
     * Loads a GLSL shader from the classpath, compiles it to SPIR-V, and creates
     * a {@code VkShaderModule}. The intermediate SPIR-V native buffer is freed
     * before returning.
     *
     * @param device       the logical Vulkan device
     * @param resourcePath classpath resource path, e.g. {@code /assets/valkyrie/shaders/gbuffer.vert}
     * @param shadercStage one of {@link #STAGE_VERTEX}, {@link #STAGE_FRAGMENT}, {@link #STAGE_COMPUTE}
     * @return the raw {@code VkShaderModule} handle
     * @throws RuntimeException if the resource is missing, unreadable, or compilation fails
     */
    public static long compileResourceToModule(VkDevice device, String resourcePath, int shadercStage) {
        if (resourcePath == null) {
            throw new IllegalArgumentException("resourcePath must not be null");
        }
        String source = loadResourceAsString(resourcePath);
        ByteBuffer spirv = compileGlslToSpirv(resourcePath, source, shadercStage);
        try {
            return createShaderModule(device, spirv);
        } finally {
            MemoryUtil.memFree(spirv);
        }
    }

    /**
     * Reads a UTF-8 text resource from the classpath.
     */
    private static String loadResourceAsString(String resourcePath) {
        try (InputStream in = VkShaderCompiler.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new RuntimeException("Shader resource not found on classpath: " + resourcePath);
            }
            byte[] bytes = in.readAllBytes();
            return new String(bytes, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read shader resource: " + resourcePath, e);
        }
    }
}
