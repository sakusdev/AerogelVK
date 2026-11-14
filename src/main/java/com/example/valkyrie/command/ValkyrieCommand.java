package com.example.valkyrie.command;

import com.example.valkyrie.ValkyrieRenderer;
import com.example.valkyrie.compat.CompatManager;
import com.example.valkyrie.config.ValkyrieConfig;
import com.example.valkyrie.shaderpack.ShaderPackManager;
import com.example.valkyrie.shaderpack.VkShaderPackPipeline;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.suggestion.SuggestionProvider;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

import java.nio.file.Path;
import java.util.List;

/**
 * Registers the client-side {@code /valkyrie} command for inspecting and switching
 * native Vulkan OptiFine/Iris shaderpacks at runtime:
 * <ul>
 *     <li>{@code /valkyrie shaderpack list}</li>
 *     <li>{@code /valkyrie shaderpack set <name>}</li>
 *     <li>{@code /valkyrie shaderpack reload}</li>
 *     <li>{@code /valkyrie shaderpack off}</li>
 * </ul>
 */
public final class ValkyrieCommand {

    private static final SuggestionProvider<FabricClientCommandSource> SHADERPACK_SUGGESTIONS = (context, builder) -> {
        builder.suggest("auto");
        builder.suggest("off");
        Path gameDir = resolveGameDir(context.getSource());
        if (gameDir != null) {
            for (String name : ShaderPackManager.listShaderPackNames(gameDir)) {
                builder.suggest(name);
            }
        }
        return builder.buildFuture();
    };

    private ValkyrieCommand() {
    }

    public static void register() {
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) ->
                dispatcher.register(ClientCommandManager.literal("valkyrie")
                        .then(ClientCommandManager.literal("shaderpack")
                                .executes(ctx -> showStatus(ctx.getSource()))
                                .then(ClientCommandManager.literal("list")
                                        .executes(ctx -> listShaderPacks(ctx.getSource())))
                                .then(ClientCommandManager.literal("set")
                                        .then(ClientCommandManager.argument("name", StringArgumentType.greedyString())
                                                .suggests(SHADERPACK_SUGGESTIONS)
                                                .executes(ctx -> setShaderPack(
                                                        ctx.getSource(),
                                                        StringArgumentType.getString(ctx, "name")))))
                                .then(ClientCommandManager.literal("reload")
                                        .executes(ctx -> reloadShaderPack(ctx.getSource())))
                                .then(ClientCommandManager.literal("off")
                                        .executes(ctx -> disableShaderPack(ctx.getSource())))
                        )
                )
        );
    }

    private static int showStatus(FabricClientCommandSource source) {
        ValkyrieConfig config = ValkyrieConfig.get();
        ValkyrieRenderer renderer = ValkyrieRenderer.getInstance();
        VkShaderPackPipeline pipeline = renderer.getShaderPackPipeline();
        String activeName = (pipeline != null && renderer.isShaderPackActive())
                ? pipeline.getActivePackName()
                : "built-in PBR";
        String irisNote = CompatManager.shouldYieldToExternalShaderPipeline()
                ? " (yielding to Iris)"
                : "";
        source.sendFeedback(Component.literal("[Valkyrie] Config shaderPack=" + config.getShaderPack()
                + ", enabled=" + config.isShaderPackEnabled()
                + ", active=" + activeName + irisNote));
        return 1;
    }

    private static int listShaderPacks(FabricClientCommandSource source) {
        Path gameDir = resolveGameDir(source);
        if (gameDir == null) {
            source.sendError(Component.literal("[Valkyrie] Game directory unavailable."));
            return 0;
        }
        List<String> packs = ShaderPackManager.listShaderPackNames(gameDir);
        ValkyrieConfig config = ValkyrieConfig.get();
        if (packs.isEmpty()) {
            source.sendFeedback(Component.literal(
                    "[Valkyrie] No shaderpacks found in shaderpacks/ or shaders/. Active: "
                            + config.getShaderPack()));
        } else {
            source.sendFeedback(Component.literal(
                    "[Valkyrie] Available shaderpacks (" + packs.size() + "): " + String.join(", ", packs)
                            + " | Configured: " + config.getShaderPack()
                            + " (enabled=" + config.isShaderPackEnabled() + ")"));
        }
        return 1;
    }

    private static int setShaderPack(FabricClientCommandSource source, String rawName) {
        String packName = rawName == null ? "" : rawName.trim();
        if (packName.isEmpty()) {
            source.sendError(Component.literal("[Valkyrie] Specify a shaderpack name, 'auto', or 'off'."));
            return 0;
        }
        if ("off".equalsIgnoreCase(packName) || "none".equalsIgnoreCase(packName)) {
            return disableShaderPack(source);
        }

        Path gameDir = resolveGameDir(source);
        if (!"auto".equalsIgnoreCase(packName) && gameDir != null
                && ShaderPackManager.findShaderPack(gameDir, packName).isEmpty()) {
            source.sendError(Component.literal("[Valkyrie] Shaderpack '" + packName
                    + "' not found in shaderpacks/ or shaders/. Use '/valkyrie shaderpack list'."));
            return 0;
        }

        ValkyrieConfig config = ValkyrieConfig.get();
        config.setShaderPack(packName);
        config.save();

        if (CompatManager.shouldYieldToExternalShaderPipeline()) {
            source.sendFeedback(Component.literal("[Valkyrie] Saved shaderpack '" + config.getShaderPack()
                    + "', currently yielding to active Iris shaderpack."));
            return 1;
        }

        ValkyrieRenderer renderer = ValkyrieRenderer.getInstance();
        if (!renderer.isInitialized()) {
            renderer.reloadShaderPack();
            source.sendFeedback(Component.literal("[Valkyrie] Shaderpack set to '" + config.getShaderPack()
                    + "' (will load on renderer init)."));
            return 1;
        }

        boolean ok = renderer.reloadShaderPack();
        VkShaderPackPipeline pipeline = renderer.getShaderPackPipeline();
        if (ok && pipeline != null) {
            source.sendFeedback(Component.literal("[Valkyrie] Activated Vulkan shaderpack '"
                    + pipeline.getActivePackName() + "' ("
                    + pipeline.getCompiledPasses().size() + " passes, "
                    + pipeline.getCompileWarnings().size() + " warnings)."));
            return 1;
        } else {
            source.sendError(Component.literal("[Valkyrie] Failed to compile shaderpack '" + packName
                    + "'; fell back to built-in deferred PBR pipeline."));
            return 0;
        }
    }

    private static int reloadShaderPack(FabricClientCommandSource source) {
        ValkyrieConfig config = ValkyrieConfig.get();
        config.reload();

        if (!config.isShaderPackConfigured()) {
            ValkyrieRenderer.getInstance().unloadShaderPack();
            source.sendFeedback(Component.literal("[Valkyrie] Config reloaded; shaderpacks are off (built-in PBR active)."));
            return 1;
        }

        if (CompatManager.shouldYieldToExternalShaderPipeline()) {
            source.sendFeedback(Component.literal("[Valkyrie] Config reloaded ('" + config.getShaderPack()
                    + "'), currently yielding to active Iris shaderpack."));
            return 1;
        }

        ValkyrieRenderer renderer = ValkyrieRenderer.getInstance();
        if (!renderer.isInitialized()) {
            renderer.reloadShaderPack();
            source.sendFeedback(Component.literal("[Valkyrie] Queued shaderpack reload for '" + config.getShaderPack() + "'."));
            return 1;
        }

        boolean ok = renderer.reloadShaderPack();
        VkShaderPackPipeline pipeline = renderer.getShaderPackPipeline();
        if (ok && pipeline != null) {
            source.sendFeedback(Component.literal("[Valkyrie] Reloaded Vulkan shaderpack '"
                    + pipeline.getActivePackName() + "' ("
                    + pipeline.getCompiledPasses().size() + " passes, "
                    + pipeline.getCompileWarnings().size() + " warnings)."));
            return 1;
        } else {
            source.sendError(Component.literal("[Valkyrie] Failed to reload shaderpack '"
                    + config.getShaderPack() + "'; fell back to built-in deferred PBR pipeline."));
            return 0;
        }
    }

    private static int disableShaderPack(FabricClientCommandSource source) {
        ValkyrieConfig config = ValkyrieConfig.get();
        config.disableShaderPack();
        config.save();
        ValkyrieRenderer.getInstance().unloadShaderPack();
        source.sendFeedback(Component.literal("[Valkyrie] Shaderpack disabled; built-in deferred PBR pipeline active."));
        return 1;
    }

    private static Path resolveGameDir(FabricClientCommandSource source) {
        try {
            if (source != null && source.getClient() != null && source.getClient().gameDirectory != null) {
                return source.getClient().gameDirectory.toPath();
            }
            Minecraft mc = Minecraft.getInstance();
            if (mc != null && mc.gameDirectory != null) {
                return mc.gameDirectory.toPath();
            }
        } catch (Throwable ignored) {
        }
        return null;
    }
}
