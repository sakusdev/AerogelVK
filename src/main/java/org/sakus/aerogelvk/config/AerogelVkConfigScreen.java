package org.sakus.aerogelvk.config;

import me.shedaniel.clothconfig2.api.ConfigBuilder;
import me.shedaniel.clothconfig2.api.ConfigCategory;
import me.shedaniel.clothconfig2.api.ConfigEntryBuilder;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;
import org.sakus.aerogelvk.ValkyrieRenderer;
import org.sakus.aerogelvk.compat.CompatManager;
import org.sakus.aerogelvk.shaderpack.ShaderPackManager;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;

/**
 * Cloth Config GUI for AerogelVK ({@link ValkyrieConfig}).
 * Covers all 14 configuration keys, preserves unknown properties on save,
 * and triggers the existing runtime reload path upon saving.
 */
public final class AerogelVkConfigScreen {

    private AerogelVkConfigScreen() {
    }

    public static Screen create(Screen parent) {
        ValkyrieConfig config = ValkyrieConfig.get();
        config.loadOrCreate();

        ConfigBuilder builder = ConfigBuilder.create()
                .setParentScreen(parent)
                .setTitle(Component.translatable("config.aerogelvk.title"));

        ConfigEntryBuilder entryBuilder = builder.entryBuilder();

        // 1. Rendering
        ConfigCategory rendering = builder.getOrCreateCategory(
                Component.translatable("config.aerogelvk.category.rendering"));

        rendering.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.hiZOcclusionEnabled"),
                        config.isHiZOcclusionEnabled())
                .setDefaultValue(true)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.hiZOcclusionEnabled"))
                .setSaveConsumer(config::setHiZOcclusionEnabled)
                .build());

        rendering.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.ssrEnabled"),
                        config.isSsrEnabled())
                .setDefaultValue(true)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.ssrEnabled"))
                .setSaveConsumer(config::setSsrEnabled)
                .build());

        rendering.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.fxaaEnabled"),
                        config.isFxaaEnabled())
                .setDefaultValue(true)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.fxaaEnabled"))
                .setSaveConsumer(config::setFxaaEnabled)
                .build());

        int initialExposureTenths = Math.max(1, Math.min(40, Math.round(config.getExposure() * 10.0f)));
        rendering.addEntry(entryBuilder.startIntSlider(
                        Component.translatable("config.aerogelvk.option.exposure"),
                        initialExposureTenths, 1, 40)
                .setDefaultValue(10)
                .setTextGetter(v -> Component.literal(String.format(Locale.ROOT, "%.1f", v / 10.0f)))
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.exposure"))
                .setSaveConsumer(v -> config.setExposure(v / 10.0f))
                .build());

        // 2. Shadows & CSM
        ConfigCategory shadows = builder.getOrCreateCategory(
                Component.translatable("config.aerogelvk.category.shadows"));

        shadows.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.csmEnabled"),
                        config.isCsmEnabled())
                .setDefaultValue(true)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.csmEnabled"))
                .setSaveConsumer(config::setCsmEnabled)
                .build());

        shadows.addEntry(entryBuilder.startIntSlider(
                        Component.translatable("config.aerogelvk.option.csmCascades"),
                        config.getCsmCascades(), 1, 3)
                .setDefaultValue(3)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.csmCascades"))
                .setSaveConsumer(config::setCsmCascades)
                .build());

        shadows.addEntry(entryBuilder.startIntSlider(
                        Component.translatable("config.aerogelvk.option.csmUpdateInterval"),
                        config.getCsmUpdateInterval(), 1, 30)
                .setDefaultValue(6)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.csmUpdateInterval"))
                .setSaveConsumer(config::setCsmUpdateInterval)
                .build());

        shadows.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.fastShadowPcf"),
                        config.isFastShadowPcf())
                .setDefaultValue(true)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.fastShadowPcf"))
                .setSaveConsumer(config::setFastShadowPcf)
                .build());

        shadows.addEntry(entryBuilder.startIntSlider(
                        Component.translatable("config.aerogelvk.option.csmResolution"),
                        config.getCsmResolution(), 512, 2048)
                .setDefaultValue(1024)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.csmResolution"))
                .requireRestart()
                .setSaveConsumer(config::setCsmResolution)
                .build());

        // 3. Shader Pack
        ConfigCategory shaderPackCat = builder.getOrCreateCategory(
                Component.translatable("config.aerogelvk.category.shaderpack"));

        final boolean[] pendingPackEnabled = {config.isShaderPackEnabled()};
        final String[] pendingPackName = {config.getShaderPack()};

        shaderPackCat.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.shaderPackEnabled"),
                        config.isShaderPackEnabled())
                .setDefaultValue(false)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.shaderPackEnabled"))
                .setSaveConsumer(v -> pendingPackEnabled[0] = v)
                .build());

        String currentPack = config.getShaderPack();
        if (currentPack == null || currentPack.isBlank()) {
            currentPack = "off";
        }
        Minecraft mc = Minecraft.getInstance();
        Path gameDir = (mc != null && mc.gameDirectory != null) ? mc.gameDirectory.toPath() : Path.of(".");
        List<String> packOptions = buildShaderPackOptions(gameDir, currentPack);

        shaderPackCat.addEntry(entryBuilder.startStringDropdownMenu(
                        Component.translatable("config.aerogelvk.option.shaderPack"),
                        currentPack)
                .setSelections(packOptions)
                .setSuggestionMode(false)
                .setDefaultValue("off")
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.shaderPack"))
                .setSaveConsumer(v -> pendingPackName[0] = v)
                .build());

        int initialScalePct = Math.max(50, Math.min(100, Math.round(config.getShaderPackRenderScale() * 100.0f)));
        shaderPackCat.addEntry(entryBuilder.startIntSlider(
                        Component.translatable("config.aerogelvk.option.shaderPackRenderScale"),
                        initialScalePct, 50, 100)
                .setDefaultValue(100)
                .setTextGetter(v -> Component.literal(v + "%"))
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.shaderPackRenderScale"))
                .setSaveConsumer(v -> config.setShaderPackRenderScale(v / 100.0f))
                .build());

        shaderPackCat.addEntry(entryBuilder.startIntField(
                        Component.translatable("config.aerogelvk.option.shaderPackShadowResolutionCap"),
                        config.getShaderPackShadowResolutionCap())
                .setDefaultValue(0)
                .setMin(0)
                .setMax(8192)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.shaderPackShadowResolutionCap"))
                .setSaveConsumer(config::setShaderPackShadowResolutionCap)
                .build());

        // 4. Compatibility
        ConfigCategory compat = builder.getOrCreateCategory(
                Component.translatable("config.aerogelvk.category.compat"));

        compat.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.sodiumCompatEnabled"),
                        config.isSodiumCompatEnabled())
                .setDefaultValue(true)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.sodiumCompatEnabled"))
                .setSaveConsumer(config::setSodiumCompatEnabled)
                .build());

        compat.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.irisCompatEnabled"),
                        config.isIrisCompatEnabled())
                .setDefaultValue(true)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.irisCompatEnabled"))
                .setSaveConsumer(config::setIrisCompatEnabled)
                .build());

        compat.addEntry(entryBuilder.startBooleanToggle(
                        Component.translatable("config.aerogelvk.option.dhCompatEnabled"),
                        config.isDhCompatEnabled())
                .setDefaultValue(true)
                .setTooltip(Component.translatable("config.aerogelvk.tooltip.dhCompatEnabled"))
                .setSaveConsumer(config::setDhCompatEnabled)
                .build());

        builder.setSavingRunnable(() -> {
            applyShaderPackFields(config, pendingPackEnabled[0], pendingPackName[0]);
            config.save();
            reloadRuntimeFromConfig(config);
        });

        return builder.build();
    }

    public static List<String> buildShaderPackOptions(Path gameDir, String currentPack) {
        LinkedHashSet<String> options = new LinkedHashSet<>();
        options.add("off");
        options.add("auto");
        if (gameDir != null) {
            options.addAll(ShaderPackManager.listShaderPackNames(gameDir));
        }
        if (currentPack != null && !currentPack.isBlank() && !options.contains(currentPack)) {
            options.add(currentPack);
        }
        return new ArrayList<>(options);
    }

    private static void applyShaderPackFields(ValkyrieConfig config, boolean enabled, String packName) {
        if (!enabled) {
            config.setShaderPack(packName);
            config.setShaderPackEnabled(false);
        } else {
            config.setShaderPack(packName);
            config.setShaderPackEnabled(true);
        }
    }

    /**
     * Mirrors the {@code /aerogelvk shaderpack reload} runtime path so saved settings
     * take effect immediately in the active renderer.
     */
    private static void reloadRuntimeFromConfig(ValkyrieConfig config) {
        config.reload();
        ValkyrieRenderer renderer = ValkyrieRenderer.getInstance();
        if (!config.isShaderPackConfigured()) {
            renderer.unloadShaderPack();
            return;
        }
        if (CompatManager.shouldYieldToExternalShaderPipeline()) {
            return;
        }
        renderer.reloadShaderPack();
    }
}
