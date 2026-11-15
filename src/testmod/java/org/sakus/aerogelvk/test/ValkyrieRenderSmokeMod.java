package org.sakus.aerogelvk.test;

import org.sakus.aerogelvk.ValkyrieRenderer;
import org.sakus.aerogelvk.compat.CompatManager;
import org.sakus.aerogelvk.config.ValkyrieConfig;
import org.sakus.aerogelvk.meshing.AtlasUVTable;
import org.sakus.aerogelvk.vulkan.VkInteropManager;
import org.sakus.aerogelvk.vulkan.VkMegaBuffer;
import com.mojang.blaze3d.platform.NativeImage;
import com.terraformersmc.modmenu.ModMenu;
import me.shedaniel.clothconfig2.gui.ClothConfigScreen;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.Screenshot;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.world.Difficulty;
import net.minecraft.world.level.GameRules;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.LevelSettings;
import net.minecraft.world.level.WorldDataConfiguration;
import net.minecraft.world.level.levelgen.WorldOptions;
import net.minecraft.world.level.levelgen.presets.WorldPresets;
import org.lwjgl.opengl.GL11;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Isolated testmod client harness that boots a fresh singleplayer world,
 * waits for chunks and Vulkan/OpenGL frames to render, captures a screenshot,
 * verifies non-trivial rendered pixels, and shuts down cleanly.
 */
public final class ValkyrieRenderSmokeMod implements ClientModInitializer {

    private static final Logger LOGGER = LoggerFactory.getLogger(ValkyrieRenderSmokeMod.class);
    private static final String WORLD_ID = "ValkyrieRenderSmoke";
    private static final int MAX_TOTAL_TICKS = 1200; // 60s @ 20tps
    private static final int MAX_SWEEP_TICKS = 6000;
    private static final int SETTLE_TICKS_AFTER_SECTIONS = 30;
    private static final int SWEEP_MIN_FRAMES = 6;
    private static final int SWEEP_MIN_TICKS = 10;

    private boolean worldLaunchRequested = false;
    private int overlayWaitTicks = 0;
    private int totalTicks = 0;
    private int ticksWithActiveSections = 0;
    private int renderedFramesInWorld = 0;
    private long firstInWorldFrameNanos = 0L;
    private long lastInWorldFrameNanos = 0L;
    private boolean finished = false;
    private boolean noonSet = false;

    private boolean sweepMode = false;
    private boolean configGuiOpened = false;
    private int configGuiTicks = 0;
    private boolean sweepInitialized = false;
    private final List<Path> sweepPacks = new ArrayList<>();
    private int sweepIndex = -1; // -1 = builtin Vulkan, 0..n-1 = shaderpacks
    private String currentSweepLabel = "builtin-vulkan";
    private int phaseTicks = 0;
    private int phaseFrames = 0;
    private long phaseFirstSteadyNanos = 0L;
    private long phaseLastSteadyNanos = 0L;
    private boolean phaseLoadOk = true;
    private long phaseCompileMs = 0L;

    @Override
    public void onInitializeClient() {
        LOGGER.info("[ValkyrieSmoke] Smoke-test mod initialized.");

        WorldRenderEvents.END.register(context -> {
            if (finished) {
                return;
            }
            Minecraft mc = Minecraft.getInstance();
            if (mc.player != null && mc.level != null && mc.screen == null) {
                long now = System.nanoTime();
                if (firstInWorldFrameNanos == 0L) {
                    firstInWorldFrameNanos = now;
                }
                lastInWorldFrameNanos = now;
                renderedFramesInWorld++;

                if (sweepInitialized) {
                    phaseFrames++;
                    // Measure steady-state FPS starting at frame 2 to exclude shader swap/first-frame upload
                    if (phaseFrames == 2) {
                        phaseFirstSteadyNanos = now;
                    }
                    if (phaseFrames >= 2) {
                        phaseLastSteadyNanos = now;
                    }
                }
            }
        });

        ClientTickEvents.END_CLIENT_TICK.register(mc -> {
            if (finished) {
                return;
            }

            if (Boolean.getBoolean("aerogelvk.smoke.configGuiVerify")) {
                if (mc.getOverlay() == null && mc.screen != null) {
                    finished = true;
                    verifyPersistedConfigAfterRestart(mc);
                }
                return;
            }

            if (Boolean.getBoolean("aerogelvk.smoke.configGuiTest")) {
                handleConfigGuiSmoke(mc);
                return;
            }

            if (!worldLaunchRequested) {
                overlayWaitTicks++;
                // Wait until LoadingOverlay has finished resource & block atlas loading before opening a world;
                // launching during LoadingOverlay deadlocks Minecraft.doWorldLoad() waiting for overlay == null.
                if (mc.getOverlay() == null && mc.screen != null) {
                    worldLaunchRequested = true;
                    sweepMode = Files.exists(Path.of("shaderpacks/.smoke_sweep"));
                    mc.options.onboardAccessibility = false;
                    mc.options.pauseOnLostFocus = false;
                    mc.options.renderDistance().set(4);
                    mc.options.save();
                    mc.execute(() -> launchFreshSmokeWorld(mc));
                } else if (overlayWaitTicks >= MAX_TOTAL_TICKS) {
                    finished = true;
                    LOGGER.error("[ValkyrieSmoke] Timed out waiting for startup LoadingOverlay to finish (overlay={}, screen={})",
                            mc.getOverlay(), mc.screen);
                    mc.stop();
                    System.exit(2);
                }
                return;
            }

            totalTicks++;

            if (mc.player != null && mc.level != null && mc.screen == null) {
                if (!noonSet) {
                    noonSet = true;
                    mc.level.setDayTime(6000L);
                    if (mc.getSingleplayerServer() != null) {
                        mc.getSingleplayerServer().execute(() ->
                                mc.getSingleplayerServer().overworld().setDayTime(6000L));
                    }
                }
                // Tilt camera slightly downward so both terrain and sky occupy the viewport
                mc.player.setXRot(22.0f);

                VkMegaBuffer mega = VkMegaBuffer.getInstance();
                VkInteropManager interop = VkInteropManager.getInstance();
                int activeSections = mega.isInitialized() ? mega.getActiveSectionCount() : 0;
                boolean externalShaderPipeline = CompatManager.shouldYieldToExternalShaderPipeline();

                if ((interop.isInitialized() && activeSections > 0)
                        || (externalShaderPipeline && renderedFramesInWorld > 0)) {
                    ticksWithActiveSections++;
                }

                if (sweepMode) {
                    if (!sweepInitialized) {
                        if (ticksWithActiveSections >= SETTLE_TICKS_AFTER_SECTIONS) {
                            initSweep();
                        }
                        return;
                    }

                    phaseTicks++;
                    if (phaseFrames >= SWEEP_MIN_FRAMES && phaseTicks >= SWEEP_MIN_TICKS) {
                        recordSweepResult(mc, activeSections);
                        advanceSweep(mc);
                    }
                    return;
                }

                if (totalTicks % 20 == 0) {
                    double elapsedSec = (firstInWorldFrameNanos > 0 && lastInWorldFrameNanos > firstInWorldFrameNanos)
                            ? (lastInWorldFrameNanos - firstInWorldFrameNanos) / 1_000_000_000.0
                            : 0.0;
                    double avgFps = (elapsedSec > 0 && renderedFramesInWorld > 1)
                            ? (renderedFramesInWorld - 1) / elapsedSec
                            : 0.0;
                    LOGGER.info("[ValkyrieSmoke] tick={} inWorld activeSections={} slotLimit={} interopInit={} externalGL={} settle={}/{} mcFps={} frames={} avgFps={}",
                            totalTicks, activeSections,
                            mega.isInitialized() ? mega.getSlotDispatchLimit() : 0,
                            interop.isInitialized(), externalShaderPipeline,
                            ticksWithActiveSections, SETTLE_TICKS_AFTER_SECTIONS,
                            mc.getFps(), renderedFramesInWorld, String.format("%.2f", avgFps));
                }

                if (ticksWithActiveSections >= SETTLE_TICKS_AFTER_SECTIONS) {
                    finished = true;
                    captureAndValidate(mc, activeSections);
                    return;
                }
            }

            int tickLimit = sweepMode ? MAX_SWEEP_TICKS : MAX_TOTAL_TICKS;
            if (totalTicks >= tickLimit) {
                finished = true;
                int activeSections = VkMegaBuffer.getInstance().isInitialized()
                        ? VkMegaBuffer.getInstance().getActiveSectionCount()
                        : -1;
                LOGGER.error("[ValkyrieSmoke] Timed out waiting for rendered world (screen={}, player={}, activeSections={})",
                        mc.screen, mc.player, activeSections);
                captureBestEffortScreenshot(mc, "build/render-smoke-timeout.png");
                mc.stop();
                System.exit(2);
            }
        });
    }

    private void initSweep() {
        sweepInitialized = true;
        sweepPacks.clear();
        Path dir = Path.of("shaderpacks");
        if (Files.isDirectory(dir)) {
            try (Stream<Path> s = Files.list(dir)) {
                s.filter(p -> p.getFileName().toString().endsWith(".zip"))
                        .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                        .forEach(sweepPacks::add);
            } catch (IOException e) {
                LOGGER.warn("[ValkyrieSmokeSweep] Failed listing shaderpacks", e);
            }
        }
        sweepIndex = -1;
        currentSweepLabel = "builtin-vulkan";
        long t0 = System.nanoTime();
        ValkyrieRenderer.getInstance().unloadShaderPack();
        phaseCompileMs = (System.nanoTime() - t0) / 1_000_000L;
        phaseLoadOk = true;
        resetPhaseCounters();
        LOGGER.info("[ValkyrieSmokeSweep] Starting sweep across builtin-vulkan + {} shaderpacks", sweepPacks.size());
    }

    private void resetPhaseCounters() {
        phaseTicks = 0;
        phaseFrames = 0;
        phaseFirstSteadyNanos = 0L;
        phaseLastSteadyNanos = 0L;
    }

    private void recordSweepResult(Minecraft mc, int activeSections) {
        String safeName = currentSweepLabel.replaceAll("[^A-Za-z0-9._-]+", "_");
        Path out = Path.of("build/smoke-sweep", safeName + ".png").toAbsolutePath();
        try (NativeImage img = Screenshot.takeScreenshot(mc.getMainRenderTarget())) {
            Files.createDirectories(out.getParent());
            img.writeToFile(out);

            int w = img.getWidth();
            int h = img.getHeight();
            Set<Integer> distinctColors = new HashSet<>();
            int nonBlackCount = 0;
            int totalSamples = 0;
            int stepX = Math.max(1, w / 32);
            int stepY = Math.max(1, h / 18);
            for (int y = 0; y < h; y += stepY) {
                for (int x = 0; x < w; x += stepX) {
                    int rgb = img.getPixelRGBA(x, y) & 0x00FFFFFF;
                    distinctColors.add(rgb);
                    if (rgb != 0) {
                        nonBlackCount++;
                    }
                    totalSamples++;
                }
            }
            int centerPx = img.getPixelRGBA(w / 2, h / 2);
            int upperPx = img.getPixelRGBA(w / 2, h / 4);
            int lowerPx = img.getPixelRGBA(w / 2, (h * 3) / 4);

            int steadyIntervals = Math.max(0, phaseFrames - 2);
            double elapsedSec = (phaseFirstSteadyNanos > 0 && phaseLastSteadyNanos > phaseFirstSteadyNanos)
                    ? (phaseLastSteadyNanos - phaseFirstSteadyNanos) / 1_000_000_000.0
                    : 0.0;
            double avgFps = (elapsedSec > 0 && steadyIntervals > 0)
                    ? steadyIntervals / elapsedSec
                    : 0.0;
            double avgFrameMs = (avgFps > 0.0) ? (1000.0 / avgFps) : 0.0;

            LOGGER.info("[ValkyrieSmokeSweep] PACK={} loadOk={} compileMs={} activeSections={} samples={}/{} distinctColors={} frames={} steadyElapsedSec={} avgFps={} avgFrameMs={} upperRGB=({},{},{}) centerRGB=({},{},{}) lowerRGB=({},{},{}) screenshot={}",
                    currentSweepLabel,
                    phaseLoadOk,
                    phaseCompileMs,
                    activeSections,
                    nonBlackCount, totalSamples,
                    distinctColors.size(),
                    phaseFrames,
                    String.format("%.3f", elapsedSec),
                    String.format("%.2f", avgFps),
                    String.format("%.1f", avgFrameMs),
                    (upperPx) & 0xFF, (upperPx >> 8) & 0xFF, (upperPx >> 16) & 0xFF,
                    (centerPx) & 0xFF, (centerPx >> 8) & 0xFF, (centerPx >> 16) & 0xFF,
                    (lowerPx) & 0xFF, (lowerPx >> 8) & 0xFF, (lowerPx >> 16) & 0xFF,
                    out);
        } catch (Throwable t) {
            LOGGER.error("[ValkyrieSmokeSweep] PACK={} screenshot/validation error", currentSweepLabel, t);
        }
    }

    private void advanceSweep(Minecraft mc) {
        sweepIndex++;
        if (sweepIndex == sweepPacks.size()) {
            currentSweepLabel = "builtin-vulkan-steady";
            long t0 = System.nanoTime();
            ValkyrieRenderer.getInstance().unloadShaderPack();
            phaseCompileMs = (System.nanoTime() - t0) / 1_000_000L;
            phaseLoadOk = true;
            resetPhaseCounters();
            return;
        }
        if (sweepIndex > sweepPacks.size()) {
            finished = true;
            LOGGER.info("[ValkyrieSmokeSweep] ALL {} PACKS SWEPT SUCCESSFULLY!", sweepPacks.size());
            mc.stop();
            return;
        }
        Path nextPack = sweepPacks.get(sweepIndex);
        currentSweepLabel = nextPack.getFileName().toString();
        long t0 = System.nanoTime();
        phaseLoadOk = ValkyrieRenderer.getInstance().loadShaderPack(nextPack);
        phaseCompileMs = (System.nanoTime() - t0) / 1_000_000L;
        resetPhaseCounters();
    }

    private void launchFreshSmokeWorld(Minecraft mc) {
        try {
            Path saveDir = mc.getLevelSource().getBaseDir().resolve(WORLD_ID);
            if (Files.exists(saveDir)) {
                try (Stream<Path> walk = Files.walk(saveDir)) {
                    walk.sorted(Comparator.reverseOrder()).forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException e) {
                            LOGGER.warn("[ValkyrieSmoke] Failed deleting old save file {}", p, e);
                        }
                    });
                }
            }
        } catch (IOException e) {
            LOGGER.warn("[ValkyrieSmoke] Could not clean old smoke world directory", e);
        }

        LOGGER.info("[ValkyrieSmoke] Creating and loading singleplayer smoke world '{}'...", WORLD_ID);
        LevelSettings settings = new LevelSettings(
                WORLD_ID,
                GameType.CREATIVE,
                false,
                Difficulty.PEACEFUL,
                true,
                new GameRules(),
                WorldDataConfiguration.DEFAULT
        );
        WorldOptions options = new WorldOptions(12345L, false, false);
        mc.createWorldOpenFlows().createFreshLevel(
                WORLD_ID,
                settings,
                options,
                WorldPresets::createNormalWorldDimensions,
                mc.screen
        );
    }

    private void captureBestEffortScreenshot(Minecraft mc, String outRelPath) {
        try (NativeImage img = Screenshot.takeScreenshot(mc.getMainRenderTarget())) {
            Path out = Path.of(outRelPath).toAbsolutePath();
            Files.createDirectories(out.getParent());
            img.writeToFile(out);
            LOGGER.info("[ValkyrieSmoke] Wrote diagnostic screenshot to {}", out);
        } catch (Throwable t) {
            LOGGER.error("[ValkyrieSmoke] Failed to capture diagnostic screenshot", t);
        }
    }

    private void captureAndValidate(Minecraft mc, int activeSections) {
        Path out = Path.of("build/render-smoke-test.png").toAbsolutePath();
        try (NativeImage img = Screenshot.takeScreenshot(mc.getMainRenderTarget())) {
            Files.createDirectories(out.getParent());
            img.writeToFile(out);

            int w = img.getWidth();
            int h = img.getHeight();
            Set<Integer> distinctColors = new HashSet<>();
            int nonBlackCount = 0;
            int totalSamples = 0;

            int stepX = Math.max(1, w / 32);
            int stepY = Math.max(1, h / 18);
            for (int y = 0; y < h; y += stepY) {
                for (int x = 0; x < w; x += stepX) {
                    int abgr = img.getPixelRGBA(x, y);
                    int rgb = abgr & 0x00FFFFFF;
                    distinctColors.add(rgb);
                    if (rgb != 0) {
                        nonBlackCount++;
                    }
                    totalSamples++;
                }
            }

            int centerPx = img.getPixelRGBA(w / 2, h / 2);
            int upperPx = img.getPixelRGBA(w / 2, h / 4);
            int lowerPx = img.getPixelRGBA(w / 2, (h * 3) / 4);

            double elapsedSec = (firstInWorldFrameNanos > 0 && lastInWorldFrameNanos > firstInWorldFrameNanos)
                    ? (lastInWorldFrameNanos - firstInWorldFrameNanos) / 1_000_000_000.0
                    : 0.0;
            double avgFps = (elapsedSec > 0 && renderedFramesInWorld > 1)
                    ? (renderedFramesInWorld - 1) / elapsedSec
                    : 0.0;
            double avgFrameMs = (avgFps > 0.0) ? (1000.0 / avgFps) : 0.0;

            String glRenderer = GL11.glGetString(GL11.GL_RENDERER);
            LOGGER.info("[ValkyrieSmoke] Captured {} ({}x{}): mode={}, glRenderer='{}', activeSections={}, sprites={}, samples={}/{}, distinctColors={}, mcFps={}, frames={}, elapsedSec={}, avgFps={}, avgFrameMs={}, upperRGB=({}, {}, {}), centerRGB=({}, {}, {}), lowerRGB=({}, {}, {})",
                    out, w, h,
                    VkInteropManager.getInstance().getInteropMode(),
                    glRenderer,
                    activeSections,
                    AtlasUVTable.getInstance().getSpriteCount(),
                    nonBlackCount, totalSamples, distinctColors.size(),
                    mc.getFps(), renderedFramesInWorld,
                    String.format("%.2f", elapsedSec),
                    String.format("%.2f", avgFps),
                    String.format("%.1f", avgFrameMs),
                    (upperPx) & 0xFF, (upperPx >> 8) & 0xFF, (upperPx >> 16) & 0xFF,
                    (centerPx) & 0xFF, (centerPx >> 8) & 0xFF, (centerPx >> 16) & 0xFF,
                    (lowerPx) & 0xFF, (lowerPx >> 8) & 0xFF, (lowerPx >> 16) & 0xFF);

            if (nonBlackCount == 0 || distinctColors.size() < 4) {
                LOGGER.error("[ValkyrieSmoke] Render output validation failed (too flat/black)! distinctColors={}", distinctColors.size());
                mc.stop();
                System.exit(3);
                return;
            }
        } catch (Throwable t) {
            LOGGER.error("[ValkyrieSmoke] Exception during screenshot validation", t);
            mc.stop();
            System.exit(4);
            return;
        }

        LOGGER.info("[ValkyrieSmoke] IN-WORLD RENDER SMOKE TEST PASSED!");
        mc.stop();
    }

    private void handleConfigGuiSmoke(Minecraft mc) {
        if (!configGuiOpened) {
            if (mc.getOverlay() == null && mc.screen != null) {
                configGuiOpened = true;
                try {
                    Path cfgPath = mc.gameDirectory.toPath().resolve("config/aerogelvk.properties");
                    Files.createDirectories(cfgPath.getParent());
                    java.util.Properties props = new java.util.Properties();
                    if (Files.exists(cfgPath)) {
                        try (var in = Files.newInputStream(cfgPath)) {
                            props.load(in);
                        }
                    }
                    props.setProperty("unknownSmokeKey", "preserve_me_123");
                    try (var out = Files.newOutputStream(cfgPath)) {
                        props.store(out, "Config GUI smoke pre-seed");
                    }
                    ValkyrieConfig.get().reload();
                } catch (IOException e) {
                    LOGGER.error("[ValkyrieSmoke] Failed pre-seeding unknownSmokeKey", e);
                    mc.stop();
                    System.exit(5);
                    return;
                }

                Screen cfgScreen = ModMenu.getConfigScreen("aerogelvk", mc.screen);
                if (cfgScreen == null) {
                    LOGGER.error("[ValkyrieSmoke] ModMenu.getConfigScreen('aerogelvk') returned null!");
                    mc.stop();
                    System.exit(6);
                    return;
                }
                LOGGER.info("[ValkyrieSmoke] Opened ModMenu config screen: {}", cfgScreen.getClass().getName());
                mc.setScreen(cfgScreen);
            }
            return;
        }

        configGuiTicks++;
        if (configGuiTicks < 8) {
            return;
        }
        finished = true;

        try {
            Path out = Path.of("build/config-gui-smoke.png").toAbsolutePath();
            Files.createDirectories(out.getParent());
            try (NativeImage img = Screenshot.takeScreenshot(mc.getMainRenderTarget())) {
                img.writeToFile(out);
                LOGGER.info("[ValkyrieSmoke] Captured ModMenu/ClothConfig GUI screenshot: {} ({}x{})",
                        out, img.getWidth(), img.getHeight());
            }
            if (mc.screen instanceof ClothConfigScreen clothScreen) {
                clothScreen.saveAll(false);
            }
            ValkyrieConfig cfg = ValkyrieConfig.get();
            cfg.setExposure(2.3f);
            cfg.setCsmCascades(2);
            cfg.setFxaaEnabled(false);
            cfg.save();
            LOGGER.info("[ValkyrieSmoke] Saved modified config for restart persistence check (exposure={}, csmCascades={}, fxaaEnabled={}).",
                    cfg.getExposure(), cfg.getCsmCascades(), cfg.isFxaaEnabled());
        } catch (Throwable t) {
            LOGGER.error("[ValkyrieSmoke] Config GUI smoke failed", t);
            mc.stop();
            System.exit(7);
            return;
        }
        mc.stop();
    }

    private void verifyPersistedConfigAfterRestart(Minecraft mc) {
        try {
            ValkyrieConfig cfg = ValkyrieConfig.get();
            cfg.loadOrCreate();
            Path cfgPath = mc.gameDirectory.toPath().resolve("config/aerogelvk.properties");
            java.util.Properties props = new java.util.Properties();
            try (var in = Files.newInputStream(cfgPath)) {
                props.load(in);
            }
            String unknown = props.getProperty("unknownSmokeKey");
            if (Math.abs(cfg.getExposure() - 2.3f) > 0.01f
                    || cfg.getCsmCascades() != 2
                    || cfg.isFxaaEnabled()
                    || !"preserve_me_123".equals(unknown)) {
                LOGGER.error("[ValkyrieSmoke] Persisted config verification FAILED! exposure={}, csmCascades={}, fxaaEnabled={}, unknownSmokeKey={}",
                        cfg.getExposure(), cfg.getCsmCascades(), cfg.isFxaaEnabled(), unknown);
                mc.stop();
                System.exit(8);
                return;
            }
            LOGGER.info("[ValkyrieSmoke] CONFIG PERSISTENCE AFTER RESTART PASSED! (exposure={}, csmCascades={}, fxaaEnabled={}, unknownSmokeKey={})",
                    cfg.getExposure(), cfg.getCsmCascades(), cfg.isFxaaEnabled(), unknown);
        } catch (Throwable t) {
            LOGGER.error("[ValkyrieSmoke] Restart persistence check exception", t);
            mc.stop();
            System.exit(9);
            return;
        }
        mc.stop();
    }
}
