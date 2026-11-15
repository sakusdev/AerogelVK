package org.sakus.aerogelvk.test;

import org.sakus.aerogelvk.ValkyrieRenderer;
import org.sakus.aerogelvk.compat.CompatManager;
import org.sakus.aerogelvk.config.ValkyrieConfig;
import org.sakus.aerogelvk.meshing.AtlasUVTable;
import org.sakus.aerogelvk.meshing.GreedyMesher;
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
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.sounds.SoundSource;
import net.minecraft.world.Difficulty;
import net.minecraft.world.level.GameRules;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.LevelSettings;
import net.minecraft.world.level.WorldDataConfiguration;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.SlabBlock;
import net.minecraft.world.level.block.StairBlock;
import net.minecraft.world.level.block.StandingSignBlock;
import net.minecraft.world.level.block.entity.SignBlockEntity;
import net.minecraft.world.level.block.state.properties.DoubleBlockHalf;
import net.minecraft.world.level.block.state.properties.SlabType;
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
    private static final int SETTLE_TICKS_AFTER_SECTIONS = Integer.getInteger("aerogelvk.smoke.settleTicks", 30);
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
    private boolean shapeShowcaseBuilt = false;
    private volatile boolean shapeShowcaseServerDone = false;
    private int shapeShowcaseSettleTicks = 0;

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
                    String repro = System.getProperty("aerogelvk.smoke.repro", "");
                    int defaultRd = (Boolean.getBoolean("aerogelvk.smoke.shapeShowcase") || (!repro.isEmpty() && !"end".equals(repro))) ? 2 : 4;
                    int rd = Integer.getInteger("aerogelvk.smoke.renderDistance", defaultRd);
                    mc.options.renderDistance().set(rd);
                    mc.options.getSoundSourceOptionInstance(SoundSource.MASTER).set(0.0);
                    mc.options.getSoundSourceOptionInstance(SoundSource.MUSIC).set(0.0);
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

                String reproMode = System.getProperty("aerogelvk.smoke.repro", "");
                if (Boolean.getBoolean("aerogelvk.smoke.shapeShowcase") || !reproMode.isEmpty()) {
                    VkMegaBuffer mega = VkMegaBuffer.getInstance();
                    int activeSections = mega.isInitialized() ? mega.getActiveSectionCount() : 0;
                    if (!shapeShowcaseBuilt && mc.getSingleplayerServer() != null) {
                        shapeShowcaseBuilt = true;
                        mc.getSingleplayerServer().execute(() -> {
                            ServerLevel targetLevel = mc.getSingleplayerServer().overworld();
                            if ("end".equals(reproMode)) {
                                ServerLevel endLevel = mc.getSingleplayerServer().getLevel(net.minecraft.world.level.Level.END);
                                if (endLevel != null) {
                                    targetLevel = endLevel;
                                }
                            } else if (!reproMode.isEmpty()) {
                                buildReproScene(targetLevel, reproMode);
                            } else {
                                buildShapeShowcase(targetLevel);
                            }
                            if ("night".equals(reproMode)) {
                                targetLevel.setDayTime(18000L);
                            } else if ("walk".equals(reproMode)) {
                                targetLevel.setDayTime(2000L);
                            } else {
                                targetLevel.setDayTime(6000L);
                            }
                            final ServerLevel finalLevel = targetLevel;
                            for (var sp : mc.getSingleplayerServer().getPlayerList().getPlayers()) {
                                boolean walkMode = "walk".equals(reproMode);
                                sp.getAbilities().flying = !walkMode;
                                sp.onUpdateAbilities();
                                if ("end".equals(reproMode)) {
                                    sp.teleportTo(finalLevel, 0.0, 72.0, -35.0, java.util.Set.of(), 0.0f, 15.0f);
                                } else if ("redstone".equals(reproMode)) {
                                    sp.teleportTo(finalLevel, 7.5, 162.5, 1.5, java.util.Set.of(), 0.0f, 38.0f);
                                } else if (walkMode) {
                                    sp.teleportTo(finalLevel, 8.0, 160.0, 2.0, java.util.Set.of(), 0.0f, 10.0f);
                                } else {
                                    sp.teleportTo(finalLevel, 8.0, 162.0, -1.5, java.util.Set.of(), 0.0f, 24.0f);
                                }
                            }
                            shapeShowcaseServerDone = true;
                        });
                    } else if (shapeShowcaseBuilt && shapeShowcaseServerDone) {
                        boolean walkMode = "walk".equals(reproMode);
                        mc.player.getAbilities().flying = !walkMode;
                        if ("night".equals(reproMode)) {
                            mc.level.setDayTime(18000L);
                        } else if (walkMode) {
                            mc.level.setDayTime(2000L);
                            mc.options.bobView().set(true);
                            mc.options.keyUp.setDown(true);
                        }
                        if (!"end".equals(reproMode)) {
                            mc.player.setYRot(0.0f);
                            mc.player.setXRot("redstone".equals(reproMode) ? 38.0f : (walkMode ? 10.0f : 24.0f));
                            mc.player.yRotO = 0.0f;
                            mc.player.xRotO = mc.player.getXRot();
                        }
                        if (activeSections > 0
                                && renderedFramesInWorld >= 4
                                && ("end".equals(reproMode) || GreedyMesher.SectionSnapshot.capture(mc.level, 0, 10, 0) != null)) {
                            shapeShowcaseSettleTicks++;
                        }
                        if (shapeShowcaseSettleTicks % 5 == 0 && shapeShowcaseSettleTicks > 0) {
                            LOGGER.info("[ValkyrieSmoke] repro={} settle={}/15 activeSections={}",
                                    reproMode, shapeShowcaseSettleTicks, activeSections);
                        }
                        if (shapeShowcaseSettleTicks >= 15) {
                            if (reproMode.isEmpty()) {
                                logShapeShowcaseSectionMetrics(mc);
                            }
                            finished = true;
                            captureAndValidate(mc, activeSections);
                            return;
                        }
                    }
                    return;
                }

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

                boolean waitChunkSettle = Boolean.getBoolean("aerogelvk.smoke.waitChunkSettle");
                if (waitChunkSettle) {
                    if (ticksWithActiveSections >= 20 && ValkyrieRenderer.getInstance().isChunkQueueIdle()) {
                        double loadElapsedSec = (firstInWorldFrameNanos > 0 && lastInWorldFrameNanos > firstInWorldFrameNanos)
                                ? (lastInWorldFrameNanos - firstInWorldFrameNanos) / 1_000_000_000.0
                                : 0.0;
                        double loadAvgFps = (loadElapsedSec > 0 && renderedFramesInWorld > 1)
                                ? (renderedFramesInWorld - 1) / loadElapsedSec
                                : 0.0;
                        LOGGER.info("[ValkyrieSmoke] CHUNK_SETTLED! activeSections={} elapsedSec={} frames={} loadAvgFps={}",
                                activeSections, String.format("%.2f", loadElapsedSec), renderedFramesInWorld, String.format("%.2f", loadAvgFps));
                        finished = true;
                        captureAndValidate(mc, activeSections);
                        return;
                    }
                } else if (ticksWithActiveSections >= SETTLE_TICKS_AFTER_SECTIONS) {
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
        String customOut = System.getProperty("aerogelvk.smoke.screenshotOut", "build/render-smoke-test.png");
        Path out = Path.of(customOut).toAbsolutePath();
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

    private void buildReproScene(ServerLevel level, String mode) {
        for (int x = 1; x <= 14; x++) {
            for (int z = 1; z <= 14; z++) {
                level.setBlock(new BlockPos(x, 159, z),
                        ("plants".equals(mode) ? Blocks.GRASS_BLOCK : Blocks.STONE).defaultBlockState(),
                        Block.UPDATE_ALL);
                for (int y = 160; y <= 165; y++) {
                    level.setBlock(new BlockPos(x, y, z), Blocks.AIR.defaultBlockState(), Block.UPDATE_ALL);
                }
            }
        }
        switch (mode) {
            case "plants" -> {
                // 1-block controls on the left
                level.setBlock(new BlockPos(2, 160, 5), Blocks.SHORT_GRASS.defaultBlockState(), Block.UPDATE_ALL);
                level.setBlock(new BlockPos(4, 160, 5), Blocks.POPPY.defaultBlockState(), Block.UPDATE_ALL);
                // 2-block tall plants
                Block[] tall = { Blocks.TALL_GRASS, Blocks.ROSE_BUSH, Blocks.LILAC, Blocks.SUNFLOWER, Blocks.LARGE_FERN };
                for (int i = 0; i < tall.length; i++) {
                    int x = 6 + i * 2;
                    level.setBlock(new BlockPos(x, 160, 5),
                            tall[i].defaultBlockState().setValue(net.minecraft.world.level.block.DoublePlantBlock.HALF, DoubleBlockHalf.LOWER),
                            Block.UPDATE_ALL);
                    level.setBlock(new BlockPos(x, 161, 5),
                            tall[i].defaultBlockState().setValue(net.minecraft.world.level.block.DoublePlantBlock.HALF, DoubleBlockHalf.UPPER),
                            Block.UPDATE_ALL);
                }
            }
            case "fluids" -> {
                // Stone rim & staircase for flowing water + lava pool
                for (int x = 2; x <= 13; x++) {
                    level.setBlock(new BlockPos(x, 160, 3), Blocks.STONE.defaultBlockState(), Block.UPDATE_ALL);
                    level.setBlock(new BlockPos(x, 160, 10), Blocks.STONE.defaultBlockState(), Block.UPDATE_ALL);
                }
                for (int z = 3; z <= 10; z++) {
                    level.setBlock(new BlockPos(2, 160, z), Blocks.STONE.defaultBlockState(), Block.UPDATE_ALL);
                    level.setBlock(new BlockPos(9, 160, z), Blocks.STONE.defaultBlockState(), Block.UPDATE_ALL);
                    level.setBlock(new BlockPos(13, 160, z), Blocks.STONE.defaultBlockState(), Block.UPDATE_ALL);
                }
                // Water source at (3, 160, 5..8) flowing east -> steps down
                for (int z = 5; z <= 8; z++) {
                    level.setBlock(new BlockPos(3, 160, z), Blocks.WATER.defaultBlockState(), Block.UPDATE_ALL);
                    for (int lvl = 1; lvl <= 5; lvl++) {
                        level.setBlock(new BlockPos(3 + lvl, 160, z),
                                Blocks.WATER.defaultBlockState().setValue(net.minecraft.world.level.block.LiquidBlock.LEVEL, lvl),
                                Block.UPDATE_ALL);
                    }
                }
                // Lava pool on the right (x=10..12, z=4..9)
                for (int x = 10; x <= 12; x++) {
                    for (int z = 4; z <= 9; z++) {
                        level.setBlock(new BlockPos(x, 160, z), Blocks.LAVA.defaultBlockState(), Block.UPDATE_ALL);
                    }
                }
            }
            case "redstone" -> {
                // Straight line, cross, corner, and step-up redstone wire
                for (int x = 3; x <= 11; x++) {
                    level.setBlock(new BlockPos(x, 160, 6), Blocks.REDSTONE_WIRE.defaultBlockState(), Block.UPDATE_ALL);
                }
                for (int z = 4; z <= 8; z++) {
                    level.setBlock(new BlockPos(7, 160, z), Blocks.REDSTONE_WIRE.defaultBlockState(), Block.UPDATE_ALL);
                }
                // Step-up block at x=11..12
                level.setBlock(new BlockPos(12, 160, 6), Blocks.STONE.defaultBlockState(), Block.UPDATE_ALL);
                level.setBlock(new BlockPos(12, 161, 6), Blocks.REDSTONE_WIRE.defaultBlockState(), Block.UPDATE_ALL);
                level.setBlock(new BlockPos(13, 160, 6), Blocks.REDSTONE_BLOCK.defaultBlockState(), Block.UPDATE_ALL);
            }
            case "day", "night", "walk" -> {
                // Simple terrain + pillar + torch for day/night/walk comparison
                for (int y = 160; y <= 164; y++) {
                    level.setBlock(new BlockPos(8, y, 8), Blocks.OAK_LOG.defaultBlockState(), Block.UPDATE_ALL);
                    level.setBlock(new BlockPos(5, y, 10), Blocks.STONE_BRICKS.defaultBlockState(), Block.UPDATE_ALL);
                }
                level.setBlock(new BlockPos(7, 160, 8), Blocks.TORCH.defaultBlockState(), Block.UPDATE_ALL);
            }
            default -> {
            }
        }
        LOGGER.info("[ValkyrieSmoke] Built repro scene '{}' in section (0, 10, 0)", mode);
    }

    private void buildShapeShowcase(ServerLevel level) {
        // Build aerial stone platform in section (0, 10, 0): x=1..14, y=159, z=1..14
        for (int x = 1; x <= 14; x++) {
            for (int z = 1; z <= 14; z++) {
                level.setBlock(new BlockPos(x, 159, z), Blocks.STONE.defaultBlockState(), Block.UPDATE_ALL);
                for (int y = 160; y <= 165; y++) {
                    level.setBlock(new BlockPos(x, y, z), Blocks.AIR.defaultBlockState(), Block.UPDATE_ALL);
                }
            }
        }
        // Row 1 (z = 3): single fence, connected fences, glass pane, iron bars, door
        level.setBlock(new BlockPos(2, 160, 3), Blocks.OAK_FENCE.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(4, 160, 3), Blocks.OAK_FENCE.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(5, 160, 3), Blocks.OAK_FENCE.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(7, 160, 3), Blocks.GLASS_PANE.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(8, 160, 3), Blocks.GLASS_PANE.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(10, 160, 3), Blocks.IRON_BARS.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(11, 160, 3), Blocks.IRON_BARS.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(13, 160, 3),
                Blocks.OAK_DOOR.defaultBlockState().setValue(DoorBlock.FACING, Direction.NORTH).setValue(DoorBlock.HALF, DoubleBlockHalf.LOWER),
                Block.UPDATE_ALL);
        level.setBlock(new BlockPos(13, 161, 3),
                Blocks.OAK_DOOR.defaultBlockState().setValue(DoorBlock.FACING, Direction.NORTH).setValue(DoorBlock.HALF, DoubleBlockHalf.UPPER),
                Block.UPDATE_ALL);

        // Row 2 (z = 7): cobblestone wall (2 connected), glass block, oak stairs, oak slab
        level.setBlock(new BlockPos(3, 160, 7), Blocks.COBBLESTONE_WALL.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(4, 160, 7), Blocks.COBBLESTONE_WALL.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(6, 160, 7), Blocks.GLASS.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(new BlockPos(8, 160, 7),
                Blocks.OAK_STAIRS.defaultBlockState().setValue(StairBlock.FACING, Direction.NORTH),
                Block.UPDATE_ALL);
        level.setBlock(new BlockPos(10, 160, 7),
                Blocks.OAK_SLAB.defaultBlockState().setValue(SlabBlock.TYPE, SlabType.BOTTOM),
                Block.UPDATE_ALL);
        level.setBlock(new BlockPos(12, 160, 7),
                Blocks.OAK_SIGN.defaultBlockState().setValue(StandingSignBlock.ROTATION, 8),
                Block.UPDATE_ALL);
        if (level.getBlockEntity(new BlockPos(12, 160, 7)) instanceof SignBlockEntity signBe) {
            signBe.updateText(text -> text.setMessage(0, net.minecraft.network.chat.Component.literal("AerogelVK")), true);
            signBe.setChanged();
        }
        shapeShowcaseServerDone = true;
        LOGGER.info("[ValkyrieSmoke] Built shape showcase platform in section (0, 10, 0)");
    }

    private void logShapeShowcaseSectionMetrics(Minecraft mc) {
        if (mc.level == null) return;
        GreedyMesher.SectionSnapshot snap = GreedyMesher.SectionSnapshot.capture(mc.level, 0, 10, 0);
        if (snap == null) {
            LOGGER.warn("[ValkyrieSmoke] Shape showcase snapshot (0,10,0) was null");
            return;
        }
        GreedyMesher.MeshedSectionResult res = GreedyMesher.mesh(snap);
        int cubeOpaqueQuads = res.quadCount();
        int cubeTransQuads = res.translucentQuadCount();
        int shapeOpaqueQuads = res.halfScaleOpaqueQuadCount();
        int shapeTransQuads = res.halfScaleTranslucentQuadCount();
        int totalQuads = cubeOpaqueQuads + cubeTransQuads + shapeOpaqueQuads + shapeTransQuads;
        LOGGER.info("[ValkyrieSmoke] SHAPE_METRICS section=(0,10,0) cubeOpaqueQ={} cubeTransQ={} shapeOpaqueQ={} shapeTransQ={} totalQuads={} totalVerts={}",
                cubeOpaqueQuads, cubeTransQuads, shapeOpaqueQuads, shapeTransQuads, totalQuads, totalQuads * 4);
    }
}
