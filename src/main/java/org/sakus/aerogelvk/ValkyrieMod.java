package org.sakus.aerogelvk;

import org.sakus.aerogelvk.command.ValkyrieCommand;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Environment(EnvType.CLIENT)
public class ValkyrieMod implements ClientModInitializer {
    public static final String MOD_ID = "aerogelvk";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitializeClient() {
        LOGGER.info("[Valkyrie] Initializing Valkyrie Vulkan Deferred Renderer for Minecraft 1.21.1...");
        org.sakus.aerogelvk.compat.CompatManager.init();
        ValkyrieCommand.register();

        // Register JVM / lifecycle shutdown hook to clean up ValkyrieRenderer
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            LOGGER.info("[Valkyrie] JVM Shutdown Hook triggered, cleaning up ValkyrieRenderer...");
            try {
                ValkyrieRenderer.getInstance().shutdown();
            } catch (Throwable t) {
                LOGGER.error("[Valkyrie] Failed to cleanly shut down ValkyrieRenderer", t);
            }
        }, "Valkyrie-Shutdown-Hook"));

        LOGGER.info("[Valkyrie] ValkyrieMod client initialization complete.");
    }
}
