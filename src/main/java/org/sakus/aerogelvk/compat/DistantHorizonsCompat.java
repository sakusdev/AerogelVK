package org.sakus.aerogelvk.compat;

import org.sakus.aerogelvk.config.ValkyrieConfig;
import com.seibel.distanthorizons.api.DhApi;
import com.seibel.distanthorizons.api.interfaces.render.IDhApiRenderProxy;
import com.seibel.distanthorizons.api.methods.events.abstractEvents.DhApiAfterDhInitEvent;
import com.seibel.distanthorizons.api.methods.events.abstractEvents.DhApiBeforeApplyShaderRenderEvent;
import com.seibel.distanthorizons.api.methods.events.sharedParameterObjects.DhApiCancelableEventParam;
import com.seibel.distanthorizons.api.methods.events.sharedParameterObjects.DhApiEventParam;
import com.seibel.distanthorizons.api.methods.events.sharedParameterObjects.DhApiRenderParam;
import com.seibel.distanthorizons.api.objects.DhApiResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Integrates ValkyrieRender with Distant Horizons via the official DhApi.
 *
 * DH renders its LODs during Fabric's WorldRenderEvents.AFTER_SETUP (before Valkyrie's
 * terrain composite). We intercept DhApiBeforeApplyShaderRenderEvent so DH keeps its
 * LOD color/depth in its offscreen textures, and then composite those distant LODs behind
 * Valkyrie's near terrain depth inside VkInteropManager.
 */
public final class DistantHorizonsCompat {

    private static final Logger LOGGER = LoggerFactory.getLogger(DistantHorizonsCompat.class);

    private static volatile boolean dhInitialized = false;
    private static volatile boolean dhRenderedThisFrame = false;
    private static boolean registered = false;

    private DistantHorizonsCompat() {
    }

    public static synchronized void register() {
        if (registered) {
            return;
        }

        DhApi.events.bind(DhApiAfterDhInitEvent.class, new DhApiAfterDhInitEvent() {
            @Override
            public void afterDistantHorizonsInit(DhApiEventParam<Void> event) {
                dhInitialized = true;
                LOGGER.info("[Valkyrie] Distant Horizons DhApi initialized; deferred LOD compositing enabled.");
            }
        });

        DhApi.events.bind(DhApiBeforeApplyShaderRenderEvent.class, new DhApiBeforeApplyShaderRenderEvent() {
            @Override
            public void beforeRender(DhApiCancelableEventParam<DhApiRenderParam> event) {
                if (!ValkyrieConfig.get().isDhCompatEnabled()) {
                    return;
                }
                dhRenderedThisFrame = true;
                // Prevent DH from copying LODs into Minecraft's MainTarget before Valkyrie blits;
                // Valkyrie will composite DH's offscreen color/depth behind near terrain instead.
                event.cancelEvent();
            }
        });

        registered = true;
        LOGGER.info("[Valkyrie] Registered Distant Horizons DhApi event listeners.");
    }

    public static boolean isDhInitialized() {
        return dhInitialized;
    }

    /**
     * If DH rendered LODs for the current frame, populates outColorAndDepth[0..1] with
     * {dhColorGlTexId, dhDepthGlTexId} and resets the per-frame flag.
     */
    public static boolean consumeDhFrameTextures(int[] outColorAndDepth) {
        if (!dhRenderedThisFrame) {
            return false;
        }
        dhRenderedThisFrame = false;

        IDhApiRenderProxy proxy = DhApi.Delayed.renderProxy;
        if (proxy == null) {
            return false;
        }
        try {
            // getDh*TextureId() works across DH 2.x/3.0-3.2 and 3.3+ (where it defaults to *GlId())
            DhApiResult<Integer> colorRes = proxy.getDhColorTextureId();
            DhApiResult<Integer> depthRes = proxy.getDhDepthTextureId();
            if (colorRes != null && colorRes.success && colorRes.payload != null
                    && depthRes != null && depthRes.success && depthRes.payload != null
                    && colorRes.payload > 0 && depthRes.payload > 0) {
                outColorAndDepth[0] = colorRes.payload;
                outColorAndDepth[1] = depthRes.payload;
                return true;
            }
        } catch (Throwable ignored) {
            // DH render proxy may not be ready during early world load frames
        }
        return false;
    }
}
