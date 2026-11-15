package org.sakus.aerogelvk.config;

import com.terraformersmc.modmenu.api.ConfigScreenFactory;
import com.terraformersmc.modmenu.api.ModMenuApi;

/**
 * ModMenu integration entrypoint for AerogelVK.
 */
public final class AerogelVkModMenuIntegration implements ModMenuApi {

    @Override
    public ConfigScreenFactory<?> getModConfigScreenFactory() {
        return AerogelVkConfigScreen::create;
    }
}
