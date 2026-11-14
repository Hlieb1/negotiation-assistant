package ua.diploma.negotiation.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import ua.diploma.negotiation.net.Mode

data class AppSettings(
    val host: String = "192.168.0.100",
    val port: Int = 8000,
    val mode: Mode = Mode.FUZZY,
    val backCamera: Boolean = true,
    val sendFps: Int = 4,
) {
    val url: String get() = "ws://${host.trim()}:$port/ws"
}

private val Context.dataStore by preferencesDataStore(name = "settings")

class SettingsStore(private val context: Context) {
    val settings: Flow<AppSettings> = context.dataStore.data.map { prefs ->
        val defaults = AppSettings()
        AppSettings(
            host = prefs[HOST] ?: defaults.host,
            port = prefs[PORT] ?: defaults.port,
            mode = Mode.entries.firstOrNull { it.wire == prefs[MODE] } ?: defaults.mode,
            backCamera = prefs[BACK_CAMERA] ?: defaults.backCamera,
            sendFps = prefs[SEND_FPS] ?: defaults.sendFps,
        )
    }

    suspend fun save(settings: AppSettings) {
        context.dataStore.edit { prefs ->
            prefs[HOST] = settings.host.trim()
            prefs[PORT] = settings.port
            prefs[MODE] = settings.mode.wire
            prefs[BACK_CAMERA] = settings.backCamera
            prefs[SEND_FPS] = settings.sendFps
        }
    }

    private companion object {
        val HOST = stringPreferencesKey("host")
        val PORT = intPreferencesKey("port")
        val MODE = stringPreferencesKey("mode")
        val BACK_CAMERA = booleanPreferencesKey("back_camera")
        val SEND_FPS = intPreferencesKey("send_fps")
    }
}
