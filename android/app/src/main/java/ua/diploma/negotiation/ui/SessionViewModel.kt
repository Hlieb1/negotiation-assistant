package ua.diploma.negotiation.ui

import android.app.Application
import android.util.Base64
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import ua.diploma.negotiation.capture.AudioStreamer
import ua.diploma.negotiation.data.AppSettings
import ua.diploma.negotiation.data.SettingsStore
import ua.diploma.negotiation.net.ConnectionStatus
import ua.diploma.negotiation.net.Hint
import ua.diploma.negotiation.net.NegotiationClient
import ua.diploma.negotiation.net.Protocol
import ua.diploma.negotiation.net.ServerMessage

enum class Screen { SETTINGS, SESSION }

data class UiState(
    val screen: Screen = Screen.SETTINGS,
    val settings: AppSettings = AppSettings(),
    val connection: ConnectionStatus = ConnectionStatus.DISCONNECTED,
    val running: Boolean = false,
    val hint: Hint? = null,
    val message: String? = null,
    val framesSent: Int = 0,
    val audioChunksSent: Int = 0,
)

class SessionViewModel(app: Application) : AndroidViewModel(app) {
    private val store = SettingsStore(app)
    private val client = NegotiationClient(viewModelScope)
    private val audio = AudioStreamer(::sendAudio)

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state

    init {
        viewModelScope.launch {
            val saved = store.settings.first()
            _state.update { it.copy(settings = saved) }
        }
        viewModelScope.launch { client.status.collect { status -> _state.update { it.copy(connection = status) } } }
        viewModelScope.launch { client.messages.collect(::onMessage) }
        client.onConnected = {
            if (_state.value.running) client.send(Protocol.session("start", _state.value.settings.mode))
        }
    }

    fun saveSettings(settings: AppSettings) {
        _state.update { it.copy(settings = settings) }
        viewModelScope.launch { store.save(settings) }
    }

    fun openSession() = _state.update { it.copy(screen = Screen.SESSION, message = null) }

    fun openSettings() {
        stop()
        _state.update { it.copy(screen = Screen.SETTINGS) }
    }

    fun start() {
        if (_state.value.running) return
        _state.update { it.copy(running = true, hint = null, message = null, framesSent = 0, audioChunksSent = 0) }
        client.connect(_state.value.settings.url)
        audio.start(viewModelScope) { error -> _state.update { it.copy(message = error) } }
    }

    fun stop() {
        if (!_state.value.running) return
        client.send(Protocol.session("stop"))
        audio.stop()
        _state.update { it.copy(running = false) }
        viewModelScope.launch {
            delay(300)
            if (!_state.value.running) client.disconnect()
        }
    }

    fun onFrame(ts: Long, jpegBase64: String) {
        if (!_state.value.running || client.queuedBytes > MAX_QUEUED_BYTES) return
        if (client.send(Protocol.videoFrame(ts, jpegBase64))) {
            _state.update { it.copy(framesSent = it.framesSent + 1) }
        }
    }

    private fun sendAudio(ts: Long, pcm: ByteArray) {
        if (!_state.value.running) return
        val payload = Protocol.audioChunk(ts, Base64.encodeToString(pcm, Base64.NO_WRAP), AudioStreamer.SAMPLE_RATE)
        if (client.send(payload)) _state.update { it.copy(audioChunksSent = it.audioChunksSent + 1) }
    }

    private fun onMessage(message: ServerMessage) {
        when (message) {
            is ServerMessage.HintMessage -> _state.update { it.copy(hint = message.hint) }
            is ServerMessage.Status -> _state.update {
                it.copy(message = if (message.session == "started") "Сесію розпочато" else "Сесію завершено")
            }
            is ServerMessage.Error -> _state.update { it.copy(message = "Помилка сервера: ${message.detail}") }
        }
    }

    override fun onCleared() {
        audio.stop()
        client.disconnect()
    }

    private companion object {
        const val MAX_QUEUED_BYTES = 512 * 1024L
    }
}
