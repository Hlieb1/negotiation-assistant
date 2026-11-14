package ua.diploma.negotiation.ui

import android.app.Application
import android.util.Base64
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import ua.diploma.negotiation.capture.AudioStreamer
import ua.diploma.negotiation.data.AppSettings
import ua.diploma.negotiation.data.SettingsStore
import ua.diploma.negotiation.net.ConnectionStatus
import ua.diploma.negotiation.net.Hint
import ua.diploma.negotiation.net.NegotiationClient
import ua.diploma.negotiation.net.Protocol
import ua.diploma.negotiation.net.ServerMessage

enum class Screen { SETTINGS, SESSION }

sealed interface Enrollment {
    data object Idle : Enrollment
    data class Recording(val secondsLeft: Int) : Enrollment
    data object Sending : Enrollment
    data class Done(val ok: Boolean, val message: String) : Enrollment
}

data class UiState(
    val screen: Screen = Screen.SETTINGS,
    val settings: AppSettings = AppSettings(),
    val connection: ConnectionStatus = ConnectionStatus.DISCONNECTED,
    val running: Boolean = false,
    val hint: Hint? = null,
    val message: String? = null,
    val framesSent: Int = 0,
    val audioChunksSent: Int = 0,
    val voiceProfile: Boolean? = null,
    val enrollment: Enrollment = Enrollment.Idle,
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
            val current = _state.value
            if (current.running) client.send(Protocol.session("start", current.settings.mode, current.settings.context))
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

    fun enrollVoice(settings: AppSettings) {
        if (_state.value.running || _state.value.enrollment is Enrollment.Recording ||
            _state.value.enrollment == Enrollment.Sending) return
        saveSettings(settings)
        viewModelScope.launch {
            val recording = async(Dispatchers.IO) { AudioStreamer.recordOnce(ENROLL_SECONDS) }
            for (left in ENROLL_SECONDS downTo 1) {
                _state.update { it.copy(enrollment = Enrollment.Recording(left)) }
                delay(1000)
            }
            val pcm = recording.await()
            if (pcm == null) {
                _state.update { it.copy(enrollment = Enrollment.Done(false, "Мікрофон недоступний")) }
                return@launch
            }
            _state.update { it.copy(enrollment = Enrollment.Sending) }
            client.connect(settings.url)
            val connected = withTimeoutOrNull(CONNECT_TIMEOUT_MS) {
                client.status.first { it == ConnectionStatus.CONNECTED }
            }
            if (connected == null) {
                client.disconnect()
                _state.update { it.copy(enrollment = Enrollment.Done(false, "Сервер недоступний: ${settings.url}")) }
                return@launch
            }
            val reply = async(start = CoroutineStart.UNDISPATCHED) {
                withTimeoutOrNull(REPLY_TIMEOUT_MS) {
                    client.messages.filterIsInstance<ServerMessage.Enrolled>().first()
                }
            }
            client.send(
                Protocol.enroll(System.currentTimeMillis(), Base64.encodeToString(pcm, Base64.NO_WRAP),
                    AudioStreamer.SAMPLE_RATE)
            )
            val result = reply.await()
            client.disconnect()
            val done = when {
                result == null -> Enrollment.Done(false, "Сервер не відповів")
                result.ok -> Enrollment.Done(true, "Зразок голосу збережено (${result.speechSeconds} с мови)")
                else -> Enrollment.Done(false, "Не вдалося: ${result.detail}")
            }
            _state.update { it.copy(enrollment = done, voiceProfile = if (result?.ok == true) true else it.voiceProfile) }
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
                it.copy(
                    message = if (message.session == "started") "Сесію розпочато" else "Сесію завершено",
                    voiceProfile = message.voiceProfile,
                )
            }
            is ServerMessage.Enrolled -> Unit
            is ServerMessage.Error -> _state.update { it.copy(message = "Помилка сервера: ${message.detail}") }
        }
    }

    override fun onCleared() {
        audio.stop()
        client.disconnect()
    }

    companion object {
        const val ENROLL_SECONDS = 12
        private const val MAX_QUEUED_BYTES = 512 * 1024L
        private const val CONNECT_TIMEOUT_MS = 6000L
        private const val REPLY_TIMEOUT_MS = 20000L
    }
}
