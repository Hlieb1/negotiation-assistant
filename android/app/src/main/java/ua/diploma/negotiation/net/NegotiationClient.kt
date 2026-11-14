package ua.diploma.negotiation.net

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

enum class ConnectionStatus(val label: String) {
    DISCONNECTED("не підключено"),
    CONNECTING("підключення…"),
    CONNECTED("підключено"),
    RECONNECTING("перепідключення…"),
}

class NegotiationClient(private val scope: CoroutineScope) {
    private val http = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .pingInterval(10, TimeUnit.SECONDS)
        .build()

    private val _status = MutableStateFlow(ConnectionStatus.DISCONNECTED)
    val status: StateFlow<ConnectionStatus> = _status

    private val _messages = MutableSharedFlow<ServerMessage>(extraBufferCapacity = 32)
    val messages: SharedFlow<ServerMessage> = _messages

    var onConnected: (() -> Unit)? = null

    @Volatile private var socket: WebSocket? = null
    @Volatile private var wanted = false
    private var url: String = ""
    private var attempt = 0
    private var reconnectJob: Job? = null

    val queuedBytes: Long get() = socket?.queueSize() ?: 0L

    @Synchronized
    fun connect(url: String) {
        this.url = url
        wanted = true
        attempt = 0
        open()
    }

    @Synchronized
    fun disconnect() {
        wanted = false
        reconnectJob?.cancel()
        socket?.close(NORMAL_CLOSURE, null)
        socket = null
        _status.value = ConnectionStatus.DISCONNECTED
    }

    fun send(text: String): Boolean =
        _status.value == ConnectionStatus.CONNECTED && socket?.send(text) == true

    private fun open() {
        _status.value = if (attempt == 0) ConnectionStatus.CONNECTING else ConnectionStatus.RECONNECTING
        socket = http.newWebSocket(Request.Builder().url(url).build(), Listener())
    }

    private fun scheduleReconnect(failed: WebSocket) {
        synchronized(this) {
            if (!wanted || socket !== failed) return
            socket = null
            _status.value = ConnectionStatus.RECONNECTING
            val delayMs = (1000L shl attempt.coerceAtMost(3)).coerceAtMost(MAX_BACKOFF_MS)
            attempt++
            reconnectJob = scope.launch {
                delay(delayMs)
                synchronized(this@NegotiationClient) { if (wanted && socket == null) open() }
            }
        }
    }

    private inner class Listener : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            if (webSocket !== socket) return
            attempt = 0
            _status.value = ConnectionStatus.CONNECTED
            onConnected?.invoke()
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            Protocol.parse(text)?.let { _messages.tryEmit(it) }
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            scheduleReconnect(webSocket)
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            scheduleReconnect(webSocket)
        }
    }

    private companion object {
        const val NORMAL_CLOSURE = 1000
        const val MAX_BACKOFF_MS = 8000L
    }
}
