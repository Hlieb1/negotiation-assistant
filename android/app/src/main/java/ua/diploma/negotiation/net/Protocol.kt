package ua.diploma.negotiation.net

import org.json.JSONObject

enum class Mode(val wire: String, val label: String) {
    FUZZY("fuzzy", "А — нечітка логіка"),
    LLM("llm", "Б — мовна модель"),
    HYBRID("hybrid", "Гібрид"),
}

data class Hint(
    val ts: Long,
    val state: String,
    val confidence: Double,
    val reply: String,
    val tone: String,
    val mode: String,
    val latencyMs: Long,
)

sealed interface ServerMessage {
    data class HintMessage(val hint: Hint) : ServerMessage
    data class Status(val session: String, val mode: String, val sessionId: String) : ServerMessage
    data class Error(val detail: String) : ServerMessage
}

object Protocol {
    fun session(action: String, mode: Mode? = null): String = JSONObject()
        .put("type", "session")
        .put("action", action)
        .apply { if (mode != null) put("mode", mode.wire) }
        .toString()

    fun videoFrame(ts: Long, jpegBase64: String): String = JSONObject()
        .put("type", "video_frame")
        .put("ts", ts)
        .put("jpeg_b64", jpegBase64)
        .toString()

    fun audioChunk(ts: Long, pcmBase64: String, sampleRate: Int): String = JSONObject()
        .put("type", "audio_chunk")
        .put("ts", ts)
        .put("pcm16_b64", pcmBase64)
        .put("sample_rate", sampleRate)
        .toString()

    fun parse(text: String): ServerMessage? = runCatching {
        val json = JSONObject(text)
        when (json.getString("type")) {
            "hint" -> ServerMessage.HintMessage(
                Hint(
                    ts = json.getLong("ts"),
                    state = json.getString("state"),
                    confidence = json.getDouble("confidence"),
                    reply = json.getString("reply"),
                    tone = json.getString("tone"),
                    mode = json.getString("mode"),
                    latencyMs = json.getLong("latency_ms"),
                )
            )
            "status" -> ServerMessage.Status(
                json.getString("session"), json.getString("mode"), json.optString("session_id")
            )
            "error" -> ServerMessage.Error(json.getString("detail"))
            else -> null
        }
    }.getOrNull()
}
