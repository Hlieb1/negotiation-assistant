package ua.diploma.negotiation.capture

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AudioStreamer(private val onChunk: (ts: Long, pcm16le: ByteArray) -> Unit) {
    private var job: Job? = null

    @SuppressLint("MissingPermission")
    fun start(scope: CoroutineScope, onError: (String) -> Unit) {
        if (job != null) return
        job = scope.launch(Dispatchers.IO) {
            val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
            val record = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE, CHANNEL, ENCODING,
                maxOf(minBuffer, SAMPLE_RATE * 2)
            )
            if (record.state != AudioRecord.STATE_INITIALIZED) {
                record.release()
                onError("Мікрофон недоступний")
                return@launch
            }
            val chunk = ShortArray(SAMPLE_RATE * CHUNK_SECONDS)
            record.startRecording()
            try {
                while (isActive) {
                    var filled = 0
                    var startTs = 0L
                    while (filled < chunk.size && isActive) {
                        val read = record.read(chunk, filled, chunk.size - filled)
                        if (read < 0) {
                            onError("Помилка запису звуку ($read)")
                            return@launch
                        }
                        if (filled == 0 && read > 0) {
                            startTs = System.currentTimeMillis() - read * 1000L / SAMPLE_RATE
                        }
                        filled += read
                    }
                    if (filled == chunk.size) onChunk(startTs, toLittleEndian(chunk))
                }
            } finally {
                record.stop()
                record.release()
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }

    private fun toLittleEndian(samples: ShortArray): ByteArray {
        val buffer = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asShortBuffer().put(samples)
        return buffer.array()
    }

    companion object {
        const val SAMPLE_RATE = 16000
        const val CHUNK_SECONDS = 2
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
    }
}
