package ua.diploma.negotiation.capture

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Base64
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

class FrameAnalyzer(
    private val sendFps: () -> Int,
    private val onFrame: (ts: Long, jpegBase64: String) -> Unit,
) : ImageAnalysis.Analyzer {

    private var lastSentMs = 0L

    override fun analyze(image: ImageProxy) {
        image.use {
            val now = System.currentTimeMillis()
            if (now - lastSentMs < 1000L / sendFps().coerceIn(1, 10)) return
            lastSentMs = now
            onFrame(now, encode(image.toBitmap(), image.imageInfo.rotationDegrees))
        }
    }

    private fun encode(source: Bitmap, rotationDegrees: Int): String {
        val longSide = maxOf(source.width, source.height).toFloat()
        val scale = (MAX_SIDE / longSide).coerceAtMost(1f)
        val matrix = Matrix().apply {
            postScale(scale, scale)
            postRotate(rotationDegrees.toFloat())
        }
        val frame = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        val out = ByteArrayOutputStream()
        frame.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
        if (frame !== source) frame.recycle()
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
    }

    private companion object {
        const val MAX_SIDE = 640f
        const val JPEG_QUALITY = 80
    }
}
