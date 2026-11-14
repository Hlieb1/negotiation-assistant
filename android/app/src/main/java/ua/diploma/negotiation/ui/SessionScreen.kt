package ua.diploma.negotiation.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import ua.diploma.negotiation.net.ConnectionStatus
import ua.diploma.negotiation.net.Hint

private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)

private val STATE_COLORS = mapOf(
    "злість" to Color(0xFFE53935),
    "напруга" to Color(0xFFFB8C00),
    "сумнів" to Color(0xFFFDD835),
    "згода" to Color(0xFF43A047),
    "нейтральний" to Color(0xFF90A4AE),
)

@Composable
fun SessionScreen(state: UiState, viewModel: SessionViewModel) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
        })
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
        granted = result.values.all { it }
    }
    LaunchedEffect(Unit) { if (!granted) launcher.launch(REQUIRED_PERMISSIONS) }

    val view = LocalView.current
    DisposableEffect(state.running) {
        view.keepScreenOn = state.running
        onDispose { view.keepScreenOn = false }
    }

    var showConsent by remember { mutableStateOf(false) }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (granted) {
            CameraPreview(
                backCamera = state.settings.backCamera,
                sendFps = state.settings.sendFps,
                onFrame = viewModel::onFrame,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            PermissionRequest { launcher.launch(REQUIRED_PERMISSIONS) }
        }

        Column(Modifier.fillMaxSize().safeDrawingPadding().padding(12.dp)) {
            StatusBar(state, onSettings = viewModel::openSettings)
            Spacer(Modifier.weight(1f))
            HintCard(state.hint, state.running)
            Spacer(Modifier.height(12.dp))
            state.message?.let {
                Text(it, color = Color.White, style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.height(8.dp))
            }
            if (state.running) {
                Button(
                    onClick = viewModel::stop,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFC62828)),
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                ) { Text("Стоп", fontSize = 18.sp) }
            } else {
                Button(
                    onClick = { showConsent = true },
                    enabled = granted,
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                ) { Text("Старт", fontSize = 18.sp) }
            }
        }
    }

    if (showConsent) {
        ConsentDialog(
            onConfirm = { showConsent = false; viewModel.start() },
            onDismiss = { showConsent = false },
        )
    }
}

@Composable
private fun StatusBar(state: UiState, onSettings: () -> Unit) {
    Row(verticalAlignment = Alignment.Top) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            val connection = when {
                !state.running && state.connection == ConnectionStatus.DISCONNECTED -> "підключиться після «Старт»"
                else -> state.connection.label
            }
            val connected = state.connection == ConnectionStatus.CONNECTED
            Chip(if (connected) Color(0xFF2E7D32) else Color(0xCC37474F), "Сервер: $connection")
            Chip(Color(0xCC37474F), "Режим: ${state.settings.mode.label}")
            if (state.running && state.voiceProfile == false) {
                Chip(Color(0xCCE65100), "Зразок вашого голосу не записано: усі голоси вважаються співрозмовником")
            }
            if (state.running) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.clip(RoundedCornerShape(12.dp)).background(Color(0xCC000000)).padding(10.dp, 4.dp),
                ) {
                    Box(Modifier.size(10.dp).clip(CircleShape).background(Color.Red))
                    Spacer(Modifier.width(6.dp))
                    Text(
                        "Аналіз · кадрів ${state.framesSent} · аудіо ${state.audioChunksSent * 2} с",
                        color = Color.White, fontSize = 13.sp,
                    )
                }
            }
        }
        Spacer(Modifier.width(8.dp))
        IconButton(
            onClick = onSettings,
            modifier = Modifier.clip(CircleShape).background(Color(0xCC37474F)),
        ) {
            Icon(Icons.Filled.Settings, contentDescription = "Налаштування", tint = Color.White)
        }
    }
}

@Composable
private fun Chip(color: Color, text: String) {
    Text(
        text, color = Color.White, fontSize = 13.sp,
        modifier = Modifier.clip(RoundedCornerShape(12.dp)).background(color).padding(10.dp, 4.dp),
    )
}

@Composable
private fun HintCard(hint: Hint?, running: Boolean) {
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xE6101418))
            .padding(16.dp),
    ) {
        if (hint == null) {
            Text(
                if (running) "Аналізую співрозмовника…" else "Натисніть «Старт», щоб почати аналіз",
                color = Color.White, fontSize = 18.sp,
            )
            return@Column
        }
        val color = STATE_COLORS[hint.state] ?: Color.White
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(14.dp).clip(CircleShape).background(color))
            Spacer(Modifier.width(8.dp))
            Text(hint.state.replaceFirstChar { it.uppercase() }, color = color, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.weight(1f))
            Text("${hint.latencyMs} мс", color = Color(0xFF90A4AE), fontSize = 13.sp)
        }
        Spacer(Modifier.height(6.dp))
        LinearProgressIndicator(
            progress = { hint.confidence.toFloat().coerceIn(0f, 1f) },
            color = color,
            modifier = Modifier.fillMaxWidth(),
        )
        Text("Впевненість ${(hint.confidence * 100).toInt()} %", color = Color(0xFF90A4AE), fontSize = 12.sp)
        if (hint.heard.isNotBlank()) {
            Spacer(Modifier.height(10.dp))
            Text("Співрозмовник: «${hint.heard}»", color = Color(0xFFB0BEC5), fontSize = 15.sp)
        }
        Spacer(Modifier.height(10.dp))
        Text(hint.reply, color = Color.White, fontSize = 26.sp, lineHeight = 32.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(8.dp))
        Text("Тон: ${hint.tone}", color = Color(0xFFFFCC80), fontSize = 18.sp)
    }
}

@Composable
private fun PermissionRequest(onRequest: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            "Для роботи потрібен доступ до камери й мікрофона.",
            color = Color.White, fontSize = 18.sp,
        )
        Spacer(Modifier.height(16.dp))
        Button(onClick = onRequest) { Text("Надати доступ") }
    }
}

@Composable
private fun ConsentDialog(onConfirm: () -> Unit, onDismiss: () -> Unit) {
    var informed by remember { mutableStateOf(false) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Попередьте співрозмовника") },
        text = {
            Column {
                Text(
                    "Застосунок знімає обличчя і записує голос співрозмовника, щоб оцінити його емоційний стан. " +
                        "Перед початком повідомте співрозмовника про аналіз і отримайте його згоду. " +
                        "Відео і звук не зберігаються, на ноутбук передаються лише для аналізу в реальному часі."
                )
                Spacer(Modifier.height(12.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.clickable { informed = !informed },
                ) {
                    Checkbox(checked = informed, onCheckedChange = { informed = it })
                    Text("Співрозмовника попереджено, згоду отримано")
                }
            }
        },
        confirmButton = { TextButton(onClick = onConfirm, enabled = informed) { Text("Почати") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Скасувати") } },
    )
}
