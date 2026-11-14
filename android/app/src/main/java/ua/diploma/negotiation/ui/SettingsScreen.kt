package ua.diploma.negotiation.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import ua.diploma.negotiation.data.AppSettings
import ua.diploma.negotiation.net.Mode

private const val ENROLL_TEXT = "Добрий день. Я хочу обговорити умови нашої співпраці: терміни, вартість " +
    "і порядок оплати. Мені важливо, щоб домовленість була вигідною для обох сторін, тому пропоную " +
    "спокійно пройтися по кожному пункту."

@Composable
fun SettingsScreen(
    settings: AppSettings,
    enrollment: Enrollment,
    onSave: (AppSettings) -> Unit,
    onContinue: () -> Unit,
    onEnroll: (AppSettings) -> Unit,
) {
    val androidContext = LocalContext.current
    var host by remember(settings) { mutableStateOf(settings.host) }
    var port by remember(settings) { mutableStateOf(settings.port.toString()) }
    var mode by remember(settings) { mutableStateOf(settings.mode) }
    var backCamera by remember(settings) { mutableStateOf(settings.backCamera) }
    var sendFps by remember(settings) { mutableStateOf(settings.sendFps.toFloat()) }
    var context by remember(settings) { mutableStateOf(settings.context) }
    val portValue = port.toIntOrNull()
    val valid = host.isNotBlank() && portValue != null && portValue in 1..65535
    val current = { AppSettings(host.trim(), portValue ?: 8000, mode, backCamera, sendFps.toInt(), context.trim()) }

    val micLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) onEnroll(current())
    }
    val startEnroll = {
        val granted = ContextCompat.checkSelfPermission(androidContext, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) onEnroll(current()) else micLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        Text(
            "Налаштування",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(start = 20.dp, end = 20.dp, top = 16.dp, bottom = 4.dp),
        )
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("Контекст переговорів", style = MaterialTheme.typography.titleMedium)
            OutlinedTextField(
                value = context, onValueChange = { if (it.length <= 2000) context = it },
                placeholder = {
                    Text("Тема, ваша мета і межі. Напр.: продаю CRM, ціна 60 000 грн/рік, знижка до 10 %, мета — договір цього тижня")
                },
                minLines = 3, maxLines = 8,
                modifier = Modifier.fillMaxWidth(),
            )

            Text("Зразок мого голосу", style = MaterialTheme.typography.titleMedium)
            EnrollmentCard(enrollment, enabled = valid, onRecord = startEnroll)

            Text("Сервер", style = MaterialTheme.typography.titleMedium)
            Text(
                "Ноутбук із сервером і телефон мають бути в одній мережі Wi-Fi. " +
                    "Якщо телефон підключено до ноутбука через adb reverse, вкажіть 127.0.0.1.",
                style = MaterialTheme.typography.bodyMedium,
            )
            OutlinedTextField(
                value = host, onValueChange = { host = it }, singleLine = true,
                label = { Text("IP-адреса сервера") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                modifier = Modifier.fillMaxWidth(),
            )
            OutlinedTextField(
                value = port, onValueChange = { port = it.filter(Char::isDigit) }, singleLine = true,
                label = { Text("Порт") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth(),
            )

            Text("Модуль рішень", style = MaterialTheme.typography.titleMedium)
            Mode.entries.forEach { option -> Choice(option.label, selected = mode == option) { mode = option } }

            Text("Камера", style = MaterialTheme.typography.titleMedium)
            Choice("Основна (на співрозмовника)", selected = backCamera) { backCamera = true }
            Choice("Фронтальна", selected = !backCamera) { backCamera = false }

            Text("Кадрів на сервер за секунду: ${sendFps.toInt()}", style = MaterialTheme.typography.titleMedium)
            Slider(value = sendFps, onValueChange = { sendFps = it }, valueRange = 3f..5f, steps = 1)

            Spacer(Modifier.height(8.dp))
            Button(
                enabled = valid,
                modifier = Modifier.fillMaxWidth(),
                onClick = {
                    onSave(current())
                    onContinue()
                },
            ) { Text("Зберегти і перейти до сесії") }
        }
    }
}

@Composable
private fun EnrollmentCard(enrollment: Enrollment, enabled: Boolean, onRecord: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                "Щоб система відрізняла ваш голос від голосу співрозмовника, один раз запишіть " +
                    "${SessionViewModel.ENROLL_SECONDS} с свого голосу: натисніть кнопку і прочитайте текст уголос.",
                style = MaterialTheme.typography.bodyMedium,
            )
            when (enrollment) {
                is Enrollment.Recording -> {
                    Text("Читайте вголос (${enrollment.secondsLeft} с):", style = MaterialTheme.typography.titleSmall)
                    Text("«$ENROLL_TEXT»", style = MaterialTheme.typography.bodyLarge)
                }
                Enrollment.Sending -> Text("Надсилаю зразок на сервер…")
                is Enrollment.Done -> Text(
                    enrollment.message,
                    color = if (enrollment.ok) Color(0xFF81C784) else Color(0xFFE57373),
                )
                Enrollment.Idle -> Unit
            }
            val busy = enrollment is Enrollment.Recording || enrollment == Enrollment.Sending
            OutlinedButton(onClick = onRecord, enabled = enabled && !busy, modifier = Modifier.fillMaxWidth()) {
                Text(if (enrollment is Enrollment.Done && enrollment.ok) "Записати зразок ще раз" else "Записати зразок голосу")
            }
        }
    }
}

@Composable
private fun Choice(label: String, selected: Boolean, onSelect: () -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .selectable(selected = selected, onClick = onSelect),
    ) {
        RadioButton(selected = selected, onClick = onSelect)
        Text(label)
    }
}
