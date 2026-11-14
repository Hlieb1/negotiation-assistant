package ua.diploma.negotiation.ui

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
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import ua.diploma.negotiation.data.AppSettings
import ua.diploma.negotiation.net.Mode

@Composable
fun SettingsScreen(settings: AppSettings, onSave: (AppSettings) -> Unit, onContinue: () -> Unit) {
    var host by remember(settings) { mutableStateOf(settings.host) }
    var port by remember(settings) { mutableStateOf(settings.port.toString()) }
    var mode by remember(settings) { mutableStateOf(settings.mode) }
    var backCamera by remember(settings) { mutableStateOf(settings.backCamera) }
    var sendFps by remember(settings) { mutableStateOf(settings.sendFps.toFloat()) }
    val portValue = port.toIntOrNull()
    val valid = host.isNotBlank() && portValue != null && portValue in 1..65535

    Column(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Налаштування", style = MaterialTheme.typography.headlineMedium)
        Text(
            "Ноутбук із сервером і телефон мають бути в одній мережі Wi-Fi.",
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
        Mode.entries.forEach { option ->
            Choice(option.label, selected = mode == option) { mode = option }
        }

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
                onSave(AppSettings(host.trim(), portValue ?: 8000, mode, backCamera, sendFps.toInt()))
                onContinue()
            },
        ) { Text("Зберегти і перейти до сесії") }
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
