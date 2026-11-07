# Протокол обміну

Транспорт — WebSocket `ws://<IP ноутбука>:8000/ws`, кожне повідомлення — один текстовий JSON-кадр.
Усі часові мітки `ts` — Unix-час у мілісекундах.

## Клієнт → сервер

| Тип | Поля | Примітка |
|---|---|---|
| `session` | `action`: `start` \| `stop`, `mode`: `fuzzy` \| `llm` \| `hybrid` | `mode` необов'язковий, типово — `session.default_mode` з `config.yaml` |
| `video_frame` | `ts`, `jpeg_b64` | `ts` — момент зйомки кадру; JPEG 640×480, base64 |
| `audio_chunk` | `ts`, `pcm16_b64`, `sample_rate` | `ts` — початок фрагмента; PCM 16 біт, моно, little-endian, base64; типово 16000 Гц |

Кадри й аудіо до `session/start` сервер ігнорує.

## Сервер → клієнт

| Тип | Поля |
|---|---|
| `status` | `ts`, `session`: `started` \| `stopped`, `mode`, `session_id` |
| `hint` | `ts`, `state`, `confidence` (0–1), `reply`, `tone`, `mode`, `latency_ms` |
| `error` | `ts`, `detail` |

`latency_ms` рахується на сервері від моменту отримання останніх даних, що вплинули на підказку,
до моменту надсилання підказки, тому не залежить від розбіжності годинників телефона й ноутбука.

## Приклад

```json
{"type": "session", "action": "start", "mode": "fuzzy"}
{"type": "video_frame", "ts": 1794040623646, "jpeg_b64": "/9j/4AAQ..."}
{"type": "audio_chunk", "ts": 1794040621646, "pcm16_b64": "AAABAP7/...", "sample_rate": 16000}
{"type": "hint", "ts": 1794040624012, "state": "сумнів", "confidence": 0.62,
 "reply": "...", "tone": "спокійний, впевнений", "mode": "fuzzy", "latency_ms": 366}
```
