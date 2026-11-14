import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from .protocol import Mode

SERVER_DIR = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else SERVER_DIR / path


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class SessionSettings(BaseModel):
    default_mode: Mode = "fuzzy"


class VideoSettings(BaseModel):
    enabled: bool = True
    detector_backend: str = "opencv"
    classifier: str = "hsemotion"
    hsemotion_model: str = "enet_b2_7"
    face_margin: float = 0.1
    max_fps: float = 4.0
    max_width: int = 640
    equalize: bool = False
    min_face_size: int = 40
    smoothing_window_s: float = 1.5
    min_face_ratio: float = 0.3


class AudioSettings(BaseModel):
    enabled: bool = True
    denoise: bool = False
    speech_padding_s: float = 0.25
    vad_aggressiveness: int = 2
    vad_frame_ms: int = 30
    min_rms_db: float = -55.0
    target_level_db: float | None = -20.0
    max_gain_db: float = 30.0
    noise_margin_db: float = 6.0
    voice_model: str = "emotion2vec/emotion2vec_plus_large"
    voice_device: str = "cuda:0"
    min_speech_s: float = 0.5
    smoothing_window_s: float = 6.0


class SpeakerSettings(BaseModel):
    enabled: bool = True
    source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: Path = Path("models/spkrec-ecapa-voxceleb")
    profile_path: Path = Path("profiles/owner.npy")
    device: str = "cuda:0"
    threshold: float = 0.35
    min_speech_s: float = 0.8
    min_enroll_s: float = 6.0


class AsrSettings(BaseModel):
    enabled: bool = True
    model: str = "large-v3-turbo"
    language: str = "uk"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    context_prompt_chars: int = 200
    beam_size: int = 1
    min_pause_s: float = 0.6
    max_utterance_s: float = 10.0


class FusionSettings(BaseModel):
    emotion_weights: dict[str, float] = {"face": 0.65, "voice": 0.35}
    valence_weights: dict[str, float] = {"face": 0.4, "voice": 0.6}
    prosody_arousal_weight: float = 0.5
    max_age_s: float = 6.0
    state_switch_margin: float = 0.05
    baseline_history: int = 30
    baseline_min_samples: int = 3
    rules_path: Path = Path("rules/state_rules.yaml")


class LLMSettings(BaseModel):
    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    prompt_path: Path = Path("app/decision/llm_prompt_uk.json")
    timeout_s: float = 4.5
    temperature: float = 0.3
    max_tokens: int = 100
    keep_alive: str = "60m"


class DecisionSettings(BaseModel):
    fuzzy_rules_path: Path = Path("rules/fuzzy_rules.txt")
    replies_path: Path = Path("app/decision/replies_uk.json")
    min_interval_s: float = 4.0
    stable_s: float = 1.0
    min_confidence: float = 0.1
    history_size: int = 5
    speech_min_interval_s: float = 2.0
    state_hint_modes: list[str] = ["fuzzy"]
    dialogue_turns: int = 6
    llm: LLMSettings = LLMSettings()


class StorageSettings(BaseModel):
    log_dir: Path = Path("logs")
    record_media: bool = False
    recordings_dir: Path = Path("recordings")


class Settings(BaseModel):
    server: ServerSettings = ServerSettings()
    session: SessionSettings = SessionSettings()
    video: VideoSettings = VideoSettings()
    audio: AudioSettings = AudioSettings()
    asr: AsrSettings = AsrSettings()
    speaker: SpeakerSettings = SpeakerSettings()
    fusion: FusionSettings = FusionSettings()
    decision: DecisionSettings = DecisionSettings()
    storage: StorageSettings = StorageSettings()


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path or os.environ.get("NEGOTIATION_CONFIG") or SERVER_DIR / "config.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(data)
