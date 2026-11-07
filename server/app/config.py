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
    hint_min_interval_s: float = 4.0


class VideoSettings(BaseModel):
    enabled: bool = True
    detector_backend: str = "opencv"
    align: bool = True
    max_fps: float = 4.0
    max_width: int = 640
    equalize: bool = False
    min_face_size: int = 40
    smoothing_window_s: float = 1.5
    min_face_ratio: float = 0.3


class StorageSettings(BaseModel):
    log_dir: Path = Path("logs")
    record_media: bool = False
    recordings_dir: Path = Path("recordings")


class Settings(BaseModel):
    server: ServerSettings = ServerSettings()
    session: SessionSettings = SessionSettings()
    video: VideoSettings = VideoSettings()
    storage: StorageSettings = StorageSettings()


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path or os.environ.get("NEGOTIATION_CONFIG") or SERVER_DIR / "config.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(data)
