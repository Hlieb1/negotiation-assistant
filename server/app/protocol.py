import time
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

Mode = Literal["fuzzy", "llm", "hybrid"]


def now_ms() -> int:
    return time.time_ns() // 1_000_000


class VideoFrame(BaseModel):
    type: Literal["video_frame"] = "video_frame"
    ts: int
    jpeg_b64: str


class AudioChunk(BaseModel):
    type: Literal["audio_chunk"] = "audio_chunk"
    ts: int
    pcm16_b64: str
    sample_rate: int = Field(default=16000, gt=0)


class SessionControl(BaseModel):
    type: Literal["session"] = "session"
    action: Literal["start", "stop"]
    mode: Mode | None = None


ClientMessage = Annotated[Union[VideoFrame, AudioChunk, SessionControl], Field(discriminator="type")]
_client_message = TypeAdapter(ClientMessage)


def parse_client_message(raw: str | bytes) -> VideoFrame | AudioChunk | SessionControl:
    return _client_message.validate_json(raw)


class Hint(BaseModel):
    type: Literal["hint"] = "hint"
    ts: int
    state: str
    confidence: float = Field(ge=0.0, le=1.0)
    reply: str
    tone: str
    mode: Mode
    latency_ms: int


class Status(BaseModel):
    type: Literal["status"] = "status"
    ts: int
    session: Literal["started", "stopped"]
    mode: Mode
    session_id: str


class Error(BaseModel):
    type: Literal["error"] = "error"
    ts: int
    detail: str
