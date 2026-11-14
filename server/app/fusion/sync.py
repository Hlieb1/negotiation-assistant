from dataclasses import dataclass, field

from ..video.smoothing import SmoothedEmotion


@dataclass
class Observation:
    ts: int
    face: SmoothedEmotion | None = None
    voice: SmoothedEmotion | None = None
    prosody: dict = field(default_factory=dict)
    transcript: str = ""

    @property
    def modalities(self) -> list[str]:
        return [name for name in ("face", "voice") if getattr(self, name) is not None]


def synchronize(face: SmoothedEmotion | None, voice: SmoothedEmotion | None, prosody: dict, prosody_ts: int | None,
                transcript: str, max_age_s: float) -> Observation | None:
    stamps = [s.ts for s in (face, voice) if s is not None]
    if prosody_ts is not None:
        stamps.append(prosody_ts)
    if not stamps:
        return None
    reference = max(stamps)
    oldest = reference - int(max_age_s * 1000)
    return Observation(
        ts=reference,
        face=face if face is not None and face.ts >= oldest else None,
        voice=voice if voice is not None and voice.ts >= oldest else None,
        prosody=prosody if prosody_ts is not None and prosody_ts >= oldest else {},
        transcript=transcript,
    )
