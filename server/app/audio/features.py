import librosa
import numpy as np

from .preprocessing import SAMPLE_RATE, SpeechSegment

HOP = 256
MIN_PAUSE_S = 0.2


def _semitone_std(f0: np.ndarray) -> float:
    return float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 1 else 0.0


def pause_stats(segment: SpeechSegment) -> tuple[float, int]:
    mask = segment.mask
    if not mask.any():
        return 1.0, 0
    speech = np.flatnonzero(mask)
    inner = mask[speech[0]:speech[-1] + 1]
    frame_s = segment.frame_len / SAMPLE_RATE
    pauses, run = 0, 0
    for voiced in inner:
        if voiced:
            pauses += run * frame_s >= MIN_PAUSE_S
            run = 0
        else:
            run += 1
    return float(1.0 - inner.mean()), int(pauses)


def voice_features(segment: SpeechSegment) -> dict[str, float]:
    speech = segment.speech_audio()
    speech_s = len(speech) / SAMPLE_RATE
    pause_ratio, pauses = pause_stats(segment)
    features = {
        "speech_s": round(speech_s, 2),
        "gain_db": round(segment.gain_db, 1),
        "speech_ratio": round(segment.speech_ratio, 3),
        "pause_ratio": round(pause_ratio, 3),
        "pauses_per_min": round(pauses / (len(segment.audio) / SAMPLE_RATE) * 60, 1),
    }
    if speech_s < 0.3:
        return features

    f0, voiced, _ = librosa.pyin(speech, fmin=65, fmax=400, sr=SAMPLE_RATE, frame_length=1024, hop_length=HOP)
    f0 = f0[voiced & np.isfinite(f0)]
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=speech, hop_length=HOP)[0], ref=1.0)
    onsets = librosa.onset.onset_detect(y=speech, sr=SAMPLE_RATE, hop_length=HOP, units="time")
    mfcc = librosa.feature.mfcc(y=speech, sr=SAMPLE_RATE, n_mfcc=13, hop_length=HOP)

    features.update({
        "f0_mean_hz": round(float(np.mean(f0)), 1) if len(f0) else 0.0,
        "f0_std_st": round(_semitone_std(f0), 2) if len(f0) else 0.0,
        "f0_range_st": round(float(12 * np.log2(np.percentile(f0, 95) / np.percentile(f0, 5))), 2) if len(f0) > 1 else 0.0,
        "voiced_ratio": round(float(voiced.mean()), 3),
        "loudness_db": round(float(np.mean(rms_db)) - segment.gain_db, 1),
        "loudness_std_db": round(float(np.std(rms_db)), 1),
        "speech_rate": round(len(onsets) / speech_s, 2),
        "spectral_centroid_hz": round(float(librosa.feature.spectral_centroid(y=speech, sr=SAMPLE_RATE, hop_length=HOP).mean()), 0),
        "mfcc": np.round(mfcc.mean(axis=1), 2).tolist(),
    })
    return features
