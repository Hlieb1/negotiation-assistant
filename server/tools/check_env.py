"""Environment sanity check for the diploma server.

Usage:  python tools/check_env.py
Prints versions of key libraries, GPU availability and Ollama status.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path


def add_nvidia_dll_dirs() -> None:
    """Make pip-installed CUDA DLLs (nvidia-cublas/cudnn) visible on Windows."""
    if os.name != "nt":
        return
    site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site.exists():
        return
    for bin_dir in site.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def check_module(name: str) -> None:
    try:
        mod = importlib.import_module(name)
        print(f"  [ok] {name:16s} {getattr(mod, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [!!] {name:16s} FAILED: {exc}")


def main() -> None:
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    print(f"ffmpeg: {shutil.which('ffmpeg') or 'NOT FOUND'}")

    print("Modules:")
    for name in ["fastapi", "uvicorn", "pydantic", "yaml", "numpy", "sklearn", "cv2",
                 "tensorflow", "deepface", "librosa", "webrtcvad", "noisereduce",
                 "sounddevice", "faster_whisper", "ctranslate2", "skfuzzy", "httpx"]:
        check_module(name)

    add_nvidia_dll_dirs()
    print("GPU:")
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        print(f"  CTranslate2 CUDA devices: {n}")
        if n:
            print(f"  CTranslate2 compute types (cuda): {sorted(ctranslate2.get_supported_compute_types('cuda'))}")
    except Exception as exc:  # noqa: BLE001
        print(f"  CTranslate2 CUDA check failed: {exc}")

    if "--whisper" in sys.argv:
        print("faster-whisper GPU smoke test (model 'small', downloads ~0.5 GB once):")
        try:
            import numpy as np
            from faster_whisper import WhisperModel
            model = WhisperModel("small", device="cuda", compute_type="float16")
            segments, info = model.transcribe(np.zeros(16000, dtype=np.float32), language="uk")
            list(segments)
            print(f"  [ok] whisper small on cuda, lang={info.language}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [!!] whisper on cuda failed: {exc}")

    print("Ollama (HTTP API, http://127.0.0.1:11434):")
    try:
        import httpx
        version = httpx.get("http://127.0.0.1:11434/api/version", timeout=5).json()["version"]
        models = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5).json()["models"]
        print(f"  [ok] version {version}; models: {[m['name'] for m in models] or 'none'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [!!] Ollama server not reachable ({exc}). Start it with: ollama serve")


if __name__ == "__main__":
    main()
