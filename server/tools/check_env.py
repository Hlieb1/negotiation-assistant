import importlib
import os
import shutil
import sys
from pathlib import Path

MODULES = ["fastapi", "uvicorn", "pydantic", "yaml", "numpy", "sklearn", "cv2", "tensorflow", "deepface",
           "librosa", "webrtcvad", "noisereduce", "sounddevice", "faster_whisper", "ctranslate2", "skfuzzy", "httpx"]


def add_nvidia_dll_dirs() -> None:
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if os.name != "nt" or not nvidia.exists():
        return
    for bin_dir in nvidia.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def check_module(name: str) -> None:
    try:
        module = importlib.import_module(name)
        print(f"  [ok] {name:16s} {getattr(module, '__version__', '')}")
    except Exception as exc:
        print(f"  [!!] {name:16s} {exc}")


def check_gpu() -> None:
    import ctranslate2

    count = ctranslate2.get_cuda_device_count()
    print(f"  CTranslate2 CUDA devices: {count}")
    if count:
        print(f"  compute types: {sorted(ctranslate2.get_supported_compute_types('cuda'))}")


def check_whisper() -> None:
    import numpy as np
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cuda", compute_type="float16")
    segments, info = model.transcribe(np.zeros(16000, dtype=np.float32), language="uk")
    list(segments)
    print(f"  [ok] whisper small on cuda, language={info.language}")


def check_ollama() -> None:
    import httpx

    version = httpx.get("http://127.0.0.1:11434/api/version", timeout=5).json()["version"]
    models = [m["name"] for m in httpx.get("http://127.0.0.1:11434/api/tags", timeout=5).json()["models"]]
    print(f"  [ok] version {version}, models: {models or 'none'}")


def run(title: str, check) -> None:
    print(title)
    try:
        check()
    except Exception as exc:
        print(f"  [!!] {exc}")


def main() -> None:
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    print(f"ffmpeg: {shutil.which('ffmpeg') or 'not found'}")
    print("Modules:")
    for name in MODULES:
        check_module(name)
    add_nvidia_dll_dirs()
    run("GPU:", check_gpu)
    if "--whisper" in sys.argv:
        run("faster-whisper:", check_whisper)
    run("Ollama (http://127.0.0.1:11434):", check_ollama)


if __name__ == "__main__":
    main()
