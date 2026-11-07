$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r requirements.txt
& $python -m pip install -r requirements-gpu.txt

# deepface and hsemotion-onnx pull other OpenCV builds that shadow the contrib one (no GUI, no CascadeClassifier)
& $python -m pip uninstall -y opencv-contrib-python-headless opencv-python opencv-contrib-python
& $python -m pip install --no-deps opencv-contrib-python==5.0.0.93

& $python tools\check_env.py
