import argparse
import asyncio
import base64
import json
import sys
import time

import cv2
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000


def now_ms() -> int:
    return time.time_ns() // 1_000_000


async def stream_video(ws, args: argparse.Namespace, stop: asyncio.Event) -> None:
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        print("Камеру не знайдено, відео не надсилається.")
        return
    period = 1.0 / args.fps
    try:
        while not stop.is_set():
            started = time.monotonic()
            ok, frame = await asyncio.to_thread(capture.read)
            if ok:
                ts = now_ms()
                frame = cv2.resize(frame, (640, 480))
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                await ws.send(json.dumps({
                    "type": "video_frame",
                    "ts": ts,
                    "jpeg_b64": base64.b64encode(jpeg.tobytes()).decode(),
                }))
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        capture.release()


async def stream_audio(ws, args: argparse.Namespace, stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    chunks: asyncio.Queue = asyncio.Queue()
    chunk_ms = int(args.chunk_s * 1000)

    def on_audio(indata, frames, time_info, status):
        loop.call_soon_threadsafe(chunks.put_nowait, (now_ms() - chunk_ms, indata.copy()))

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                blocksize=int(SAMPLE_RATE * args.chunk_s), callback=on_audio)
    except sd.PortAudioError as exc:
        print(f"Мікрофон недоступний ({exc}), аудіо не надсилається.")
        return
    with stream:
        while not stop.is_set():
            try:
                ts, samples = await asyncio.wait_for(chunks.get(), timeout=0.5)
            except TimeoutError:
                continue
            await ws.send(json.dumps({
                "type": "audio_chunk",
                "ts": ts,
                "pcm16_b64": base64.b64encode(samples.tobytes()).decode(),
                "sample_rate": SAMPLE_RATE,
            }))


async def receive(ws) -> None:
    async for raw in ws:
        message = json.loads(raw)
        kind = message.get("type")
        if kind == "hint":
            print(f"[{message['state']} | {message['confidence']:.2f} | {message['mode']} | "
                  f"{message['latency_ms']} мс] {message['reply']} (тон: {message['tone']})")
        elif kind == "status":
            print(f"Сесія {message['session_id']}: {message['session']}, режим {message['mode']}")
        elif kind == "error":
            print(f"Помилка сервера: {message['detail']}")
        else:
            print(raw)


async def run(args: argparse.Namespace) -> None:
    url = f"ws://{args.host}:{args.port}/ws"
    async with websockets.connect(url, max_size=None) as ws:
        print(f"Підключено до {url}. Зупинка: Ctrl+C.")
        await ws.send(json.dumps({"type": "session", "action": "start", "mode": args.mode}))
        stop = asyncio.Event()
        tasks = [asyncio.create_task(receive(ws))]
        if not args.no_video:
            tasks.append(asyncio.create_task(stream_video(ws, args, stop)))
        if not args.no_audio:
            tasks.append(asyncio.create_task(stream_audio(ws, args, stop)))
        try:
            if args.duration:
                await asyncio.sleep(args.duration)
            else:
                await asyncio.Future()
        finally:
            stop.set()
            await ws.send(json.dumps({"type": "session", "action": "stop"}))
            await asyncio.sleep(0.5)
            for task in tasks:
                task.cancel()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Тестовий клієнт: вебкамера і мікрофон ноутбука замість телефона")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["fuzzy", "llm", "hybrid"], default="fuzzy")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--chunk-s", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.0, help="секунд; 0 — до Ctrl+C")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    try:
        asyncio.run(run(parser.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
