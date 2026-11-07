import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .config import Settings, load_settings
from .protocol import Error, now_ms, parse_client_message
from .session import Session

logger = logging.getLogger("negotiation")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Negotiation assistant")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("client connected: %s", websocket.client)
        session = Session(settings, websocket.send_text)
        try:
            while True:
                raw = await websocket.receive_text()
                recv_ms = now_ms()
                try:
                    message = parse_client_message(raw)
                except ValidationError as exc:
                    await session.send(Error(ts=recv_ms, detail=exc.errors()[0]["msg"]))
                    continue
                await session.handle(message, recv_ms)
        except WebSocketDisconnect:
            logger.info("client disconnected: %s", websocket.client)
        finally:
            await session.stop()

    return app


app = create_app()
