import logging

import uvicorn

from .config import load_settings
from .main import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
settings = load_settings()
uvicorn.run(create_app(settings), host=settings.server.host, port=settings.server.port, ws_max_size=8 * 1024 * 1024)
