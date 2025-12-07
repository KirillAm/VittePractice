"""
run.py

Упрощённый скрипт для локального запуска Flask-приложения «умный глаз».

Запуск:
    python run.py

Параметры можно переопределить через переменные окружения:
    FLASK_RUN_HOST (по умолчанию 0.0.0.0)
    FLASK_RUN_PORT (по умолчанию 5000)
    FLASK_DEBUG    (по умолчанию 1 — режим отладки включён)
"""

import os

from app import create_app
from app.config import Config

# Создаём приложение с базовой конфигурацией
app = create_app(Config)


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    debug_env = os.getenv("FLASK_DEBUG", "1")
    debug = debug_env not in {"0", "false", "False"}

    # В dev-режиме включён reloader/ debugger
    app.run(host=host, port=port, debug=debug)
