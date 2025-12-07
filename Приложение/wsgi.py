"""
wsgi.py

WSGI-точка входа для production-сервера (gunicorn, uWSGI и т.п.).

Пример запуска gunicorn:
    gunicorn wsgi:app --bind 0.0.0.0:8000 --workers 4

Важно: конфигурация берётся из app.config.Config,
       которая, в свою очередь, читает переменные окружения и .env.
"""

from app import create_app
from app.config import Config

# WSGI-приложение, которое будет подхватывать сервер
app = create_app(Config)
