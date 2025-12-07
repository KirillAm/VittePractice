# app/config.py
"""
Конфигурация Flask-приложения «умный глаз».

Здесь задаются:
- пути к весам модели и data.yaml;
- лимиты по размеру загружаемых файлов;
- параметры подключения к ProxyAPI (OpenAI-совместимый API).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта: папка, где лежит run.py / wsgi.py
BASE_DIR = Path(__file__).resolve().parent.parent

# Подгружаем .env, если есть
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-not-for-production")

    # Лимит на размер загружаемого файла (32 МБ)
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024

    # Пути к модели (можно переопределить через переменные окружения)
    MODEL_WEIGHTS_PATH = os.getenv(
        "MODEL_WEIGHTS_PATH",
        str(BASE_DIR / "models" / "yolov8n_combined_best.pt"),
    )
    MODEL_DATA_YAML = os.getenv(
        "MODEL_DATA_YAML",
        str(BASE_DIR / "models" / "data.yaml"),
    )

    # Torch-устройство: 'cpu' или 'cuda:0'
    TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")

    # ProxyAPI / OpenAI-совместимый клиент
    PROXYAPI_API_KEY = os.getenv("PROXYAPI_API_KEY", "")
    PROXYAPI_BASE_URL = os.getenv(
        "PROXYAPI_BASE_URL",
        "https://openai.api.proxyapi.ru/v1",
    )
    # Модель — подставь ту, которую реально хочешь использовать через ProxyAPI
    PROXYAPI_MODEL = os.getenv(
        "PROXYAPI_MODEL",
        "gpt-4o-mini",
    )
