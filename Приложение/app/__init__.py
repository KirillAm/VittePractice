# app/__init__.py

from pathlib import Path
from typing import Optional, Type

from flask import Flask

from .config import Config
from .detector import Detector
from .ai_assistant import AssistantClient
from .utils import ensure_dir

# Глобальные ссылки на объекты-одиночки
detector: Optional[Detector] = None
assistant: Optional[AssistantClient] = None


def create_app(config_class: Type[Config] = Config) -> Flask:
    """Фабрика Flask-приложения."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # ---------------------------
    # Каталоги для статических файлов
    # ---------------------------
    # Flask сам ставит static_folder = "<путь>/app/static"
    static_root = Path(app.static_folder)

    upload_root = ensure_dir(static_root / "uploads")
    results_root = ensure_dir(static_root / "results")

    # Сохраняем в конфиг, чтобы использовать в routes
    app.config["UPLOAD_FOLDER"] = str(upload_root)
    app.config["RESULTS_FOLDER"] = str(results_root)

    # ---------------------------
    # Инициализация детектора YOLO
    # ---------------------------
    model_path = Path(app.config["MODEL_WEIGHTS_PATH"])
    data_yaml_path = Path(app.config["MODEL_DATA_YAML"])

    global detector
    detector = Detector(
        weights_path=model_path,
        data_yaml_path=data_yaml_path,
        device=app.config.get("TORCH_DEVICE", "cpu"),
        conf_threshold=float(app.config.get("CONF_THRESHOLD", 0.25)),
        iou_threshold=float(app.config.get("IOU_THRESHOLD", 0.45)),
    )

    # ---------------------------
    # Инициализация ИИ-ассистента
    # ---------------------------
    global assistant
    assistant = AssistantClient(
        api_key=app.config["PROXYAPI_API_KEY"],
        base_url=app.config["PROXYAPI_BASE_URL"],
        model=app.config["PROXYAPI_MODEL"],
    )

    # Регистрация blueprint'а
    from .routes import main_bp

    app.register_blueprint(main_bp)

    return app
