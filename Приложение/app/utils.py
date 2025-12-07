# app/utils.py
"""
Вспомогательные функции для Flask-приложения «умный глаз»:
- работа с путями и директориями;
- генерация безопасных имён файлов;
- проверка допустимых расширений;
- простые преобразования структур детекций в удобный формат.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional


# ---------- Работа с путями и директориями ----------


def ensure_dir(path: Path) -> Path:
    """
    Гарантирует существование каталога (создаёт с parents=True, exist_ok=True)
    и возвращает тот же Path (удобно для чейнинга).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Возвращает корень проекта, рассчитанный как два уровня выше текущего файла:
    app/utils.py -> app/ -> <корень проекта>.
    """
    return Path(__file__).resolve().parent.parent


# ---------- Работа с именами файлов ----------

# Допустимые расширения по умолчанию
DEFAULT_ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


def allowed_file(
    filename: str,
    allowed_extensions: Optional[Iterable[str]] = None,
) -> bool:
    """
    Проверяет, допустимо ли расширение файла для загрузки.

    :param filename: исходное имя файла (как пришло от пользователя)
    :param allowed_extensions: коллекция расширений вида {".jpg", ".png"},
                               если None — берём DEFAULT_ALLOWED_EXTENSIONS
    """
    if not filename:
        return False

    ext = Path(filename).suffix.lower()
    exts = set(allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS)
    return ext in exts


def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    «Очищает» строку для использования в путях/имени файла.

    - Переводит в нижний регистр
    - Удаляет небезопасные символы
    - Пробелы/дефисы/подчёркивания -> один дефис
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    value = re.sub(r"[-\s]+", "-", value).strip("-_")
    return value or "file"


def generate_safe_filename(original_name: str, prefix: Optional[str] = None) -> str:
    """
    Генерирует безопасное имя файла:
    - берёт только расширение из исходного имени;
    - формирует случайный uuid;
    - опционально добавляет текстовый префикс (slug).
    """
    ext = Path(original_name).suffix.lower()
    if not ext:
        ext = ".bin"

    uid = uuid.uuid4().hex
    if prefix:
        safe_prefix = slugify(prefix)
        return f"{safe_prefix}_{uid}{ext}"
    return f"{uid}{ext}"


# ---------- Преобразование путей для статики / URL ----------


def to_relative_path(path: Path, root: Path) -> str:
    """
    Преобразует абсолютный путь к файлу в относительный относительно root.

    Удобно, если нужно потом передать результат в url_for('static', filename=...).
    """
    path = path.resolve()
    root = root.resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        # Если path не лежит внутри root — возвращаем только имя файла
        rel = path.name
    return str(rel).replace("\\", "/")


# ---------- Работа с результатами детекции ----------


def detections_to_table(
    detections: List[Dict[str, Any]],
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Превращает список детекций в компактную табличную структуру для шаблонов/JSON.

    Ожидаемый формат одного элемента detections:
        {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
        }
    """
    table: List[Dict[str, Any]] = []
    for det in detections:
        conf = float(det.get("confidence", 0.0))
        if conf < min_confidence:
            continue
        table.append(
            {
                "class_name": det.get("class_name", ""),
                "confidence": round(conf, 3),
                "bbox": det.get("bbox", []),
            }
        )
    return table


def summarize_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Считает количество объектов по имени класса.

    Возвращает словарь вида:
        {"monitor": 12, "keyboard": 10, ...}
    """
    counts: Dict[str, int] = {}
    for det in detections:
        cls = det.get("class_name", "")
        if not cls:
            continue
        counts[cls] = counts.get(cls, 0) + 1
    return counts
