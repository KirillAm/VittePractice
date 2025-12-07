# app/routes.py
from pathlib import Path

from flask import (
    Blueprint,
    render_template,
    request,
    current_app,
    redirect,
    url_for,
    flash,
)

from . import detector, assistant
from .utils import (
    ensure_dir,
    allowed_file,
    generate_safe_filename,
    detections_to_table,
    summarize_by_class,
    to_relative_path,
)

main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # POST: загрузка файла
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Не выбран файл для анализа.", "error")
        return redirect(url_for("main.index"))

    if not allowed_file(file.filename):
        flash("Недопустимый тип файла. Разрешены изображения и видео.", "error")
        return redirect(url_for("main.index"))

    upload_root = Path(current_app.config["UPLOAD_FOLDER"])
    ensure_dir(upload_root)

    # Сохраняем оригинал
    safe_name = generate_safe_filename(file.filename, prefix="upload")
    orig_path = upload_root / safe_name
    file.save(str(orig_path))

    # Решаем, картинка это или видео по расширению
    ext = orig_path.suffix.lower()
    is_image = ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    if is_image:
        return _handle_image(orig_path)
    else:
        return _handle_video(orig_path)

def _handle_image(orig_path: Path):
    """Обработка одиночного изображения."""
    if detector is None:
        flash("Детектор объектов не инициализирован.", "error")
        return redirect(url_for("main.index"))

    # Папка для результатов
    results_root = Path(current_app.config["RESULTS_FOLDER"])
    ensure_dir(results_root)

    # Путь к сохранённому изображению с детекциями
    detected_name = f"det_{orig_path.name}"
    detected_path = results_root / detected_name

    # 1. Детекция YOLO
    detections = detector.detect_image(
        image_path=orig_path,
        save_path=detected_path,
    )

    table = detections_to_table(detections, min_confidence=0.25)
    counts = summarize_by_class(detections)

    # 2. Сырая инвентаризационная логика по результатам YOLO
    num_workplaces = counts.get("monitor", 0)

    missing = {}
    extra = {}

    # Пример «идеальной» комплектации: monitor + cpu + keyboard + mouse
    for cls_name in ["cpu", "keyboard", "mouse"]:
        have = counts.get(cls_name, 0)
        if have < num_workplaces:
            missing[cls_name] = num_workplaces - have

    # «Лишние» объекты — техника вне базового набора
    for cls_name, cnt in counts.items():
        if cls_name not in ["monitor", "cpu", "keyboard", "mouse"]:
            extra[cls_name] = cnt

    inventory = {
        "class_counts": counts,
        "num_workplaces_estimate": num_workplaces,
        "missing_items": missing,
        "extra_items": extra,
    }

    # 3. Коррекция инвентаризации по изображению с помощью ИИ-ассистента
    vision_comment = ""
    if assistant is not None:
        try:
            refine_result = assistant.refine_inventory_with_vision(orig_path, inventory)
            inventory = refine_result.get("inventory", inventory)
            vision_comment = refine_result.get("comment", "")
        except Exception as e:
            vision_comment = (
                "Не удалось скорректировать инвентаризацию по изображению: "
                f"{e}"
            )

    # 4. Текстовый отчёт уже по уточнённой инвентаризации
    report_text = ""
    if assistant is not None:
        try:
            report_text = assistant.build_inventory_report(inventory)
        except Exception as e:
            report_text = (
                "Не удалось получить текстовый отчёт от ИИ-ассистента. "
                f"Техническая ошибка: {e}"
            )

    # Оба файла лежат внутри app/static, поэтому делаем относительные пути
    static_root = Path(current_app.static_folder)
    orig_rel = to_relative_path(orig_path, static_root)
    det_rel = to_relative_path(detected_path, static_root)

    return render_template(
        "result_image.html",
        orig_image_path=url_for("static", filename=orig_rel),
        detected_image_path=url_for("static", filename=det_rel),
        detections=table,
        inventory=inventory,          # уже уточнённая инвентаризация
        report_text=report_text,
        vision_comment=vision_comment # можно при желании вывести в шаблоне
    )

def _handle_video(orig_path: Path):
    """
    Заглушка для обработки видео.

    Здесь можно реализовать:
    - выборку кадров через равные интервалы;
    - детекцию на каждом кадре;
    - сохранение обработанного видео / кадров;
    - агрегацию статистики.

    Пока просто сообщаем, что видео не поддерживается, и возвращаемся на главную.
    """
    flash("Обработка видео пока не реализована. Используйте изображения.", "info")
    return redirect(url_for("main.index"))
