# -*- coding: utf-8 -*-
"""
### Шаг 1. Подготовка окружения в Google Colab и настройка доступа к Kaggle

На этом шаге мы настраиваем рабочее окружение для экспериментов:

1. Создаём базовую структуру проекта в каталоге Colab (`/content/computer_lab_detector`).
2. Устанавливаем необходимые библиотеки:
   * `ultralytics` — реализация моделей YOLOv8;
   * `kaggle` — для работы с Kaggle API и скачивания датасетов;
   * базовые утилиты для анализа данных и работы с изображениями (`numpy`, `pandas`, `matplotlib`, `opencv-python` и др.).
3. Настраиваем доступ к Kaggle через `kaggle.json`, загружая файл с помощью `files.upload()` напрямую в Colab (без Google Drive).
4. Проверяем, что Kaggle API работает (вывод версии и тестовый поиск датасета LabEquipVis).

После выполнения всех ячеек этого шага окружение будет готово для скачивания и предварительной обработки датасетов.
"""
# Базовая настройка проекта и проверка версии Python

import os
import sys
from pathlib import Path

# Базовая директория проекта внутри Colab
PROJECT_ROOT = Path("/content") / "computer_lab_detector"

# Подкаталоги для данных и моделей
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

# Создаём директории, если их ещё нет
for p in [PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"Python version: {sys.version}")

# На всякий случай переходим в корень проекта
os.chdir(PROJECT_ROOT)
print("Текущая рабочая директория:", os.getcwd())

# Установка необходимых библиотек (YOLOv8, Kaggle и утилиты)

# Флаг quiet=-q уменьшает шум от pip, но при необходимости можно убрать, чтобы видеть полный лог.
!pip install -q ultralytics kaggle opencv-python matplotlib pandas numpy tqdm

# Быстрая проверка, что ultralytics установился
from ultralytics import YOLO

print("Ultralytics YOLO успешно импортирован.")

# Загрузка kaggle.json через files.upload() и настройка Kaggle API
import os
from google.colab import files

# Загружаем kaggle.json вручную с локального компьютера
print("Выберите файл kaggle.json (скачанный с вашего аккаунта Kaggle).")
uploaded_files = files.upload()

if "kaggle.json" not in uploaded_files:
    raise RuntimeError(
        "Файл kaggle.json не найден среди загруженных. "
        "Убедитесь, что выбрали правильный файл и повторите загрузку."
    )

# Создаём каталог для конфигурации Kaggle
kaggle_dir = Path("/root/.kaggle")
kaggle_dir.mkdir(parents=True, exist_ok=True)

kaggle_config_path = kaggle_dir / "kaggle.json"

# Сохраняем загруженный kaggle.json в /root/.kaggle
with open(kaggle_config_path, "wb") as f:
    f.write(uploaded_files["kaggle.json"])

# Ограничиваем права доступа к файлу (обязательно для Kaggle API)
os.chmod(kaggle_config_path, 0o600)

# На всякий случай укажем путь к конфигу через переменную окружения
os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)

print("kaggle.json успешно сохранён в", kaggle_config_path)

# Проверка работы Kaggle API

# Проверим, что kaggle установлен и видит наш конфиг
!kaggle --version

print("\nПробуем найти датасет LabEquipVis на Kaggle (по ключевому слову 'labequipvis'):")
!kaggle datasets list -s "labequipvis" | head -n 10

"""### Шаг 2. Загрузка и первичная организация датасетов LabEquipVis и E-Waste

На этом шаге мы:

1. Скачиваем два датасета с Kaggle при помощи Kaggle CLI:
   * **LabEquipVis: Dataset of Computer Lab Equipment** — изображения компьютерных лабораторий с разметкой в формате YOLO.
   * **E Waste Image Dataset** (`akshat103/e-waste-image-dataset`) — изображения 10 типов электронных устройств, разложенные по папкам классов внутри директорий `train`, `val`, `test` в подпапке `modified-dataset`.
2. Распаковываем архивы в подкаталоги `data/raw/labequipvis/` и `data/raw/e_waste_image_dataset/`.
3. Выполняем быстрый «sanity check»:
   * для **LabEquipVis** подсчитываем количество изображений и файлов разметки (`.txt`);
   * для **E-waste** выводим список классов и количество изображений в каждом сплите (`train`, `val`, `test`) из `modified-dataset`.

Это подготовит «сырой» слой данных (`raw`), который затем будет конвертирован в единый формат YOLO для общего датасета «компьютерные классы».

"""

# Конфигурация путей и вспомогательные функции

import os
from pathlib import Path
import zipfile

# Если проект не инициализировали на шаге 1 (или запущен новый сеанс), создаём пути заново
try:
    PROJECT_ROOT
except NameError:
    PROJECT_ROOT = Path("/content") / "computer_lab_detector"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    RUNS_DIR = PROJECT_ROOT / "runs"

    for p in [PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RUNS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_DATA_DIR:", RAW_DATA_DIR)

# Описание используемых датасетов
DATASETS = {
    "labequipvis": {
        "kaggle_id": "bmshahriaalam/labequipvis-dataset-of-computer-lab-equipment",
        "root_dir": RAW_DATA_DIR / "labequipvis",
    },
    "ewaste": {
        "kaggle_id": "akshat103/e-waste-image-dataset",
        "root_dir": RAW_DATA_DIR / "e_waste_image_dataset",
    },
}


def download_and_unzip_kaggle_dataset(kaggle_id: str, target_dir: Path, force_download: bool = False):
    """
    Скачивает датасет с Kaggle и распаковывает архив в целевую директорию.

    kaggle_id      — строка формата "owner/dataset-name"
    target_dir     — путь, куда сложить zip и распакованный контент
    force_download — если True, перекачивает архив даже если он уже есть
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    slug = kaggle_id.split("/")[-1]
    zip_path = target_dir / f"{slug}.zip"

    if force_download or not zip_path.exists():
        print(f"\n=== Скачиваем {kaggle_id} в {target_dir} ===")
        exit_code = os.system(
            f'kaggle datasets download -d {kaggle_id} -p "{target_dir}" --force'
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Ошибка при скачивании датасета {kaggle_id}. "
                "Проверьте доступность датасета и настройки Kaggle."
            )
        if not zip_path.exists():
            # если вдруг архив назван иначе — берём первый попавшийся .zip
            zips = list(target_dir.glob("*.zip"))
            if not zips:
                raise RuntimeError(
                    f"После скачивания не найден zip-файл в {target_dir}."
                )
            zip_path = zips[0]

    print(f"Архив найден: {zip_path.name}")

    print(f"Распаковываем архив в {target_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    print("Распаковка завершена.")


def show_dir_tree(root: Path, max_depth: int = 2, max_files_per_dir: int = 5):
    """
    Красивый вывод структуры каталогов (ограниченной глубины),
    чтобы не захламлять вывод десятками тысяч строк.
    """
    root = Path(root)
    print(f"\nСтруктура каталога: {root}")
    root_depth = len(root.parts)

    for current_root, dirs, files in os.walk(root):
        current_path = Path(current_root)
        depth = len(current_path.parts) - root_depth
        if depth > max_depth:
            continue

        indent = "  " * depth
        print(f"{indent}{current_path.name}/")

        files_sorted = sorted(files)
        for fname in files_sorted[:max_files_per_dir]:
            print(f"{indent}  {fname}")
        if len(files_sorted) > max_files_per_dir:
            print(f"{indent}  ... и ещё {len(files_sorted) - max_files_per_dir} файлов")

    print("—" * 60)


def find_split_dirs_recursive(root: Path):
    """
    Рекурсивно ищет директории train / val / test (без учёта регистра)
    и возвращает словарь {имя_сплита: Path}.
    """
    root = Path(root)
    split_dirs = {}
    target_names = {"train", "test", "val", "valid", "validation"}

    for current_root, dirs, files in os.walk(root):
        for d in dirs:
            name_lower = d.lower()
            if name_lower in target_names:
                split_dirs[name_lower] = Path(current_root) / d

    return split_dirs

# Скачивание и первичный осмотр LabEquipVis
lab_cfg = DATASETS["labequipvis"]
lab_root = lab_cfg["root_dir"]

download_and_unzip_kaggle_dataset(
    kaggle_id=lab_cfg["kaggle_id"],
    target_dir=lab_root,
    force_download=False,
)

show_dir_tree(lab_root, max_depth=3, max_files_per_dir=5)

# Быстрый подсчёт количества изображений и файлов разметки (.txt)
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
num_images = 0
num_label_txt = 0

for current_root, dirs, files in os.walk(lab_root):
    for fname in files:
        ext = Path(fname).suffix.lower()
        if ext in image_exts:
            num_images += 1
        elif ext == ".txt":
            num_label_txt += 1

print(f"\nLabEquipVis: найдено изображений: {num_images}, файлов разметки (.txt): {num_label_txt}")

# Скачивание и первичный осмотр E Waste Image Dataset
ew_cfg = DATASETS["ewaste"]
ew_root = ew_cfg["root_dir"]

download_and_unzip_kaggle_dataset(
    kaggle_id=ew_cfg["kaggle_id"],
    target_dir=ew_root,
    force_download=False,
)

# В этом датасете все нужные сплиты лежат в подпапке modified-dataset
ew_modified_root = ew_root / "modified-dataset"
if not ew_modified_root.exists():
    print(
        "\nПапка modified-dataset не найдена, покажем общую структуру датасета "
        "и остановимся, чтобы не накосячить."
    )
    show_dir_tree(ew_root, max_depth=4, max_files_per_dir=5)
else:
    print("\nИспользуем корень E-waste датасета:", ew_modified_root)
    show_dir_tree(ew_modified_root, max_depth=3, max_files_per_dir=5)

    # Ищем директории train / val / test рекурсивно
    split_dirs = find_split_dirs_recursive(ew_modified_root)

    if not split_dirs:
        print("\nНе удалось найти директории train/val/test даже рекурсивно. Проверьте структуру вручную.")
    else:
        print("\nНайдены директории сплитов E-waste (modified-dataset):")
        for split_name, split_path in split_dirs.items():
            print(f"  {split_name} -> {split_path}")

        # Для каждого сплита выводим список классов и количество изображений по классам
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for split_name, split_path in split_dirs.items():
            print(f"\n=== Сплит: {split_name} ===")
            # Внутри сплита лежат папки классов
            classes = [d for d in split_path.iterdir() if d.is_dir()]
            if not classes:
                print("  В этом сплите не найдены подкаталоги классов.")
                continue

            class_counts = {}
            for class_dir in sorted(classes, key=lambda p: p.name):
                count = 0
                for fname in class_dir.iterdir():
                    if fname.is_file() and fname.suffix.lower() in image_exts:
                        count += 1
                class_counts[class_dir.name] = count

            total_images = sum(class_counts.values())
            print(f"  Всего изображений в сплите: {total_images}")
            print("  Классы и количество изображений:")
            for cls_name, cls_count in class_counts.items():
                print(f"    {cls_name}: {cls_count}")
