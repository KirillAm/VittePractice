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

"""### Шаг 3. Приведение LabEquipVis и E-waste к общему YOLO-формату и сбор «совмещенного» датасета

На этом шаге мы:

1. Читаем `data.yaml` из **LabEquipVis (Augmented Data)** и извлекаем список классов.
2. Сканируем **E-waste** (`modified-dataset`) и берём названия классов из папок (`Battery`, `Keyboard`, `Mouse`, …).
3. Приводим имена классов к единому «каноническому» виду (нижний регистр, подчёркивания вместо пробелов) и строим общий словарь классов.
   * Классы LabEquipVis идут «базой».
   * Классы E-waste добавляются только если ещё не встречались.
4. Собираем единый датасет в `data/processed/combined_yolo/` в формате YOLOv8:
   * `images/train`, `images/val`, `images/test`
   * `labels/train`, `labels/val`, `labels/test`
   * для **LabEquipVis**:
     * копируем изображения из `Augmented Data/train|valid|test/images`;
     * переписываем разметку из `labels`, переиндексируя классы в соответствии с новым словарём;
     * префикс имён файлов: `lab_…`.
   * для **E-waste**:
     * для каждого изображения в папке класса создаём один бокс на весь кадр (`0.5 0.5 1.0 1.0`);
     * класс берём из папки, переиндексируем по общему словарю;
     * префикс имён файлов: `ew_…`.
5. Создаём `data.yaml` для совмещённого датасета и выводим сводную статистику по сплитам.

Этот шаг даёт готовый объединённый набор данных, с которым уже можно обучать YOLOv8.

"""

# Импорт и подготовка: общие классы, чтение data.yaml LabEquipVis, классы E-waste

# Устанавливаем PyYAML, если ещё не установлен
!pip install -q pyyaml

import os
from pathlib import Path
import shutil
from collections import defaultdict
import yaml

# Базовые пути (на случай, если ядро перезапускали)
try:
    PROJECT_ROOT
except NameError:
    PROJECT_ROOT = Path("/content") / "computer_lab_detector"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
COMBINED_ROOT = PROCESSED_DATA_DIR / "combined_yolo"

LAB_ROOT = RAW_DATA_DIR / "labequipvis"
LAB_AUG_ROOT = LAB_ROOT / "Augmented Data"  # используем аугментированный набор
EW_ROOT = RAW_DATA_DIR / "e_waste_image_dataset"
EW_MOD_ROOT = EW_ROOT / "modified-dataset"

for p in [PROCESSED_DATA_DIR, COMBINED_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_DATA_DIR:", RAW_DATA_DIR)
print("COMBINED_ROOT:", COMBINED_ROOT)


def canonical_name(name: str) -> str:
    """
    Приводит имя класса к каноническому виду:
    - обрезает пробелы по краям
    - заменяет пробелы и дефисы на подчёркивания
    - переводит в нижний регистр
    """
    name = name.strip()
    name = name.replace(" ", "_").replace("-", "_")
    return name.lower()


# ---------- 1. Классы LabEquipVis (из data.yaml) ----------

lab_yaml_path = LAB_AUG_ROOT / "data.yaml"
if not lab_yaml_path.exists():
    raise FileNotFoundError(f"Не найден data.yaml LabEquipVis по пути {lab_yaml_path}")

with open(lab_yaml_path, "r") as f:
    lab_yaml = yaml.safe_load(f)

lab_names_raw = lab_yaml.get("names")
if lab_names_raw is None:
    raise ValueError("В LabEquipVis data.yaml не найден ключ 'names'.")

# В YOLO data.yaml names может быть списком или словарём {id: name}
if isinstance(lab_names_raw, dict):
    # сортируем по индексу класса
    lab_class_names = [lab_names_raw[k] for k in sorted(lab_names_raw.keys(), key=int)]
else:
    lab_class_names = list(lab_names_raw)

print("\nКлассы LabEquipVis (как в data.yaml):")
for idx, name in enumerate(lab_class_names):
    print(f"  {idx}: {name}")

# Строим общий словарь классов, начиная с LabEquipVis
canonical_to_idx = {}
final_class_names = []  # имена классов по новым индексам (для общего data.yaml)
lab_index_map = {}      # сопоставление: старый индекс LabEquipVis -> новый индекс

for old_idx, name in enumerate(lab_class_names):
    c = canonical_name(name)
    if c in canonical_to_idx:
        new_idx = canonical_to_idx[c]
    else:
        new_idx = len(final_class_names)
        canonical_to_idx[c] = new_idx
        final_class_names.append(c)
    lab_index_map[old_idx] = new_idx

print("\nСоответствие индексов LabEquipVis -> объединённые классы:")
for old_idx, name in enumerate(lab_class_names):
    c = canonical_name(name)
    print(f"  Lab {old_idx}: {name} -> {c} -> new_id={lab_index_map[old_idx]}")

# ---------- 2. Классы E-waste (из имён папок в modified-dataset/train) ----------

if not EW_MOD_ROOT.exists():
    raise FileNotFoundError(f"Не найдена папка modified-dataset по пути {EW_MOD_ROOT}")

ew_train_root = EW_MOD_ROOT / "train"
if not ew_train_root.exists():
    raise FileNotFoundError(f"Не найдена папка train в {EW_MOD_ROOT}")

# при желании можно ограничить список подключаемых классов E-waste:
# например, только компьютерные: {"Keyboard", "Mouse", "Printer"}
EWASTE_INCLUDED_CLASSES = None  # или set([...])

ew_name_to_final_idx = {}

ew_class_dirs = sorted([d for d in ew_train_root.iterdir() if d.is_dir()],
                       key=lambda p: p.name)

print("\nКлассы E-waste (по папкам в train):")
for class_dir in ew_class_dirs:
    orig_name = class_dir.name
    if EWASTE_INCLUDED_CLASSES is not None and orig_name not in EWASTE_INCLUDED_CLASSES:
        print(f"  [Пропускаем] {orig_name}")
        continue

    c = canonical_name(orig_name)
    if c in canonical_to_idx:
        new_idx = canonical_to_idx[c]
    else:
        new_idx = len(final_class_names)
        canonical_to_idx[c] = new_idx
        final_class_names.append(c)

    ew_name_to_final_idx[orig_name] = new_idx
    print(f"  {orig_name} -> {c} -> new_id={new_idx}")

print("\nИтоговый список классов (объединённый):")
for i, name in enumerate(final_class_names):
    print(f"  {i}: {name}")
print(f"Всего классов: {len(final_class_names)}")

# Формирование объединённого датасета YOLO: копирование изображений и генерация разметки
from collections import defaultdict

# Создаём структуру каталогов для объединённого датасета
for split in ["train", "val", "test"]:
    (COMBINED_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (COMBINED_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

combined_stats = {
    "lab": defaultdict(int),
    "ewaste": defaultdict(int),
}

# ---------- 1. Копирование и переиндексация LabEquipVis (Augmented Data) ----------

lab_split_map = {
    "train": "train",
    "valid": "val",   # valid -> val
    "test": "test",
}

print("\n=== Перенос LabEquipVis (Augmented Data) ===")

for src_split, dst_split in lab_split_map.items():
    src_img_dir = LAB_AUG_ROOT / src_split / "images"
    src_lbl_dir = LAB_AUG_ROOT / src_split / "labels"

    dst_img_dir = COMBINED_ROOT / "images" / dst_split
    dst_lbl_dir = COMBINED_ROOT / "labels" / dst_split

    if not src_img_dir.exists():
        print(f"  [Пропуск] Нет каталога изображений {src_img_dir}")
        continue

    img_files = sorted([p for p in src_img_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in image_exts])

    print(f"  Сплит {src_split} -> {dst_split}: найдено {len(img_files)} изображений.")

    for img_path in img_files:
        stem = img_path.stem
        src_label_path = src_lbl_dir / f"{stem}.txt"
        if not src_label_path.exists():
            # На всякий случай защищаемся от рассинхронизации
            # (в этом датасете такого быть не должно)
            # Можно вывести предупреждение и пропустить изображение
            # print(f"    [WARN] Для {img_path.name} не найден label {src_label_path.name}")
            continue

        new_stem = f"lab_{stem}"
        dst_img_path = dst_img_dir / f"{new_stem}{img_path.suffix.lower()}"
        dst_label_path = dst_lbl_dir / f"{new_stem}.txt"

        # Копируем изображение
        shutil.copy2(img_path, dst_img_path)

        # Переписываем разметку с переиндексацией классов
        with open(src_label_path, "r") as f_in, open(dst_label_path, "w") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_cls = int(parts[0])
                new_cls = lab_index_map[old_cls]
                rest = parts[1:]
                f_out.write(" ".join([str(new_cls)] + rest) + "\n")

        combined_stats["lab"][dst_split] += 1

print("Готово перенесён LabEquipVis.")


# ---------- 2. Копирование и генерация разметки для E-waste ----------

print("\n=== Перенос E-waste (modified-dataset) ===")

for split in ["train", "val", "test"]:
    split_dir = EW_MOD_ROOT / split
    if not split_dir.exists():
        print(f"  [Пропуск] Нет каталога сплита {split_dir}")
        continue

    dst_img_dir = COMBINED_ROOT / "images" / split
    dst_lbl_dir = COMBINED_ROOT / "labels" / split

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()],
                        key=lambda p: p.name)

    total_images_in_split = 0

    for class_dir in class_dirs:
        orig_name = class_dir.name
        if EWASTE_INCLUDED_CLASSES is not None and orig_name not in EWASTE_INCLUDED_CLASSES:
            # этот класс мы решили не включать
            continue

        if orig_name not in ew_name_to_final_idx:
            # класс не вошёл в итоговый словарь (например, был отфильтрован)
            continue

        cls_idx = ew_name_to_final_idx[orig_name]
        canon_cls_name = canonical_name(orig_name)

        img_files = sorted([p for p in class_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in image_exts])

        for img_path in img_files:
            total_images_in_split += 1
            new_stem = f"ew_{split}_{canon_cls_name}_{img_path.stem}"
            dst_img_path = dst_img_dir / f"{new_stem}{img_path.suffix.lower()}"
            dst_label_path = dst_lbl_dir / f"{new_stem}.txt"

            # Копируем изображение
            shutil.copy2(img_path, dst_img_path)

            # Для классификационного датасета делаем один бокс на весь кадр:
            # x_center=0.5, y_center=0.5, width=1.0, height=1.0
            with open(dst_label_path, "w") as f_out:
                f_out.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")

            combined_stats["ewaste"][split] += 1

    print(f"  Сплит {split}: перенесено {total_images_in_split} изображений.")

print("Готово перенесён E-waste.")

print("\nСводная статистика по объединённому датасету:")
for src_name, stats in combined_stats.items():
    print(f"Источник: {src_name}")
    for split, count in stats.items():
        print(f"  {split}: {count} изображений")

# Создание общего data.yaml и финальная проверка
import yaml

combined_yaml_path = COMBINED_ROOT / "data.yaml"

data_yaml = {
    "path": str(COMBINED_ROOT.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(final_class_names),
    # YOLO допускает как список, так и dict; сделаем dict {id: name}
    "names": {i: name for i, name in enumerate(final_class_names)},
}

with open(combined_yaml_path, "w") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

print("data.yaml для объединённого датасета сохранён по пути:")
print(combined_yaml_path)

print("\nИтоговый список классов:")
for i, name in enumerate(final_class_names):
    print(f"  {i}: {name}")

# Быстрая проверка количества файлов в каждом сплите
def count_files_in_dir(root: Path, ext_set):
    root = Path(root)
    count = 0
    if not root.exists():
        return 0
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in ext_set:
            count += 1
    return count

print("\nПроверка количества изображений в объединённом датасете:")
for split in ["train", "val", "test"]:
    img_dir = COMBINED_ROOT / "images" / split
    lbl_dir = COMBINED_ROOT / "labels" / split
    n_img = count_files_in_dir(img_dir, {".jpg", ".jpeg", ".png", ".bmp"})
    n_lbl = count_files_in_dir(lbl_dir, {".txt"})
    print(f"  {split}: {n_img} изображений, {n_lbl} файлов разметки")

"""### Шаг 4. Базовое обучение YOLOv8n на объединённом датасете

На этом шаге мы:

1. Подгружаем путь к объединённому датасету `combined_yolo/data.yaml` и проверяем список классов.
2. Запускаем обучение модели **YOLOv8n** (лёгкий вариант) на объединённом датасете:
   * используем предобученные веса `yolov8n.pt` (обучение с дообучением, а не «с нуля»);
   * задаём размер изображения `640`, количество эпох (по умолчанию 50) и размер батча;
   * включаем раннюю остановку по `patience`, чтобы не тратить лишнее GPU-время в Colab.
3. После обучения:
   * берём **best-weights** модели;
   * валидируем её отдельно на `test`-сплите совмещённого датасета;
   * сохраняем веса в папку `models/` и визуализируем несколько примеров предсказаний из `test`.

При необходимости параметры обучения (число эпох, размер батча, imgsz и т.п.) можно позже подправить под ограничения конкретной сессии Colab.
"""

# Подготовка: пути, загрузка data.yaml, проверка классов
from pathlib import Path
import yaml
import torch

# Базовые пути (на случай перезапуска ядра)
try:
    PROJECT_ROOT
except NameError:
    PROJECT_ROOT = Path("/content") / "computer_lab_detector"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
COMBINED_ROOT = PROCESSED_DATA_DIR / "combined_yolo"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

for p in [MODELS_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

combined_yaml_path = COMBINED_ROOT / "data.yaml"
if not combined_yaml_path.exists():
    raise FileNotFoundError(f"Не найден объединённый data.yaml: {combined_yaml_path}")

print("Используем датасет:", combined_yaml_path)

with open(combined_yaml_path, "r") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml.get("names")
if isinstance(class_names, dict):
    # dict {id: name} -> список по индексу
    class_names_list = [class_names[i] for i in sorted(class_names.keys())]
else:
    class_names_list = list(class_names)

print("\nКлассы датасета:")
for i, name in enumerate(class_names_list):
    print(f"  {i}: {name}")

device = 0 if torch.cuda.is_available() else "cpu"
print("\nCUDA доступна:", torch.cuda.is_available(), "| device:", device)

# Обучение YOLOv8n на объединённом датасете

from ultralytics import YOLO

# Загружаем предобученную модель YOLOv8n (COCO)
model = YOLO("yolov8n.pt")

# Настройки обучения — при желании подправь epochs / batch
TRAIN_EPOCHS = 50          # можно уменьшить до 30 в слабой сессии
TRAIN_BATCH = 16           # уменьшить до 8, если не хватает памяти
IMG_SIZE = 640

train_results = model.train(
    data=str(combined_yaml_path),   # путь к data.yaml
    imgsz=IMG_SIZE,
    epochs=TRAIN_EPOCHS,
    batch=TRAIN_BATCH,
    workers=2,                      # в Colab достаточно 2, чтобы не ловить баги
    device=device,
    project=str(RUNS_DIR),          # /content/computer_lab_detector/runs
    name="yolov8n_combined_v1",     # имя эксперимента
    exist_ok=True,                  # не ругаться, если папка уже есть
    seed=42,
    patience=15,                    # ранняя остановка по валидации
    verbose=True,
)

print("\nОбучение завершено.")
# В ultralytics после train у модели есть объект trainer с путём к run-директории
run_dir = Path(model.trainer.save_dir)
print("Каталог эксперимента:", run_dir)

best_weights_path = run_dir / "weights" / "best.pt"
print("Путь к лучшим весам:", best_weights_path)

if not best_weights_path.exists():
    raise FileNotFoundError(
        f"best.pt не найден по пути {best_weights_path}. Проверь логи обучения."
    )
