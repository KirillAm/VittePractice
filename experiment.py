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
