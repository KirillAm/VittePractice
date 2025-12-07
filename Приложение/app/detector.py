# app/detector.py
"""
Обёртка над YOLOv8 (ultralytics) для детекции объектов на изображении.

- Загружает веса из .pt;
- Прогоняет инференс по одному изображению;
- Сохраняет картинку с подсвеченными боксами;
- Возвращает список детекций в удобном формате.
"""

from pathlib import Path
from typing import List, Dict, Any

import cv2
import json
from ultralytics import YOLO


class Detector:
    def __init__(
        self,
        weights_path: Path,
        data_yaml_path: Path,
        device: str = "cpu",
        conf_threshold: float = 0.20,   # чуть ниже, чтобы меньше пропускать
        iou_threshold: float = 0.45,
        imgsz: int = 960,               # подробнее, чем 640
        augment: bool = True,           # тестовые аугментации при инференсе
    ) -> None:
        self.weights_path = Path(weights_path)
        self.data_yaml_path = Path(data_yaml_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.augment = augment

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Файл весов модели не найден: {self.weights_path}"
            )

        # Загружаем модель YOLOv8
        self.model = YOLO(str(self.weights_path))
        try:
            self.model.to(self.device)
        except Exception:
            # Если что-то не так с CUDA — тихо падаем на CPU
            self.device = "cpu"
            self.model.to(self.device)

        # Загружаем список классов из classes.json (чтобы не парсить yaml)
        classes_path = Path(__file__).resolve().parent / "model_config" / "classes.json"
        if classes_path.exists():
            with classes_path.open("r", encoding="utf-8") as f:
                self.class_names = json.load(f)
        else:
            # Фолбэк: имена классов будем брать из самой модели
            self.class_names = self.model.names

    def detect_image(self, image_path: Path, save_path: Path) -> List[Dict[str, Any]]:
        """
        Выполняет детекцию на одном изображении и сохраняет визуализацию.

        :param image_path: путь к исходному изображению
        :param save_path: путь для сохранения изображения с боксами
        :return: список детекций:
            [
                {
                    "class_id": int,
                    "class_name": str,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],
                },
                ...
            ]
        """
        image_path = Path(image_path)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Прогоняем модель
        results_list = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            augment=self.augment,
        )

        if not results_list:
            return []

        result = results_list[0]

        # Сохраняем изображение с боксами
        plotted = result.plot()  # BGR ndarray
        cv2.imwrite(str(save_path), plotted)

        detections: List[Dict[str, Any]] = []

        boxes = result.boxes
        if boxes is None:
            return detections

        # boxes.xyxy, boxes.cls, boxes.conf
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            # Имя класса
            if isinstance(self.class_names, dict):
                class_name = self.class_names.get(cls_id, str(cls_id))
            else:
                # Если это список
                try:
                    class_name = self.class_names[cls_id]
                except Exception:
                    class_name = str(cls_id)

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )

        return detections
