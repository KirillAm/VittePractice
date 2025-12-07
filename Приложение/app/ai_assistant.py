# app/ai_assistant.py
"""
Модуль для работы с ИИ-ассистентом через ProxyAPI (OpenAI-совместимый API).

Задачи:
- инициализировать клиент OpenAI с base_url ProxyAPI;
- скорректировать результат автоматической инвентаризации с учётом самого изображения;
- сформировать человеко-понятный отчёт по аудитории.

ВАЖНО: API-ключ не должен быть захардкожен в коде.
Он передаётся через переменную окружения PROXYAPI_API_KEY,
которую читает Config в app/config.py.
"""

import base64
import json
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI


class AssistantClient:
    """Обёртка над LLM/vision для инвентаризации аудитории."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        if not api_key:
            raise ValueError(
                "PROXYAPI_API_KEY не задан. "
                "Установите переменную окружения с ключом ProxyAPI."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        # модель должна поддерживать vision (например, gpt-4o-mini)
        self.model = model

    # ---------- Этап 1: корректировка инвентаризации по изображению ----------

    def refine_inventory_with_vision(
        self, image_path: Path, inventory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Передаёт в модель само изображение + сырую инвентаризацию и просит
        скорректировать оценки.

        На выходе:
            {
                "inventory": {...уточнённая инвентаризация...},
                "comment": "краткое текстовое пояснение"
            }
        """
        image_path = Path(image_path)
        with image_path.open("rb") as f:
            image_bytes = f.read()

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        inventory_json = json.dumps(inventory, ensure_ascii=False)

        prompt_text = (
            "Ты видишь фотографию компьютерной аудитории.\n"
            "Ниже дан предварительный результат детекции объектов (мониторы, системные блоки, "
            "клавиатуры, мыши, светильники, стулья, и т.п.), полученный моделью YOLO.\n"
            "Результат неидеален: некоторые объекты могут быть пропущены или ошибочно добавлены.\n\n"
            "Твоя задача — глядя на изображение и на этот предварительный JSON, "
            "скорректировать инвентаризацию.\n\n"
            f"Предварительная инвентаризация (JSON):\n{inventory_json}\n\n"
            "Что считать рабочим местом: стол с монитором (и, по возможности, с системным блоком, "
            "клавиатурой и мышью). Если на фото явно видно рабочее место без какого-то элемента, "
            "учитывай его как рабочее место, но пометь недостающую технику.\n\n"
            "Верни строго JSON следующего вида:\n"
            "{\n"
            '  \"refined_inventory\": {\n'
            '    \"class_counts\": {\"имя_класса\": int, ...},\n'
            '    \"num_workplaces_estimate\": int,\n'
            '    \"missing_items\": {\"имя_класса\": int, ...},\n'
            '    \"extra_items\": {\"имя_класса\": int, ...}\n'
            "  },\n"
            '  \"comment\": \"краткое текстовое пояснение, что ты поправил\"\n'
            "}\n"
            "Не добавляй никакого текста вне JSON."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }
            ],
        )

        message = response.choices[0].message
        raw_content = message.content if message and message.content is not None else "{}"

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            # Если модель вдруг нарушила формат — просто вернём исходное
            return {
                "inventory": inventory,
                "comment": raw_content,
            }

        refined = data.get("refined_inventory", inventory)
        comment = data.get("comment", "")

        # Подстрахуемся: если модель не вернула ключей, добьём их из исходного
        for key in ["class_counts", "num_workplaces_estimate", "missing_items", "extra_items"]:
            if key not in refined and key in inventory:
                refined[key] = inventory[key]

        return {
            "inventory": refined,
            "comment": comment,
        }

    # ---------- Этап 2: текстовый отчёт по (уже уточнённой) инвентаризации ----------

    def build_inventory_report(self, inventory: Dict[str, Any]) -> str:
        """
        Формирует запрос к модели и возвращает краткий отчёт по аудитории.

        Ожидаемый формат inventory:
            {
                "class_counts": {class_name: count, ...},
                "num_workplaces_estimate": int,
                "missing_items": {class_name: missing_count, ...},
                "extra_items": {class_name: extra_count, ...},
            }
        """
        class_counts = inventory.get("class_counts", {})
        num_workplaces = inventory.get("num_workplaces_estimate", 0)
        missing = inventory.get("missing_items", {})
        extra = inventory.get("extra_items", {})

        user_prompt = (
            "Ты — ассистент по управлению компьютерными аудиториями в вузе.\n"
            "У тебя есть результаты автоматического распознавания объектов "
            "на фотографии компьютерного класса (после той или иной коррекции).\n"
            "Сформируй краткий, но информативный отчёт для администратора МУИВ.\n\n"
            f"Оценка числа рабочих мест: {num_workplaces}\n"
            f"Найденные объекты по классам: {class_counts}\n"
            f"Отсутствующие элементы (по сравнению с идеальным набором): {missing}\n"
            f"Лишние или посторонние объекты: {extra}\n\n"
            "Сделай выводы: хватает ли техники на все рабочие места, есть ли "
            "критичные несоответствия (например, место без системного блока или без монитора), "
            "что стоит проверить в первую очередь. Пиши по-деловому, но живым человеческим языком, "
            "1–2 абзаца без бюрократических штампов."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        message = response.choices[0].message
        return message.content if message and message.content is not None else ""
