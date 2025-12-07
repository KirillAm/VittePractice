from collections import Counter
from typing import Dict, List, Any

# Базовый "идеальный" набор для одного места (можно корректировать)
IDEAL_WORKPLACE = {
    "monitor": 1,
    "keyboard": 1,
    "mouse": 1,
    "cpu": 1,
}

def summarize_inventory(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Считает объекты по классам и формирует простую инвентаризационную сводку."""
    class_counts = Counter(d["class_name"] for d in detections)

    # Оценим количество рабочих мест как минимум по мониторам (условно)
    num_workplaces = class_counts.get("monitor", 0)

    missing_per_place = {}
    extra_items = {}

    for item, required in IDEAL_WORKPLACE.items():
        total_required = required * num_workplaces
        total_found = class_counts.get(item, 0)

        if total_found < total_required:
            missing_per_place[item] = total_required - total_found

    # "лишние" объекты — например, батареи, мобильники и т.п.
    for cls, cnt in class_counts.items():
        if cls not in IDEAL_WORKPLACE:
            extra_items[cls] = cnt

    return {
        "class_counts": dict(class_counts),
        "num_workplaces_estimate": num_workplaces,
        "missing_items": missing_per_place,
        "extra_items": extra_items,
    }
