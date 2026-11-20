from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

NO_TUMOR = "Опухоли нет"

SECTION_KEYS = ["data", "result", "findings", "limits", "doctor"]
SECTION_TITLES = {
    "data": "Данные исследования",
    "result": "Результат автоматического анализа",
    "findings": "Находки и их расположение",
    "limits": "Сомнения и ограничения анализа",
    "doctor": "Заключение врача",
}
SECTION_ROWS = {"data": 3, "result": 4, "findings": 6, "limits": 8, "doctor": 6}


def frame_stats(view_path):
    """Границы головы и её площадь (в пикселях кадра 224×224)."""
    image = Image.open(view_path).convert("L").resize((224, 224), Image.Resampling.BILINEAR)
    foreground = np.array(image) > 10
    if not foreground.any():
        return {"head_area": 224 * 224, "head_bbox": [0, 0, 223, 223]}
    ys, xs = np.where(foreground)
    return {
        "head_area": int(foreground.sum()),
        "head_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def _pct(value):
    return f"{value * 100:.0f} %"


def make_sections(name, image_size, image_format, result, segmentation_done):
    width, height = image_size
    has_tumor = result["class"] != NO_TUMOR

    data = (
        f"Файл: {name}, {width}×{height} пикселей, формат {image_format}. "
        "Плоскость и режим съёмки не определены. Клинические данные не предоставлены."
    )

    p_tumor = result["p_tumor"]
    if has_tumor:
        text = f"Предполагается наличие опухолевого образования (вероятность {_pct(p_tumor)}). "
        types = sorted(
            (result.get("type_probabilities") or {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if types:
            text += f"Предполагаемый тип: {types[0][0]} ({_pct(types[0][1])})"
            if len(types) > 1:
                text += f"; альтернатива: {types[1][0]} ({_pct(types[1][1])})"
            text += "."
        if result.get("uncertain"):
            text += " Результат помечен как сомнительный."
    else:
        text = (
            "Признаков опухолевого образования по данным автоматического анализа не выявлено "
            f"(вероятность отсутствия опухоли {_pct(1 - p_tumor)})."
        )

    limits = []
    for reason in result.get("reasons", []):
        limits.append(reason[:1].upper() + reason[1:].rstrip(".") + ".")
    if result.get("uncertain") and not result.get("reasons"):
        limits.append("Уверенность модели в ответе ниже 90 %.")
    limits.append(
        "Анализ выполнен по одному 2D срезу; другие снимки и клинические данные не учитывались."
    )
    limits.append(
        "Размеры даны в пикселях: масштаб пикселя в файле отсутствует, размеры в мм не определены."
    )
    if segmentation_done:
        limits.append(
            "Границы области построены автоматически и приблизительны."
        )
    if has_tumor:
        limits.append(
            "Не оценивались: желудочковые системы и субарахноидальные пространства, смещение срединных структур, "
            "наличие расширений и/или дополнительных объёмных образований, число очагов и степень злокачественности."
        )
    else:
        limits.append(
            "Исключить изменения, не выявляемые данным методом, нельзя; оценка врачом обязательна."
        )

    return {
        "data": data,
        "result": text,
        "findings": "",  # формируется на странице проверки по выделенным областям
        "limits": "\n".join("— " + line for line in limits),
        "doctor": "",
    }


def make_report(name, stem, image_size, image_format, result, segmentation_done, result_dir):
    result_dir = Path(result_dir)
    report = {
        "name": name,
        "stem": stem,
        "image_size": list(image_size),
        "image_format": image_format,
        "result": result,
        "segmentation_done": segmentation_done,
        "created": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "sections": make_sections(name, image_size, image_format, result, segmentation_done),
    }
    report.update(frame_stats(result_dir / "view.png"))
    return report


def to_text(report, sections, doctor, saved_at):
    parts = [f"ЗАКЛЮЧЕНИЕ ПО РЕЗУЛЬТАТАМ АНАЛИЗА МРТ ({report['name']})", ""]
    for key in SECTION_KEYS:
        text = (sections.get(key) or "").strip()
        if text:
            parts += [SECTION_TITLES[key].upper(), text, ""]
    parts += [
        f"Врач: {doctor or '________________'}",
        f"Дата проверки: {saved_at}",
        "",
        "Результат носит справочный характер и не является диагнозом.",
    ]
    return "\n".join(parts)
