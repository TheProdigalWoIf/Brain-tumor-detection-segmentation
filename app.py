import base64
import csv
import io
import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

import report_builder
from classifier import predict as classify_image
from model_def import CLASSES
from segmenter import prepare_view, segment_image


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"

app = Flask(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_FILES = 10
app.config["MAX_CONTENT_LENGTH"] = MAX_FILES * MAX_FILE_SIZE

ALLOWED_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}

RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def ensure_directories():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_segmentation(input_path, output_path):
    segment_image(str(input_path), str(output_path))


def save_classification_csv(csv_path, result, segmentation_done):
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        writer.writerow(["field", "class", "value"])
        writer.writerow(["predicted_class", result["class"], ""])
        writer.writerow(["confidence", "", result["confidence"]])
        writer.writerow(["uncertain", "", result["uncertain"]])
        writer.writerow(["p_tumor", "", result["p_tumor"]])
        writer.writerow(["reasons", "", "; ".join(result["reasons"])])

        if result["type_probabilities"]:
            for class_name, probability in result["type_probabilities"].items():
                writer.writerow(["type_probability", class_name, probability])

        for class_name, probability in result["probabilities"].items():
            writer.writerow(["class_probability", class_name, probability])

        writer.writerow([
            "segmentation_output",
            "",
            "output.png" if segmentation_done else "",
        ])


def safe_stem(filename):
    """Имя файла без расширения, пригодное для имени файла в Windows."""
    name = Path((filename or "").replace("\\", "/")).stem
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return name[:80] or "snimok"


def analyze_file(file):
    """Обрабатывает один снимок. Возвращает словарь с результатом или с ошибкой."""
    display_name = file.filename or "без имени"
    extension = Path((file.filename or "").replace("\\", "/")).suffix.lower()

    if extension not in ALLOWED_FORMATS:
        return {"name": display_name, "error": "Допустимые форматы: PNG, JPG, BMP, TIFF и WebP."}

    run_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{run_id}{extension}"
    result_dir = RESULTS_DIR / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        file.save(upload_path)

        if upload_path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError("Файл слишком большой. Максимальный размер — 10 МБ.")

        try:
            with Image.open(upload_path) as image:
                actual_format = image.format
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Файл не является корректным изображением.") from exc

        if actual_format != ALLOWED_FORMATS[extension]:
            raise ValueError("Содержимое файла не соответствует его расширению.")

        with Image.open(upload_path) as image:
            image_size = image.size
            image.convert("RGB").save(result_dir / "input.png", format="PNG")

        classification_result = classify_image(str(upload_path))

        segmentation_done = classification_result["class"] != CLASSES[0]

        output_path = result_dir / "output.png"
        if segmentation_done:
            run_segmentation(upload_path, output_path)
            if not output_path.is_file():
                raise RuntimeError("Сегментационная модель не сохранила output.png.")
        else:
            prepare_view(str(upload_path), str(result_dir / "view.png"))

        save_classification_csv(
            result_dir / "data.csv", classification_result, segmentation_done
        )

        report = report_builder.make_report(
            display_name,
            safe_stem(file.filename),
            image_size,
            actual_format,
            classification_result,
            segmentation_done,
            result_dir,
        )
        (result_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )

        return {
            "name": display_name,
            "run_id": run_id,
            "csv_stem": safe_stem(file.filename),
            "result_url": f"/static/results/{run_id}",
            "review_url": f"/review/{run_id}",
            "result": classification_result,
            "probabilities": list(classification_result["probabilities"].items()),
            "segmentation_done": segmentation_done,
        }

    except ValueError as exc:
        shutil.rmtree(result_dir, ignore_errors=True)
        return {"name": display_name, "error": str(exc)}

    except Exception:
        app.logger.exception("Ошибка при обработке изображения")
        shutil.rmtree(result_dir, ignore_errors=True)
        return {
            "name": display_name,
            "error": "Не удалось обработать изображение. Проверьте подключение моделей и логи сервера.",
        }

    finally:
        upload_path.unlink(missing_ok=True)


@app.route("/")
def index():
    ensure_directories()
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    ensure_directories()

    files = [f for f in request.files.getlist("files") if f and f.filename]

    if not files:
        return render_template(
            "index.html", error="Сначала выберите хотя бы один снимок."
        ), 400

    if len(files) > MAX_FILES:
        return render_template(
            "index.html",
            error=f"За один раз можно загрузить не больше {MAX_FILES} снимков.",
        ), 400

    items = [analyze_file(f) for f in files]
    ok_items = [item for item in items if "error" not in item]

    used = set()
    for item in ok_items:
        name = f"{item['csv_stem']}_conclusion.csv"
        n = 2
        while name.lower() in used:
            name = f"{item['csv_stem']}_{n}_conclusion.csv"
            n += 1
        used.add(name.lower())
        item["csv_name"] = name

    zip_url = None
    if len(ok_items) > 1:
        zip_name = f"batch_{uuid.uuid4().hex}.zip"
        with zipfile.ZipFile(RESULTS_DIR / zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in ok_items:
                zf.write(RESULTS_DIR / item["run_id"] / "data.csv", arcname=item["csv_name"])
        zip_url = f"/static/results/{zip_name}"

    return render_template("index.html", items=items, zip_url=zip_url)



# Проверка заключения врачом


def run_dir_or_404(run_id):
    if not RUN_ID_RE.match(run_id):
        abort(404)
    run_dir = RESULTS_DIR / run_id
    if not (run_dir / "report.json").is_file():
        abort(404)
    return run_dir


def read_json(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def decode_png(data_url, max_bytes=3_000_000):
    """PNG из data URL (так холст браузера передаёт изображение) в объект Pillow."""
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("Ожидается изображение PNG.")
    try:
        raw = base64.b64decode(data_url[len(prefix):], validate=True)
    except ValueError as exc:
        raise ValueError("Некорректные данные изображения.") from exc
    if len(raw) > max_bytes:
        raise ValueError("Изображение слишком большое.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Некорректное изображение.") from exc
    return image


@app.route("/review/<run_id>")
def review(run_id):
    run_dir = run_dir_or_404(run_id)
    report = read_json(run_dir / "report.json")
    saved = read_json(run_dir / "review.json")

    base = f"/static/results/{run_id}"
    edited_mask = run_dir / "mask_edited.png"
    mask_path = edited_mask if edited_mask.is_file() else run_dir / "mask.png"
    mask_version = int(mask_path.stat().st_mtime) if mask_path.is_file() else 0
    final_html = run_dir / "conclusion_final.html"

    cfg = {
        "stem": report["stem"],
        "viewUrl": f"{base}/view.png",
        "maskUrl": f"{base}/{mask_path.name}?v={mask_version}" if mask_path.is_file() else None,
        "modelMaskUrl": f"{base}/mask.png" if (run_dir / "mask.png").is_file() else None,
        "headArea": report["head_area"],
        "headBbox": report["head_bbox"],
        "saveUrl": f"/review/{run_id}/save",
        "sections": (saved or report)["sections"],
        "doctor": (saved or {}).get("doctor", ""),
        "findingsSaved": bool(saved),
        "savedAt": (saved or {}).get("saved_at"),
        "finalHtmlUrl": f"{base}/conclusion_final.html" if final_html.is_file() else None,
        "finalTxtUrl": f"{base}/conclusion_final.txt" if final_html.is_file() else None,
    }
    sections_meta = [
        {
            "key": key,
            "title": report_builder.SECTION_TITLES[key],
            "rows": report_builder.SECTION_ROWS[key],
        }
        for key in report_builder.SECTION_KEYS
    ]
    return render_template("review.html", report=report, cfg=cfg, sections_meta=sections_meta)


@app.route("/review/<run_id>/save", methods=["POST"])
def review_save(run_id):
    run_dir = run_dir_or_404(run_id)
    report = read_json(run_dir / "report.json")
    payload = request.get_json(silent=True) or {}

    try:
        raw_sections = payload.get("sections") or {}
        sections = {}
        for key in report_builder.SECTION_KEYS:
            text = raw_sections.get(key, "")
            if not isinstance(text, str) or len(text) > 20000:
                raise ValueError("Некорректный текст раздела.")
            sections[key] = text.strip()

        doctor = str(payload.get("doctor", ""))[:120].strip()

        mask = decode_png(payload.get("mask"))
        if mask.size != (224, 224):
            raise ValueError("Некорректный размер маски.")
        mask = mask.convert("L").point(lambda v: 255 if v > 127 else 0)

        annotated = decode_png(payload.get("annotated")).convert("RGB")
        if max(annotated.size) > 1000:
            raise ValueError("Некорректный размер изображения.")
    except (ValueError, OSError) as exc:
        return jsonify(ok=False, error=str(exc)), 400

    model_mask_path = run_dir / "mask.png"
    if model_mask_path.is_file():
        before = np.array(Image.open(model_mask_path).convert("L")) > 127
    else:
        before = np.zeros((224, 224), dtype=bool)
    mask_changed = bool((before != (np.array(mask) > 127)).any())

    mask.save(run_dir / "mask_edited.png")
    annotated.save(run_dir / "annotated.png")

    saved_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    review_data = {
        "doctor": doctor,
        "saved_at": saved_at,
        "sections": sections,
        "draft_sections": report["sections"],
        "mask_changed_by_doctor": mask_changed,
    }
    (run_dir / "review.json").write_text(
        json.dumps(review_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image_b64 = base64.b64encode((run_dir / "annotated.png").read_bytes()).decode("ascii")
    document = render_template(
        "final.html",
        report=report,
        sections=sections,
        titles=[(key, report_builder.SECTION_TITLES[key]) for key in report_builder.SECTION_KEYS],
        doctor=doctor,
        saved_at=saved_at,
        image_b64=image_b64,
    )
    (run_dir / "conclusion_final.html").write_text(document, encoding="utf-8")
    (run_dir / "conclusion_final.txt").write_text(
        report_builder.to_text(report, sections, doctor, saved_at), encoding="utf-8-sig"
    )

    base = f"/static/results/{run_id}"
    return jsonify(
        ok=True,
        saved_at=saved_at,
        html_url=f"{base}/conclusion_final.html",
        txt_url=f"{base}/conclusion_final.txt",
    )


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return render_template(
        "index.html",
        error="Слишком большой объём загрузки. Максимум 10 файлов по 10 МБ.",
    ), 413


if __name__ == "__main__":
    ensure_directories()
    app.run(debug=True, use_reloader=True, host="127.0.0.1", port=5000)
