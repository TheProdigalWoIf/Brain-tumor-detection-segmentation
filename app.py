import csv
import shutil
import uuid
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from classifier import predict as classify_image
from segmenter import segment_image


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
}


def ensure_directories():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_segmentation(input_path, output_path):
    segment_image(str(input_path), str(output_path))


def save_classification_csv(csv_path, result):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["field", "class", "value"])
        writer.writerow(["predicted_class", result["class"], ""])
        writer.writerow([
            "confidence",
            "",
            result["confidence"],
        ])

        for class_name, probability in result["probabilities"].items():
            writer.writerow([
                "class_probability",
                class_name,
                probability,
            ])

        writer.writerow([
            "segmentation_output",
            "",
            "output.png",
        ])


@app.route("/")
def index():
    ensure_directories()
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    ensure_directories()

    # 1. Проверяем, что файл был отправлен.
    if "file" not in request.files:
        return render_template(
            "index.html",
            error="Файл не был загружен.",
        ), 400

    file = request.files["file"]

    if not file.filename:
        return render_template(
            "index.html",
            error="Сначала выберите изображение.",
        ), 400

    # 2. Проверяем расширение.
    safe_name = secure_filename(file.filename)
    extension = Path(safe_name).suffix.lower()

    if extension not in ALLOWED_FORMATS:
        return render_template(
            "index.html",
            error="Допустимые форматы: PNG, JPG и JPEG.",
        ), 400

    run_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{run_id}{extension}"
    result_dir = RESULTS_DIR / run_id

    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        file.save(upload_path)

        try:
            with Image.open(upload_path) as image:
                actual_format = image.format
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                "Файл не является корректным изображением."
            ) from exc

        expected_format = ALLOWED_FORMATS[extension]

        if actual_format != expected_format:
            raise ValueError(
                "Содержимое файла не соответствует его расширению."
            )

        input_preview = result_dir / "input.png"

        with Image.open(upload_path) as image:
            image.convert("RGB").save(input_preview, format="PNG")

        classification_result = classify_image(str(upload_path))

        output_path = result_dir / "output.png"
        run_segmentation(upload_path, output_path)

        if not output_path.is_file():
            raise RuntimeError(
                "Сегментационная модель не сохранила output.png."
            )

        csv_path = result_dir / "data.csv"
        save_classification_csv(
            csv_path,
            classification_result,
        )

        with open(csv_path, encoding="utf-8", newline="") as f:
            csv_content = f.read()

        return render_template(
            "index.html",
            result_url=f"/static/results/{run_id}",
            csv_content=csv_content,
            result=classification_result,
            probabilities=classification_result["probabilities"].items(),
        )

    except ValueError as exc:
        shutil.rmtree(result_dir, ignore_errors=True)

        return render_template(
            "index.html",
            error=str(exc),
        ), 400

    except Exception:
        app.logger.exception("Ошибка при обработке изображения")
        shutil.rmtree(result_dir, ignore_errors=True)

        return render_template(
            "index.html",
            error=(
                "Не удалось обработать изображение. "
                "Проверьте подключение моделей и логи сервера."
            ),
        ), 500

    finally:
        upload_path.unlink(missing_ok=True)


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return render_template(
        "index.html",
        error="Файл слишком большой. Максимальный размер — 10 МБ.",
    ), 413


if __name__ == "__main__":
    ensure_directories()
    app.run(debug=True, host="127.0.0.1", port=5000)