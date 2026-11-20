import os

import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    CropForeground,
    Resize,
    SpatialPad,
    ScaleIntensity,
    EnsureType,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(
    BASE_DIR, "models", "segmentation_model.pt"
)

IMAGE_SIZE = 224
VIEW_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model():
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Не найден файл весов сегментации: {WEIGHTS_PATH}"
        )

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation="sigmoid",
    )

    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


model = create_model()

preprocess_image = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    CropForeground(
        select_fn=lambda x: x > 0.04,
        margin=0,
    ),
    Resize(
        spatial_size=IMAGE_SIZE,
        size_mode="longest",
        mode="area",
    ),
    SpatialPad(
        spatial_size=(IMAGE_SIZE, IMAGE_SIZE),
        method="symmetric",
        mode="constant",
    ),
    ScaleIntensity(minv=0.0, maxv=1.0),
    EnsureType(dtype=torch.float32),
])


def _frame(input_path):
    """Предобработанный кадр 224×224 в виде тензора [1, 1, 224, 224]."""
    with Image.open(input_path) as img:
        image = img.convert("L")
    array = np.array(image, dtype=np.float32) / 255.0
    return preprocess_image(array).unsqueeze(0).to(DEVICE)


def _save_view(tensor, view_path):
    """Сохраняет кадр для просмотра (448×448) и возвращает его в виде массива 224×224."""
    base = np.clip(tensor[0, 0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    Image.fromarray(base).resize(
        (VIEW_SIZE, VIEW_SIZE), resample=Image.Resampling.BILINEAR
    ).save(view_path)
    return base


def prepare_view(input_path, view_path):
    """Кадр для просмотра без сегментации (для снимков без найденной опухоли)."""
    os.makedirs(os.path.dirname(os.path.abspath(view_path)), exist_ok=True)
    _save_view(_frame(input_path), view_path)
    return view_path


def segment_image(input_path, output_path):
    """Строит маску и сохраняет рядом с output_path три файла в одной системе координат (кадр 224×224):

    output.png  кадр 448×448 с красной подсветкой области;
    view.png    кадр 448×448 без подсветки (для страницы проверки);
    mask.png    бинарная маска 224×224 (255 = область).
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    tensor = _frame(input_path)

    with torch.no_grad():
        prediction = model(tensor)[0, 0].cpu().numpy()

    mask = prediction >= 0.5

    base = _save_view(tensor, os.path.join(out_dir, "view.png"))
    Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(out_dir, "mask.png"))

    rgb = np.stack([base] * 3, axis=-1)
    red = np.array([255, 0, 0])
    rgb[mask] = (0.5 * rgb[mask] + 0.5 * red).astype(np.uint8)
    Image.fromarray(rgb).resize(
        (VIEW_SIZE, VIEW_SIZE), resample=Image.Resampling.NEAREST
    ).save(output_path)

    return output_path
