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


def segment_image(input_path, output_path):
    with Image.open(input_path) as img:
        image = img.convert("L")

    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = preprocess_image(arr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()

    base = (tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)
    m = pred >= 0.5
    rgb[m] = (0.5 * rgb[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(rgb).resize((448, 448), Image.Resampling.BILINEAR).save(output_path)
    return output_path