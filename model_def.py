import torch
import torch.nn as nn
from torchvision import models
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    CropForeground,
    ScaleIntensityRangePercentiles,
    Resize,
    SpatialPad,
    EnsureType,
)
from monai.data import PILReader


CLASSES = [
    "Опухоли нет",
    "Гипофиз",
    "Глиома",
    "Менингиома",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model():
    model = models.resnet34(weights=None)

    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 4)

    return model


preprocess = Compose([
    LoadImage(
        image_only=True,
        reader=PILReader(converter=lambda im: im.convert('L')),
    ),
    EnsureChannelFirst(channel_dim="no_channel"),
    CropForeground(select_fn=lambda x: x > 10),
    ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
    Resize(spatial_size=224, size_mode="longest", mode="area"),
    SpatialPad(spatial_size=(224, 224,), method="symmetric"),
    EnsureType(dtype=torch.float32),
])
