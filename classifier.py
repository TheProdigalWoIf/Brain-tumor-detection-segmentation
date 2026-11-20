import torch

from model_def import create_model, preprocess, CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TUMOR_THRESHOLD = 0.1
UNCERTAIN_BELOW = 0.9
TUMOR_UNCERTAIN_BELOW = 0.9
TYPE_UNCERTAIN_BELOW = 0.9


def load_model(path):
    model = create_model().to(device)

    weights = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    return model


models_ensemble = [
    load_model(f"models/best_model_fold_{i}.pt")
    for i in range(1, 6)
]


def predict(image_path):
    image = preprocess(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = [
            torch.softmax(model(image), dim=1)
            for model in models_ensemble
        ]
        mean_probs = torch.stack(probabilities).mean(dim=0)

    probs = mean_probs[0]

    # 1. Есть ли опухоль
    p_tumor = float(1 - probs[0].item())
    has_tumor = p_tumor >= TUMOR_THRESHOLD

    # 2. Какой тип
    type_probs = probs[1:] / (probs[1:].sum() + 1e-9)
    tp = [float(x) for x in type_probs]
    type_conf = max(tp)

    if has_tumor:
        predicted_index = tp.index(type_conf) + 1
    else:
        predicted_index = 0

    confidence = float(probs[predicted_index].item())

    # 3. Причины сомнения
    reasons = []
    if has_tumor and p_tumor < TUMOR_UNCERTAIN_BELOW:
        reasons.append(
            f"наличие опухоли подтверждено не уверенно (вероятность {p_tumor * 100:.0f}%)"
        )
    if has_tumor and type_conf < TYPE_UNCERTAIN_BELOW:
        order = sorted(range(3), key=lambda i: tp[i], reverse=True)
        first, second = order[0], order[1]
        reasons.append(
            f"тип опухоли сомнительный: {CLASSES[first + 1]} {tp[first] * 100:.0f}%, "
            f"{CLASSES[second + 1]} {tp[second] * 100:.0f}%"
        )

    return {
        "class": CLASSES[predicted_index],
        "confidence": confidence,
        "probabilities": {
            CLASSES[i]: float(probs[i].item())
            for i in range(len(CLASSES))
        },
        "p_tumor": p_tumor,
        "type_probabilities": (
            {CLASSES[i + 1]: tp[i] for i in range(3)} if has_tumor else None
        ),
        "uncertain": confidence < UNCERTAIN_BELOW,
        "reasons": reasons,
    }