import torch

from model_def import create_model, preprocess, CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    image = preprocess(image_path)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = []

        for model in models_ensemble:
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            probabilities.append(probs)

        mean_probs = torch.stack(probabilities).mean(dim=0)

    predicted_index = int(mean_probs.argmax(dim=1).item())

    return {
        "class": CLASSES[predicted_index],
        "confidence": float(mean_probs[0, predicted_index].item()),
        "probabilities": {
            CLASSES[i]: float(mean_probs[0, i].item())
            for i in range(len(CLASSES))
        }
    }