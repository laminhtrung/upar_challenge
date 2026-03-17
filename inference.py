import os
import yaml
import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import postprocess_predictions


CLASS_NAMES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female", "Hair-Length-Short",
    "Hair-Length-Long", "Hair-Length-Bald", "UpperBody-Length-Short", "UpperBody-Color-Black",
    "UpperBody-Color-Blue", "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple", "UpperBody-Color-Red",
    "UpperBody-Color-White", "UpperBody-Color-Yellow", "UpperBody-Color-Other", "LowerBody-Length-Short",
    "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink", "LowerBody-Color-Purple",
    "LowerBody-Color-Red", "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"
]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_transform(image_size):
    h, w = image_size
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def main(config_path, checkpoint_path, image_path):
    cfg = load_yaml(config_path)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["num_classes"],
        pretrained=False,
        dropout=cfg["model"].get("dropout", 0.5),
        num_parts=cfg["model"].get("num_parts", 6),
        reduced_dim=cfg["model"].get("reduced_dim", 256),
    ).to(device)

    model = load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    transform = build_transform(tuple(cfg["data"]["image_size"]))
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs, preds = postprocess_predictions(logits)

    probs = probs[0].cpu().numpy()
    preds = preds[0].cpu().numpy()

    print(f"\nImage: {os.path.basename(image_path)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:35s} pred={int(preds[i])} prob={probs[i]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.image)