import os
import yaml
import argparse
import random
import torch

from PIL import Image, ImageDraw, ImageFont
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


def get_image_list(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for fname in os.listdir(image_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in exts:
            image_paths.append(os.path.join(image_dir, fname))
    image_paths.sort()
    return image_paths


def group_positive_classes(preds, probs):
    grouped = {
        "Age": [],
        "Gender": [],
        "Hair": [],
        "UpperBody": [],
        "LowerBody": [],
        "Accessory": []
    }

    for i, cls_name in enumerate(CLASS_NAMES):
        if int(preds[i]) != 1:
            continue

        prob = probs[i]

        if cls_name.startswith("Age-"):
            grouped["Age"].append((cls_name.replace("Age-", ""), prob))

        elif cls_name.startswith("Gender-"):
            grouped["Gender"].append((cls_name.replace("Gender-", ""), prob))

        elif cls_name.startswith("Hair-"):
            grouped["Hair"].append((cls_name.replace("Hair-", ""), prob))

        elif cls_name.startswith("UpperBody-"):
            grouped["UpperBody"].append((cls_name.replace("UpperBody-", ""), prob))

        elif cls_name.startswith("LowerBody-"):
            grouped["LowerBody"].append((cls_name.replace("LowerBody-", ""), prob))

        elif cls_name.startswith("Accessory-"):
            grouped["Accessory"].append((cls_name.replace("Accessory-", ""), prob))

    return grouped


def build_text_lines(grouped):
    lines = []
    for cat, items in grouped.items():
        if len(items) == 0:
            continue
        lines.append(f"{cat}:")
        for name, prob in items:
            lines.append(f"- {name} ({prob:.2f})")
        lines.append("")
    if len(lines) == 0:
        lines = ["No positive class predicted"]
    return lines


def draw_result_image(image, text_lines, save_path):
    image = image.convert("RGB")
    w, h = image.size

    font = ImageFont.load_default()
    line_height = 16
    top_margin = 10
    left_margin = 10

    # đo độ rộng vùng text
    tmp_draw = ImageDraw.Draw(image)
    max_text_width = 0
    for line in text_lines:
        bbox = tmp_draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        max_text_width = max(max_text_width, text_w)

    panel_width = max(260, max_text_width + 20)
    new_h = max(h, top_margin * 2 + len(text_lines) * line_height)

    canvas = Image.new("RGB", (w + panel_width, new_h), color=(255, 255, 255))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)

    x_text = w + left_margin
    y_text = top_margin

    for line in text_lines:
        fill = (0, 0, 0)
        if line.endswith(":"):
            fill = (0, 0, 255)
        draw.text((x_text, y_text), line, fill=fill, font=font)
        y_text += line_height

    canvas.save(save_path)


def main(config_path, checkpoint_path, image_dir, num_images=20, random_sample=False):
    cfg = load_yaml(config_path)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]["name"]

    model = build_model(
        model_name=model_name,
        num_classes=cfg["num_classes"],
        pretrained=False,
        dropout=cfg["model"].get("dropout", 0.5),
        num_parts=cfg["model"].get("num_parts", 6),
        reduced_dim=cfg["model"].get("reduced_dim", 256),
    ).to(device)

    model = load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    transform = build_transform(tuple(cfg["data"]["image_size"]))

    image_paths = get_image_list(image_dir)
    if len(image_paths) == 0:
        print(f"[ERROR] Không tìm thấy ảnh trong thư mục: {image_dir}")
        return

    if random_sample:
        selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))
    else:
        selected_paths = image_paths[:min(num_images, len(image_paths))]

    save_dir = os.path.join("test_result", model_name, "vis_20")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Tổng số ảnh tìm thấy: {len(image_paths)}")
    print(f"[INFO] Số ảnh sẽ infer và lưu: {len(selected_paths)}")
    print(f"[INFO] Output dir: {save_dir}")

    for idx, image_path in enumerate(selected_paths):
        image_name = os.path.basename(image_path)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Không mở được ảnh {image_name}: {e}")
            continue

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs, preds = postprocess_predictions(logits)

        probs = probs[0].cpu().numpy()
        preds = preds[0].cpu().numpy()

        grouped = group_positive_classes(preds, probs)
        text_lines = build_text_lines(grouped)

        print(f"\n[{idx + 1}/{len(selected_paths)}] {image_name}")
        for line in text_lines:
            print(line)

        save_path = os.path.join(save_dir, image_name)
        draw_result_image(image, text_lines, save_path)

    print(f"\n[✔] Đã lưu ảnh vào: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--random_sample", action="store_true")
    args = parser.parse_args()

    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        image_dir=args.image_dir,
        num_images=args.num_images,
        random_sample=args.random_sample
    )