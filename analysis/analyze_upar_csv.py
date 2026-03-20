import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def analyze_upar_csv(
    csv_path,
    output_dir="analysis_upar",
    skiprows=1,
    has_image_column=True,
    clamp_values=(5, 10, 15)
):
    os.makedirs(output_dir, exist_ok=True)

    # Đọc CSV
    df = pd.read_csv(csv_path, skiprows=skiprows, header=None)

    if has_image_column:
        labels_df = df.iloc[:, 1:].copy()
    else:
        labels_df = df.copy()

    labels_df = labels_df.astype(np.float32)

    num_samples = len(labels_df)
    num_classes = labels_df.shape[1]

    if num_classes != len(CLASS_NAMES):
        print(f"[Warning] Số class trong CSV = {num_classes}, nhưng CLASS_NAMES = {len(CLASS_NAMES)}")
        class_names = [f"class_{i}" for i in range(num_classes)]
    else:
        class_names = CLASS_NAMES

    pos_counts = labels_df.sum(axis=0).values
    neg_counts = num_samples - pos_counts
    pos_rate = pos_counts / num_samples
    neg_rate = neg_counts / num_samples

    raw_pos_weight = neg_counts / (pos_counts + 1e-6)

    result_df = pd.DataFrame({
        "class_id": np.arange(num_classes),
        "class_name": class_names,
        "num_positive": pos_counts.astype(int),
        "num_negative": neg_counts.astype(int),
        "positive_rate": pos_rate,
        "negative_rate": neg_rate,
        "raw_pos_weight": raw_pos_weight
    })

    # Thêm các cột clamp
    for c in clamp_values:
        result_df[f"pos_weight_clamp_{c}"] = np.minimum(raw_pos_weight, c)
        result_df[f"is_clamped_at_{c}"] = raw_pos_weight > c

    # Sắp xếp theo độ hiếm
    rare_df = result_df.sort_values(by="positive_rate", ascending=True).reset_index(drop=True)
    weight_df = result_df.sort_values(by="raw_pos_weight", ascending=False).reset_index(drop=True)

    # In summary
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"Num samples: {num_samples}")
    print(f"Num classes: {num_classes}")
    print("=" * 80)

    print("\n[1] Thống kê raw_pos_weight")
    print(f"min  : {result_df['raw_pos_weight'].min():.4f}")
    print(f"max  : {result_df['raw_pos_weight'].max():.4f}")
    print(f"mean : {result_df['raw_pos_weight'].mean():.4f}")
    print(f"median: {result_df['raw_pos_weight'].median():.4f}")

    print("\n[2] Số class bị clamp")
    for c in clamp_values:
        n = int((result_df["raw_pos_weight"] > c).sum())
        print(f"Clamp {c:>2}: {n}/{num_classes} class")

    print("\n[3] Top 10 class hiếm nhất")
    print(
        rare_df[[
            "class_id", "class_name", "num_positive", "positive_rate", "raw_pos_weight"
        ]].head(10).to_string(index=False)
    )

    print("\n[4] Top 10 class có raw_pos_weight lớn nhất")
    print(
        weight_df[[
            "class_id", "class_name", "num_positive", "positive_rate", "raw_pos_weight"
        ]].head(10).to_string(index=False)
    )

    # Nhóm class theo positive rate
    bins = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 1.0]
    labels = ["<1%", "1-3%", "3-5%", "5-10%", "10-20%", ">=20%"]
    result_df["rarity_group"] = pd.cut(
        result_df["positive_rate"], bins=bins, labels=labels, include_lowest=True
    )

    print("\n[5] Số class theo nhóm positive rate")
    print(result_df["rarity_group"].value_counts(dropna=False).sort_index())

    # Lưu file CSV
    result_df.to_csv(os.path.join(output_dir, "label_distribution_full.csv"), index=False)
    rare_df.to_csv(os.path.join(output_dir, "label_distribution_sorted_by_rarity.csv"), index=False)
    weight_df.to_csv(os.path.join(output_dir, "label_distribution_sorted_by_raw_pos_weight.csv"), index=False)

    # Plot 1: positive rate
    plt.figure(figsize=(16, 6))
    plt.bar(range(num_classes), result_df["positive_rate"].values)
    plt.xticks(range(num_classes), result_df["class_name"].values, rotation=90)
    plt.ylabel("Positive Rate")
    plt.title("Positive Rate per Class")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "positive_rate_per_class.png"))
    plt.close()

    # Plot 2: raw pos weight
    plt.figure(figsize=(16, 6))
    plt.bar(range(num_classes), result_df["raw_pos_weight"].values)
    plt.xticks(range(num_classes), result_df["class_name"].values, rotation=90)
    plt.ylabel("Raw Pos Weight")
    plt.title("Raw Pos Weight per Class")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_pos_weight_per_class.png"))
    plt.close()

    # Plot 3: compare clamp
    plt.figure(figsize=(16, 6))
    x = np.arange(num_classes)
    plt.plot(x, result_df["raw_pos_weight"].values, label="raw_pos_weight")
    for c in clamp_values:
        plt.plot(x, result_df[f"pos_weight_clamp_{c}"].values, label=f"clamp_{c}")
    plt.xticks(range(num_classes), result_df["class_name"].values, rotation=90)
    plt.ylabel("Pos Weight")
    plt.title("Raw vs Clamped Pos Weight")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_vs_clamped_pos_weight.png"))
    plt.close()

    # Gợi ý nhanh
    suggestion = {
        "num_classes_clamped_at_5": int((result_df["raw_pos_weight"] > 5).sum()),
        "num_classes_clamped_at_10": int((result_df["raw_pos_weight"] > 10).sum()),
        "num_classes_clamped_at_15": int((result_df["raw_pos_weight"] > 15).sum()),
        "raw_pos_weight_mean": float(result_df["raw_pos_weight"].mean()),
        "raw_pos_weight_max": float(result_df["raw_pos_weight"].max()),
    }

    print("\n[6] Suggestion raw")
    print(suggestion)

    if suggestion["num_classes_clamped_at_10"] <= 5:
        print("=> Gợi ý ban đầu: clamp=10 có vẻ ổn.")
    elif suggestion["num_classes_clamped_at_10"] <= 12:
        print("=> Gợi ý ban đầu: clamp=10 hoặc 15 đều đáng thử.")
    else:
        print("=> Gợi ý ban đầu: có khá nhiều class bị chạm trần 10, nên thử thêm clamp=15.")

    return result_df


if __name__ == "__main__":
    csv_path = "/root/trunglm8/upar_challenge/raw_data/train.csv"   # sửa đường dẫn nếu cần

    analyze_upar_csv(
        csv_path=csv_path,
        output_dir="analysis_upar_train",
        skiprows=1,
        has_image_column=True,
        clamp_values=(5, 10, 15)
    )