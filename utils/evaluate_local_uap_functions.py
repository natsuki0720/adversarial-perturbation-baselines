import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import torch

def generate_mask(patch_size=(8, 8), image_shape=(3, 32, 32)):
    """
    patch_size: (height, width) of the patch to activate (e.g., (8, 8))
    image_shape: (C, H, W) shape of the image (default CIFAR-10)
    
    Returns:
        mask: Tensor of shape [1, C, H, W] with 1s in patch region, 0s elsewhere
    """
    mask = torch.zeros(image_shape)
    h, w = patch_size
    mask[:, :h, :w] = 1.0
    return mask.unsqueeze(0)  # shape: [1, C, H, W]


# Local UAP適用関数
def apply_local_uap(images, delta, mask):
    return (images + delta * mask).clamp(0, 1)

# SSIMを1画像ペア間で計算（RGB平均）
def calculate_ssim(img1, img2):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return np.mean([
        ssim(img1_np[:, :, i], img2_np[:, :, i], data_range=1.0)
        for i in range(3)
    ])

# Local UAP評価関数（Accuracy + SSIM）
import pandas as pd
import numpy as np

def evaluate_local_uap(model, delta, mask, test_loader, device, 
                       model_name="UnnamedModel", attack_name="Local UAP", max_ssim_samples=100):
    model.eval()
    correct = 0
    total = 0
    ssim_vals = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = apply_local_uap(images, delta.to(device), mask.to(device))

            outputs = model(adv_images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if len(ssim_vals) < max_ssim_samples:
                for i in range(min(len(images), max_ssim_samples - len(ssim_vals))):
                    ssim_val = calculate_ssim(images[i], adv_images[i])
                    ssim_vals.append(ssim_val)

    acc = correct / total
    avg_ssim = np.mean(ssim_vals) if ssim_vals else float("nan")

    # 表形式で結果出力
    df = pd.DataFrame({
        "Model": [model_name],
        "Attack": [attack_name],
        "Accuracy (%)": [round(acc * 100, 2)],
        "SSIM": [round(avg_ssim, 4)],
        "Accuracy Drop (%)": ["–"]  # 必要なら後で比較して追加
    })

    return df

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def visualize_local_uap_prediction(model, delta, mask, dataset, index, device,
                                   model_name="Model", class_names=None):
    model.eval()

    # データ取得
    image, label = dataset[index]
    image = image.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)

    # 攻撃適用
    adv_image = apply_local_uap(image, delta.to(device), mask.to(device))

    # 推論
    with torch.no_grad():
        output_clean = model(image)
        output_adv = model(adv_image)
        pred_clean = output_clean.argmax(dim=1).item()
        pred_adv = output_adv.argmax(dim=1).item()

    # SSIM計算（正規化前の画像で）
    ssim_val = calculate_ssim(image[0], adv_image[0])

    # ラベル表示用
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # 可視化（正規化済み画像をそのまま表示）
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(image[0].permute(1, 2, 0).detach().cpu())
    axs[0].set_title(f"Original\nTrue: {class_names[label.item()]}\nPred: {class_names[pred_clean]}")

    axs[1].imshow(adv_image[0].permute(1, 2, 0).detach().cpu())
    axs[1].set_title(f"Adversarial\nPred: {class_names[pred_adv]}")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(f"{model_name} | SSIM: {ssim_val:.4f}", fontsize=12)
    plt.tight_layout()
    plt.show()

