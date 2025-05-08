import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd

normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010))
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
def evaluate_uap(model, dataloader, delta=None, device="cpu", model_name="UnnamedModel", attack_name="UAP"):
    model = model.to(device)
    model.eval()

    correct_orig = 0
    correct_adv = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        total += y.size(0)

        # 正常データでの予測
        x_norm = normalize(x)
        with torch.no_grad():
            pred_orig = model(x_norm).argmax(1)
        correct_orig += (pred_orig == y).sum().item()

        # 摂動ありの予測
        if delta is not None:
            x_adv = torch.clamp(x + delta.to(device), 0, 1)
            x_adv_norm = normalize(x_adv)
            with torch.no_grad():
                pred_adv = model(x_adv_norm).argmax(1)
            correct_adv += (pred_adv == y).sum().item()
        else:
            correct_adv = correct_orig  # same if no attack

    acc_orig = correct_orig / total
    acc_adv = correct_adv / total

    # DataFrame出力
    df = pd.DataFrame({
        "Model": [model_name],
        "Attack": [attack_name],
        "Original Accuracy (%)": [round(acc_orig * 100, 2)],
        "Adversarial Accuracy (%)": [round(acc_adv * 100, 2)],
        "Accuracy Drop (%)": [round((acc_orig - acc_adv) * 100, 2)]
    })

    return df

def denormalize(img_tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).to(img_tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1).to(img_tensor.device)
    return img_tensor * std + mean

def to_numpy_img(t):
    return np.transpose(t.detach().cpu().numpy(), (1,2,0))
 
def visualize_uap_effect(model, delta, dataset, index=0,device = "cpu",model_name="Model",class_names = CIFAR10_CLASSES):
    model.eval()
    x, y = dataset[index]
    x = x.unsqueeze(0).to(next(model.parameters()).device)
    y = torch.tensor([y]).to(x.device)

    x_adv = torch.clamp(x + delta, 0, 1)
    x_norm = normalize(x[0]).unsqueeze(0)
    x_adv_norm = normalize(x_adv[0]).unsqueeze(0)
    with torch.no_grad():
        pred_orig = model(x_norm).argmax(1).item()
        pred_adv = model(x_adv_norm).argmax(1).item()

    x_vis = denormalize(x[0])
    x_adv_vis = denormalize(x_adv[0])
    ssim_val = ssim(to_numpy_img(x_vis), to_numpy_img(x_adv_vis), channel_axis=2, data_range=1.0)


    # クラス名処理
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # 可視化
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(to_numpy_img(x[0].detach().cpu()))
    axs[0].set_title(f"Original\nTrue: {class_names[y.item()]}\nPred: {class_names[pred_orig]}")

    axs[1].imshow(to_numpy_img(x_adv[0].detach().cpu()))
    axs[1].set_title(f"Adversarial\nPred: {class_names[pred_adv]}")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(f"{model_name} | SSIM: {ssim_val:.4f}")
    plt.tight_layout()
    plt.show() 

def plot_ssim_distribution(dataset, delta, max_samples=10000, device="cpu"):

    ssim_values = []

    for i in range(min(len(dataset), max_samples)):
        x, _ = dataset[i]
        x = x.unsqueeze(0).to(device)
        x_adv = torch.clamp(x + delta, 0, 1)

        x_vis = denormalize(x[0])
        x_adv_vis = denormalize(x_adv[0])
        ssim_val = ssim(to_numpy_img(x_vis), to_numpy_img(x_adv_vis), channel_axis=2, data_range=1.0)
        ssim_values.append(ssim_val)

    # ヒストグラムを描画
    plt.figure(figsize=(8, 5))
    plt.hist(ssim_values, bins=40, range=(0.8, 1.0), color='skyblue', edgecolor='black')
    plt.title("SSIM Distribution: Original vs Adversarial")
    plt.xlabel("SSIM Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"平均 SSIM: {np.mean(ssim_values):.4f}，標準偏差: {np.std(ssim_values):.4f}")
    

def plot_ssim_vs_correctness(model, dataset, delta, max_samples=1000, device="cpu"):
    model.eval()
    ssim_list = []
    correct_list = []

    for i in range(min(len(dataset), max_samples)):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([y]).to(device)

        x_adv = torch.clamp(x + delta, 0, 1)

        # 正規化（モデル入力用）
        normalize = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3, affine=False, track_running_stats=False)  # Dummy for placeholder
        )
        x_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x[0]).unsqueeze(0)
        x_adv_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x_adv[0]).unsqueeze(0)

        with torch.no_grad():
            pred_adv = model(x_adv_norm.to(device)).argmax(1)
            correct = int(pred_adv == y)

        # SSIMは視覚的な類似性の評価
        ssim_val = ssim(
            to_numpy_img(denormalize(x[0])),
            to_numpy_img(denormalize(x_adv[0])),
            channel_axis=2,
            data_range=1.0
        )

        ssim_list.append(ssim_val)
        correct_list.append(correct)

    # 散布図を描画
    plt.figure(figsize=(8, 5))
    plt.scatter(ssim_list, correct_list, c=['blue' if c == 1 else 'red' for c in correct_list], alpha=0.6)
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.xlabel("SSIM")
    plt.ylabel("Classification Result")
    plt.title("SSIM vs Classification Correctness (Adversarial Samples)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"平均 SSIM（誤分類時）: {np.mean([s for s, c in zip(ssim_list, correct_list) if c == 0]):.4f}")
    print(f"平均 SSIM（正解時）: {np.mean([s for s, c in zip(ssim_list, correct_list) if c == 1]):.4f}")



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