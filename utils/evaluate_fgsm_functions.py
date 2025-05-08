import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import pandas as pd

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 正規化と逆正規化関数
transform_norm = transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.261))


def denormalize(img_tensor):
    mean = torch.tensor([0.491, 0.482, 0.447]).view(3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)

def to_numpy_img(tensor_img):
    return tensor_img.permute(1, 2, 0).detach().cpu().numpy()

# 可視化関数（正規化を再適用したうえで推論・SSIM・描画）
def visualize_adversarial_pair(model, adv_data_path, index=0, device='cpu', model_name="Model"):
    model.to(device)
    model.eval()

    # データ読み込み
    data = torch.load(adv_data_path)
    orig = data['original'][index]
    adv = data['adversarial'][index]
    label = data['labels'][index].item()

    # 正規化を再適用（訓練時と同じ処理）
    orig_tensor = transform_norm(orig).unsqueeze(0).to(device)
    adv_tensor = transform_norm(adv).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        pred_orig = model(orig_tensor).argmax(1).item()
        pred_adv = model(adv_tensor).argmax(1).item()

    # 可視化用に正規化を戻す
    orig_img = denormalize(orig)
    adv_img = denormalize(adv)

    # SSIM計算
    ssim_val = ssim(
        to_numpy_img(orig_img),
        to_numpy_img(adv_img),
        channel_axis=2,
        data_range=1.0
    )

    # 描画
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for ax, img, title, pred in zip(
        axs, [orig_img, adv_img], ['Original', 'Adversarial'], [pred_orig, pred_adv]
    ):
        ax.imshow(to_numpy_img(img))
        ax.set_title(f"{title}\nTrue: {classes[label]}\nPred: {classes[pred]}", fontsize=10)
        ax.axis('off')

    plt.suptitle(f"{model_name} | SSIM: {ssim_val:.4f}", fontsize=12)
    plt.tight_layout()
    plt.show()

    
def evaluate_accuracy_on_adv(model, adv_data_path, device='cpu', model_name="UnnamedModel"):
    model.to(device)
    model.eval()

    # 敵対例データの読み込み
    data = torch.load(adv_data_path)
    orig = data['original']  # shape: [N, 3, 32, 32]
    adv = data['adversarial']
    labels = data['labels'].to(device)

    # 訓練時と同じ前処理を適用
    orig_tensor = transform_norm(orig).to(device)
    adv_tensor = transform_norm(adv).to(device)

    # 推論と精度計算
    with torch.no_grad():
        pred_orig = model(orig_tensor).argmax(1)
        pred_adv = model(adv_tensor).argmax(1)

    acc_orig = (pred_orig == labels).float().mean().item()
    acc_adv = (pred_adv == labels).float().mean().item()

    # DataFrameで整形して出力
    df = pd.DataFrame({
        "Model": [model_name],
        "Original Accuracy (%)": [acc_orig * 100],
        "Adversarial Accuracy (%)": [acc_adv * 100],
        "Accuracy Drop (%)": [(acc_orig - acc_adv) * 100]
    }).round(3)
    return df  # 必要なら後続処理でも使える