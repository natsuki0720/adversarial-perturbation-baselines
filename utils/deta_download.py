import torchvision
import torchvision.transforms as transforms
import os

# 保存先ディレクトリの設定
save_dir = "data"

# データ変換の定義（標準化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10用の平均・分散
])

# データセットのダウンロード＆保存
trainset = torchvision.datasets.CIFAR10(
    root=save_dir,
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root=save_dir,
    train=False,
    download=True,
    transform=transform
)


