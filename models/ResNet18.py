from torchvision.models import resnet18
import torch.nn as nn

def get_resnet18_for_cifar10(num_classes=10):
    model = resnet18(num_classes=num_classes)
    
    # CIFAR-10（32×32画像）に合わせてconv1とmaxpoolを修正
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    return model
