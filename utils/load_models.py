import torch 
from models.CNN import CNN
from models.ResNet18 import get_resnet18_for_cifar10


def get_CNN():
    model_path = "../models/pretrained/CNN_cifar10.pth"
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_CNN_small():
    model_path = "../models/pretrained/CNN_cifar10_small.pth"
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_resnet():
    model_path = "../models/pretrained/resnet18_cifar10.pth"
    model = get_resnet18_for_cifar10()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def set_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")