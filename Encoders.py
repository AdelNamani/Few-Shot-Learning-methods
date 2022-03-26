import torch
from torch import nn
import torchvision.models as models

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def Encoder(hyperparametrs):
    x_dim = hyperparametrs['x_dim']
    hid_dim = hyperparametrs['hid_dim']
    z_dim = hyperparametrs['z_dim']
    name = hyperparametrs['encoder']
    if name == 'convnet':
        encoder = nn.Sequential(
            conv_block(x_dim[0], hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        ) 
    elif name == 'resnet18':
        encoder = models.resnet18(pretrained=True)
        dim_representation = encoder.fc.in_features
        encoder.fc = nn.Sequential(
            nn.Linear(dim_representation, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim),
            Flatten()
        )
    elif name == 'wideresnet':
        encoder = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        dim_representation = encoder.fc.in_features
        encoder.fc = nn.Sequential(
            nn.Linear(dim_representation, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim),
            Flatten()
        )
    elif name == 'resnet34':
        encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        dim_representation = encoder.fc.in_features
        encoder.fc = nn.Sequential(
            nn.Linear(dim_representation, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim),
            Flatten()
        )
    else:
        raise ValueError('Wrong value for encoder!')
    return encoder
