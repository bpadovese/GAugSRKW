import torch.nn as nn
import torchvision.models as models
from diffusers import UNet2DModel

def resnet18_for_single_channel():
    model = models.resnet18(weights=None)

    # Modifying the input layer
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modifying the fully connected layer for 2 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

def unet(sample_size=128, channels=1):
    model = UNet2DModel(
        sample_size=sample_size,  # image resolution
        in_channels=channels,  
        out_channels=channels,  
        layers_per_block=2,  #  ResNet layers per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  
        down_block_types=(
            "DownBlock2D",  
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  #  self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  
            "AttnUpBlock2D",  # self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model