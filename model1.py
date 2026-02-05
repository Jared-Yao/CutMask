import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet18

from sspcab_torch import *



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class ProjectionNet(nn.Module):

    def __init__(self, pretrained=True, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=2):
        super(ProjectionNet, self).__init__()

        self.resnet18 = resnet18(pretrained=pretrained)

        self.se_block = SEBlock(channels=512)  # Add SE block to ResNet18
        self.sspcab_block = SSPCAB(channels=512)  # Add SSPCAB block

        # Build the MLP head
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        self.head = nn.Sequential(*sequential_layers)
        self.resnet18.fc = nn.Identity()  # Remove the original ResNet fully-connected layer
        self.out = nn.Linear(last_layer, num_classes)  # Output layer with num_classes dimensions
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x):
        # Forward pass through the convolutional layers of ResNet18
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        # Apply SSPCAB block and compute the auxiliary loss
        x_sspcab = self.sspcab_block(x)
        # Ensure cost_sspcab is a scalar to avoid shape mismatch issues
        cost_sspcab = self.mse_loss(x_sspcab, x).mean()

        # Generate embedding vectors
        embeds = F.adaptive_avg_pool2d(x_sspcab, (1, 1)).flatten(start_dim=1)

        # Generate logits
        tmp = self.head(embeds)
        logits = self.out(tmp)  # Expected shape: (batch_size, num_classes)
        
        return embeds, logits, cost_sspcab



    def freeze_resnet(self):
        # Freeze the full ResNet18
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # Unfreeze the head:
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # Unfreeze all parameters:
        for param in self.parameters():
            param.requires_grad = True
