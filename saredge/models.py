from __future__ import annotations
import torch
from torch import nn
from torchvision import models

class FeatureAttention(nn.Module):

    def __init__(self, channels: int=128, reduction: int=8) -> None:
        super().__init__()
        hidden = channels // reduction
        self.gate = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid())
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x * (1.0 + self.gate(x)))

class ImageBranch(nn.Module):

    def __init__(self, pretrained: bool=True, backbone_name: str='resnet34') -> None:
        super().__init__()
        if backbone_name not in ['resnet18','resnet34']:
            raise ValueError('Unsupported backbone')
        enum = models.ResNet34_Weights if backbone_name=='resnet34' else models.ResNet18_Weights
        weights = enum.DEFAULT if pretrained else None
        backbone = getattr(models,backbone_name)(weights=weights)
        rgb_weight = backbone.conv1.weight.detach().mean(dim=1, keepdim=True)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight.copy_(rgb_weight)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(inplace=True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(image))

class NumericalBranch(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(17, 128)
        self.linear2 = nn.Linear(128, 128)
        self.attention = FeatureAttention(128, reduction=8)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, 32)
        self.activation = nn.GELU()
        self.dropouts = nn.ModuleList([nn.Dropout(0.2), nn.Dropout(0.2), nn.Dropout(0.15), nn.Dropout(0.1), nn.Identity()])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropouts[0](self.activation(self.linear1(features)))
        x = self.dropouts[1](self.activation(self.linear2(x)))
        x = self.attention(x)
        x = self.dropouts[2](self.activation(self.linear3(x)))
        x = self.dropouts[3](self.activation(self.linear4(x)))
        return self.dropouts[4](self.activation(self.linear5(x)))

class SAREdgeNetContourClassifier(nn.Module):

    def __init__(self, pretrained: bool=True) -> None:
        super().__init__()
        self.image_branch = ImageBranch(pretrained=pretrained)
        self.numerical_branch = NumericalBranch()
        self.classifier = nn.Sequential(nn.Linear(96, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        image_features = self.image_branch(image)
        numerical_features = self.numerical_branch(features)
        return self.classifier(torch.cat((image_features, numerical_features), dim=1))
