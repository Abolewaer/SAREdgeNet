
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'saredge/models.py').is_file())
sys.path.insert(0, str(ROOT))
from saredge.models import ImageBranch


class NumericBranch(nn.Module):
    def __init__(self, dimension, attention):
        super().__init__()
        self.linear1 = nn.Linear(dimension, 128)
        self.linear2 = nn.Linear(128, 128)
        self.gate = nn.Sequential(nn.Linear(128, 16), nn.GELU(), nn.Linear(16, 128), nn.Sigmoid()) if attention else None
                                                                               
        self.norm = nn.LayerNorm(128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, 32)
        self.activation = nn.GELU()
        self.dropouts = nn.ModuleList([nn.Dropout(.2), nn.Dropout(.2), nn.Dropout(.15), nn.Dropout(.1), nn.Identity()])

    def forward(self, features):
        x = self.dropouts[0](self.activation(self.linear1(features)))
        x = self.dropouts[1](self.activation(self.linear2(x)))
        x = self.norm(x * (1 + self.gate(x)) if self.gate is not None else x)
        x = self.dropouts[2](self.activation(self.linear3(x)))
        x = self.dropouts[3](self.activation(self.linear4(x)))
        return self.dropouts[4](self.activation(self.linear5(x)))


class Classifier(nn.Module):
    def __init__(self, variant, weights_file=None):
        super().__init__()
                                                                                  
        self.image_branch = ImageBranch(pretrained=False, backbone_name='resnet34')
        if weights_file is not None:
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            if 'conv1.weight' not in state or state['conv1.weight'].shape[1] != 3:
                raise ValueError('Expected original torchvision RGB ResNet-34 ImageNet state_dict')
            state = dict(state)
            state['conv1.weight'] = state['conv1.weight'].mean(dim=1, keepdim=True)
            state.pop('fc.weight', None); state.pop('fc.bias', None)
            self.image_branch.backbone.load_state_dict(state, strict=True)
        self.numeric = NumericBranch(len(variant['columns']), variant['attention']) if variant['columns'] else None
        n = 96 if self.numeric is not None else 64
        self.classifier = nn.Sequential(nn.Linear(n, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(.5),
                                        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Dropout(.3), nn.Linear(32, 2))

    def forward(self, images, numeric):
        x = self.image_branch(images)
        if self.numeric is not None:
            x = torch.cat((x, self.numeric(numeric)), dim=1)
        return self.classifier(x)
