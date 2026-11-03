import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .sar_dem_baselines import ConvNormAct


class DoubleConv(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1),
                         nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1),
                         nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True))


class AttentionGate(nn.Module):
    def __init__(self, channels, intermediate):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels, intermediate, 1), nn.BatchNorm2d(intermediate))
        self.skip = nn.Sequential(nn.Conv2d(channels, intermediate, 1), nn.BatchNorm2d(intermediate))
        self.score = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(intermediate, 1, 1),
                                   nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, gate, skip):
        return skip * self.score(self.gate(gate) + self.skip(skip))


class UNet(nn.Module):
    architecture_id = 'UNet_SAR_width80_bottleneck1456_v2'
    input_spec = {'channels': ['SAR'], 'normalization': 'legacy dtype-dependent /255 or /65535'}

    def __init__(self, in_channels=1, n_classes=2, bottleneck=1456, attention=False):
        super().__init__()
        if in_channels != 1:
            raise ValueError('This U-Net revision uses one SAR channel')
        widths = [80, 160, 320, 640, bottleneck]
        self.encoder = nn.ModuleList([DoubleConv(incoming, outgoing)
                                      for incoming, outgoing in zip([in_channels] + widths[:-1], widths)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[index + 1], widths[index], 2, stride=2)
                                 for index in range(3, -1, -1)])
        self.decoder = nn.ModuleList([DoubleConv(2 * widths[index], widths[index])
                                      for index in range(3, -1, -1)])
        if attention:
            self.attention = nn.ModuleList([AttentionGate(widths[index], widths[index] // 2)
                                           for index in range(3, -1, -1)])
        else:
            self.attention = None
        self.output = nn.Conv2d(widths[0], n_classes, 1)

    def forward(self, x):
        original_shape = x.shape[-2:]
        skips = []
        for index, layer in enumerate(self.encoder):
            x = layer(self.pool(x) if index else x)
            skips.append(x)
        for index, (up, decoder) in enumerate(zip(self.up, self.decoder)):
            x = up(x)
            skip = skips[-index - 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            if self.attention is not None:
                skip = self.attention[index](x, skip)
            x = decoder(torch.cat([x, skip], dim=1))
        return F.interpolate(self.output(x), size=original_shape, mode='bilinear', align_corners=False)


class AttentionUNet(UNet):
    architecture_id = 'AttentionUNet_SAR_width80_bottleneck1540_v2'
    input_spec = {'channels': ['SAR'], 'normalization': 'legacy raw float32 SAR'}

    def __init__(self, in_channels=1, n_classes=2):
        super().__init__(in_channels, n_classes, bottleneck=1540, attention=True)


class AtrousConv(nn.Sequential):
    def __init__(self, incoming, outgoing, rate):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=rate, dilation=rate, bias=False),
                         nn.GroupNorm(32, outgoing), nn.ReLU(inplace=True))


class DeepLabV3Plus(nn.Module):
    architecture_id = 'DeepLabV3Plus_ResNet50_SARDEM_aspp256_decoder240_v2'
    input_spec = {'channels': ['SAR', 'DEM'], 'normalization': 'legacy raw float32 SAR and DEM metres; prealigned grids'}

    def __init__(self, in_channels=2, n_classes=2, output_stride=16, pretrained=False):
        super().__init__()
        if in_channels != 2 or output_stride != 16:
            raise ValueError('This DeepLabV3+ revision requires SAR+DEM and output_stride=16')
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
                            replace_stride_with_dilation=[False, False, True])
        first = nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                first.weight.copy_(backbone.conv1.weight.mean(1, keepdim=True).repeat(1, 2, 1, 1))
        self.stem = nn.Sequential(first, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.aspp = nn.ModuleList([ConvNormAct(2048, 256, 1),
                                  *[AtrousConv(2048, 256, rate) for rate in (6, 12, 18)],
                                  nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvNormAct(2048, 256, 1))])
        self.aspp_project = nn.Sequential(ConvNormAct(1280, 256, 1), nn.Dropout2d(0.1))
        self.low_project = ConvNormAct(256, 48, 1)
        self.decoder = nn.Sequential(ConvNormAct(304, 240), ConvNormAct(240, 240),
                                      nn.Conv2d(240, n_classes, 1))

    def forward(self, x):
        low = self.layer1(self.stem(x))
        high = self.layer4(self.layer3(self.layer2(low)))
        branches = [F.interpolate(layer(high), size=high.shape[-2:], mode='bilinear',
                                   align_corners=False) for layer in self.aspp]
        high = self.aspp_project(torch.cat(branches, dim=1))
        high = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.decoder(torch.cat([high, self.low_project(low)], dim=1))
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)


def model_metadata(model):
    return {'architecture_id': model.architecture_id, 'input_spec': model.input_spec,
            'model_parameters': sum(parameter.numel() for parameter in model.parameters())}


def validate_model_checkpoint(checkpoint, model):
    if checkpoint.get('model_metadata') != model_metadata(model):
        raise ValueError('Checkpoint does not match this v2 architecture/input; retraining is required')


def validate_dem_grid(sar_path, dem_path):
    import rasterio
    import numpy as np
    with rasterio.open(sar_path) as sar, rasterio.open(dem_path) as dem:
        if sar.count != 1 or dem.count != 1 or sar.crs is None or dem.crs != sar.crs:
            raise ValueError('DeepLabV3+ requires one SAR band and one aligned DEM band')
        if sar.shape != dem.shape or not np.allclose(tuple(sar.transform), tuple(dem.transform), rtol=0, atol=1e-9):
            raise ValueError('dem_preprocessed must be aligned to the SAR grid before training or prediction')
