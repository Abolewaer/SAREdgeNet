import math

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


INPUT_SPEC = {'channels': ['SAR', 'DEM'], 'sar_scale': 255.0,
              'dem_scale_metres': 9000.0, 'dem_clip_metres': [-500.0, 9000.0],
              'dem_nodata_fill_normalized': 0.0, 'alignment': 'SAR grid, bilinear DEM'}


class SarDemDatasetMixin:
    def __init__(self, *args, dem_path=None, **kwargs):
        if not dem_path:
            raise ValueError('SAR+DEM requires dem_path during both training and prediction')
        self.dem_path = str(dem_path)
        self._dem_src = None
        self._dem_vrt = None
        with rasterio.open(self.dem_path) as dem:
            if dem.count != 1 or dem.crs is None:
                raise ValueError('DEM must be a single georeferenced elevation band in metres')
        super().__init__(*args, **kwargs)
        with rasterio.open(self.tif_path) as sar:
            if sar.count != 1 or sar.crs is None:
                raise ValueError('SAR must be a single georeferenced band on the [0,255] scale')

    def __getstate__(self):
        state = super().__getstate__()
        state['_dem_src'] = None
        state['_dem_vrt'] = None
        return state

    def __getitem__(self, idx):
        window = self.windows[idx]
        sar_src = self._get_raster_source()
        if self._dem_vrt is None:
            self._dem_src = rasterio.open(self.dem_path)
            self._dem_vrt = WarpedVRT(self._dem_src, crs=self.crs,
                                      transform=self.transform_matrix, width=self.width,
                                      height=self.height, resampling=Resampling.bilinear,
                                      dtype='float32', nodata=float('nan'))
        sar = sar_src.read(1, window=window, masked=True).astype(np.float32).filled(0)
        dem = self._dem_vrt.read(1, window=window, masked=True).filled(np.nan)
        if not np.isfinite(sar).all() or np.any((sar < 0) | (sar > 255)):
            raise ValueError('SAR values must be finite and within [0,255]')
        sar = sar / 255.0
        dem = np.clip(dem, -500.0, 9000.0) / 9000.0
        dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
        image = torch.from_numpy(np.stack([sar, dem]).astype(np.float32))
        mask = torch.from_numpy(self._get_mask_for_window(window)).long()
        if self.transform:
            image = self.transform(image)
        if self.mode == 'train':
            return image, mask
        info = {'col': window.col_off, 'row': window.row_off,
                'width': window.width, 'height': window.height}
        return image, mask, info

    def close(self):
        for name in ('_dem_vrt', '_dem_src', '_src'):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.close()
                setattr(self, name, None)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        groups = math.gcd(32, out_channels)
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   padding=kernel_size // 2, bias=False),
                         nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True))


class ResNet50Encoder(nn.Module):
    def __init__(self, in_channels=2, dilated=False):
        super().__init__()
        if in_channels != 2:
            raise ValueError('This revision requires SAR+DEM (2 input channels)')
        backbone = resnet50(weights=None, replace_stride_with_dilation=[False, dilated, dilated])
        backbone.conv1 = nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4

    def forward(self, x):
        c2 = self.layer1(self.stem(x))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class PSPNet(nn.Module):
    architecture_id = 'PSPNet_ResNet50_SARDEM_ppm512_head536_v2'

    def __init__(self, in_channels=2, n_classes=2):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels, dilated=True)
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(size),
                                                ConvNormAct(2048, 512, 1))
                                  for size in (1, 2, 3, 6)])
        self.head = nn.Sequential(ConvNormAct(4096, 536), nn.Dropout2d(0.1),
                                  nn.Conv2d(536, n_classes, 1))

    def forward(self, x):
        features = self.encoder(x)[-1]
        pyramid = [features] + [F.interpolate(stage(features), size=features.shape[-2:],
                                               mode='bilinear', align_corners=False)
                                 for stage in self.ppm]
        logits = self.head(torch.cat(pyramid, dim=1))
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)


class FPN(nn.Module):
    architecture_id = 'FPN_ResNet50_SARDEM_pyramid256_head176_v2'

    def __init__(self, in_channels=2, n_classes=2):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels)
        self.laterals = nn.ModuleList([nn.Conv2d(channels, 256, 1)
                                       for channels in (256, 512, 1024, 2048)])
        self.smooth = nn.ModuleList([ConvNormAct(256, 256) for _ in range(4)])
        self.head = nn.Sequential(ConvNormAct(1024, 176), nn.Dropout2d(0.1),
                                  nn.Conv2d(176, n_classes, 1))

    def forward(self, x):
        features = self.encoder(x)
        pyramid = [layer(feature) for layer, feature in zip(self.laterals, features)]
        for index in range(2, -1, -1):
            pyramid[index] = pyramid[index] + F.interpolate(pyramid[index + 1],
                                  size=pyramid[index].shape[-2:], mode='bilinear', align_corners=False)
        pyramid = [layer(feature) for layer, feature in zip(self.smooth, pyramid)]
        fused = torch.cat([F.interpolate(feature, size=pyramid[0].shape[-2:],
                                        mode='bilinear', align_corners=False)
                           for feature in pyramid], dim=1)
        logits = self.head(fused)
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)


def checkpoint_metadata(model):
    return {'architecture_id': model.architecture_id, 'input_spec': INPUT_SPEC,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'initialization': 'random; no ImageNet or historical checkpoint by default'}


def validate_checkpoint(checkpoint, model):
    metadata = checkpoint.get('model_metadata')
    if metadata != checkpoint_metadata(model):
        raise ValueError('Checkpoint architecture/input metadata mismatch: retrain the SAR+DEM v2 model')
