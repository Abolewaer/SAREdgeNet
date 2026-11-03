from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def read(path):
    with rasterio.open(path) as src:
        data = src.read()
        return data, src.profile.copy(), src.dataset_mask() > 0


def write(path, data, profile, valid=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    profile = profile.copy()
    profile.update(driver='GTiff', height=data.shape[1], width=data.shape[2],
                   count=data.shape[0], dtype=str(data.dtype), nodata=None, compress='deflate')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
        if valid is not None:
            dst.write_mask(valid.astype(np.uint8) * 255)


def same_grid(a, b):
    if any(a[k] != b[k] for k in ['width','height','crs','transform']):
        raise ValueError('Raster grids differ; align before processing')


def metre_grid(profile, resolution=20):
    t, crs = profile['transform'], profile['crs']
    if crs is None or not crs.is_projected or crs.linear_units != 'metre':
        raise ValueError('A projected CRS in metres is required')
    if not np.allclose([t.a, t.b, t.d, abs(t.e)], [resolution, 0, 0, resolution]):
        raise ValueError(f'A {resolution} metre, north-up grid is required')


def align(source, reference, output, categorical=False):
    _, profile, valid = read(reference)
    if profile['crs'] is None:
        raise ValueError('Reference CRS is required')
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError('Source CRS is required')
        data = src.read(1).astype(np.float32)
        data[src.read_masks(1) == 0] = np.nan
        result = np.full((profile['height'], profile['width']), np.nan, np.float32)
        reproject(data, result, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=profile['transform'], dst_crs=profile['crs'],
                  dst_nodata=np.nan, resampling=Resampling.nearest if categorical else Resampling.bilinear)
    write(output, result, profile, valid & np.isfinite(result))


def binary(data, positive=1, rgb_red=False):
    if rgb_red:
        if data.shape[0] < 3:
            raise ValueError('RGB input needs at least three bands')
        return (data[0] >= 200) & (data[1] <= 50) & (data[2] <= 50)
    return data[0] == positive


def preserve_prediction_validity(output,source):
    with rasterio.open(source) as src:
        valid = src.dataset_mask()
        profile = src.profile.copy()
    with rasterio.open(output,'r+') as dst:
        same_grid(profile,dst.profile)
        dst.nodata = None
        dst.write_mask(valid)
