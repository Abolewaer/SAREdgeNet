import numpy as np
from scipy import ndimage as ndi
from rasterio.transform import Affine

from .raster import read, write, align, same_grid


def preprocess(source, output):
    data, profile, valid = read(source)
    image = data[0].astype(np.float32)
    valid &= np.isfinite(image) & (image != 0)
    normalized = np.zeros(image.shape, np.uint8)
    if valid.any():
        values = image[valid]
        clipped = np.minimum(values, values.mean() + 2 * values.std())
        low, high = clipped.min(), clipped.max()
        if high > low:
            normalized[valid] = np.clip(255 * (clipped-low)/(high-low),0,255).astype(np.uint8)
    result = ndi.gaussian_filter(normalized,1.0).astype(np.uint8)
    result = ndi.median_filter(result,size=3).astype(np.uint8)
    result = ndi.grey_erosion(result,size=(3,3)).astype(np.uint8)
    result = ndi.grey_dilation(result,size=(3,3)).astype(np.uint8)
    result[~valid] = 0
    write(output, result, profile, valid)


def geolocation(source, output, dx=0.0, dy=0.0, target_crs=None, resolution=20.0):
    data, profile, valid = read(source)
    if profile['crs'] is None:
        raise ValueError('Source CRS is required; supply real georeferencing first')
    t = profile['transform']
    profile['transform'] = Affine(t.a,t.b,t.c+dx,t.d,t.e,t.f+dy)
    if target_crs:
        from rasterio.warp import calculate_default_transform,reproject,Resampling
        from rasterio.transform import array_bounds
        bounds = array_bounds(profile['height'],profile['width'],profile['transform'])
        transform,width,height = calculate_default_transform(profile['crs'],target_crs,
                                                               profile['width'],profile['height'],*bounds,resolution=resolution)
        result = np.zeros((data.shape[0],height,width),data.dtype)
        for index,band in enumerate(data):
            reproject(band,result[index],src_transform=profile['transform'],src_crs=profile['crs'],
                      dst_transform=transform,dst_crs=target_crs,resampling=Resampling.bilinear)
        mask = np.zeros((height,width),np.uint8)
        reproject(valid.astype(np.uint8),mask,src_transform=profile['transform'],src_crs=profile['crs'],
                  dst_transform=transform,dst_crs=target_crs,resampling=Resampling.nearest)
        profile.update(crs=target_crs,transform=transform,width=width,height=height)
        data,valid = result,mask>0
    write(output, data, profile, valid)


def aligned_dem(source, reference, output):
    """Align elevation metres and normalize over the reference processing extent.

    This is not per-contour normalization. Dataset validity is retained separately,
    so a valid minimum mapped to zero is not automatically treated as missing.
    """
    align(source, reference, output)
    data, profile, valid = read(output)
    elevation = data[0].astype(np.float64)
    valid &= np.isfinite(elevation)
    normalized = np.zeros(elevation.shape, dtype=np.float32)
    if valid.any():
        values = elevation[valid]
        mean, std = values.mean(), values.std()
        clipped = np.clip(values, mean - 2 * std, mean + 2 * std)
        low, high = clipped.min(), clipped.max()
        if high > low:
            normalized[valid] = (255.0 * (clipped - low) / (high - low)).astype(np.float32)
    write(output, normalized, profile, valid)


def attach_grid(source,reference,output):
    data,profile,valid = read(source)
    _,rp,rv = read(reference)
    if (profile['width'],profile['height']) != (rp['width'],rp['height']):
        raise ValueError('Cannot attach grid to different image dimensions')
    if rp['crs'] is None:
        raise ValueError('Reference CRS is required')
    write(output,data,rp,valid & rv)


class AlignedDEM:
    def __init__(self, dem_path, sar_path):
        dem, dp, valid = read(dem_path)
        _, sp, _ = read(sar_path)
        same_grid(dp, sp)
        self.dem = dem[0]
        self.valid = valid & np.isfinite(self.dem)

    def calculate_dem_statistics_for_region(self, region_mask):
        values = self.dem[region_mask & self.valid].astype(np.float64)
        if values.size:
            q1,q3 = np.percentile(values,[25,75])
            values = values[(values >= q1-1.5*(q3-q1)) & (values <= q3+1.5*(q3-q1))]
        if not values.size:
            return dict(mean=0.0,std=0.0,median=0.0,min=0.0,max=0.0,count=0)
        return dict(mean=round(float(values.mean()),2),std=round(float(values.std()),2),
                    median=round(float(np.median(values)),2),min=round(float(values.min()),2),
                    max=round(float(values.max()),2),count=int(values.size))
