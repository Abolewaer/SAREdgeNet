import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .raster import read,write,same_grid,binary


def seed_grow(contour_path,source,output,max_intensity=50.0,tolerance=30.0):
    if not 0 <= max_intensity <= 255 or not 0 <= tolerance <= 255:
        raise ValueError('Intensity thresholds must be in [0,255]')
    data,profile,valid = read(contour_path)
    image,sp,sv = read(source)
    same_grid(profile,sp)
    gray = image[0].astype(np.float32)
    valid &= sv & np.isfinite(gray)
    if valid.any() and (gray[valid].min() < 0 or gray[valid].max() > 255):
        raise ValueError('Seed growth requires the preprocessed [0,255] SAR image')
    contour = binary(data,rgb_red=True) & valid
    domain = ndi.binary_fill_holes(contour) & ~contour & valid
    labels,count = ndi.label(domain)
    grown = np.zeros(contour.shape,bool)
    seeds = []
    for identifier,slices in enumerate(ndi.find_objects(labels),1):
        region = labels[slices] == identifier
        local_gray = gray[slices]
        points = np.argwhere(region)
        values = local_gray[region]
        if not values.size or values.min() > max_intensity:
            continue
        seed_y,seed_x = points[values.argmin()]
        seed_value = float(local_gray[seed_y,seed_x])
        permitted = region & (local_gray <= max_intensity) & (abs(local_gray-seed_value) <= tolerance)
        seed = np.zeros(region.shape,bool)
        seed[seed_y,seed_x] = True
        result = ndi.binary_propagation(seed,mask=permitted)
        grown[slices] |= result
        seeds.append({'region':identifier,'row':int(slices[0].start+seed_y),'col':int(slices[1].start+seed_x),
                      'intensity':seed_value,'grown_pixels':int(result.sum())})
    rgb = np.repeat(gray.astype(np.uint8)[None],3,axis=0)
    mask = contour | grown
    rgb[:,mask] = np.array([255,0,0],np.uint8)[:,None]
    write(output,rgb,profile,valid)
    report = {'enclosed_regions':count,'seeds':seeds,'grown_pixels':int(grown.sum()),
              'contour_pixels':int(contour.sum()),'max_intensity':max_intensity,'tolerance':tolerance,
              'connectivity':4,'growth_domain':'enclosed retained contour interiors'}
    Path(output).with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
