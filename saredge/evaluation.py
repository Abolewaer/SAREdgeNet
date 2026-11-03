import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .raster import read, binary, same_grid, metre_grid


def reference_mask(path, profile):
    path = Path(path)
    if path.suffix.lower() in ['.shp','.gpkg','.geojson','.json']:
        import geopandas as gpd
        frame = gpd.read_file(path)
        if frame.crs is None or profile['crs'] is None:
            raise ValueError('Reference and prediction require CRS')
        frame = frame.to_crs(profile['crs'])
        geometries = [(g, 1) for g in frame.geometry if g is not None and not g.is_empty]
        shape = (profile['height'], profile['width'])
        mask = rasterize(geometries, out_shape=shape, transform=profile['transform'], fill=0) > 0 if geometries else np.zeros(shape, bool)
        return mask, np.ones(shape, bool)
    data, rp, valid = read(path)
    same_grid(profile, rp)
    return data[0] > 0, valid


def largest(mask):
    labels, count = ndi.label(mask, np.ones((3,3)))
    if not count:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def boundary_scores(prediction, reference, valid, profile):
    prediction, reference = largest(prediction), largest(reference)
    safe = ndi.binary_erosion(valid, np.ones((3,3)), border_value=1)
    pe = (prediction & ~ndi.binary_erosion(prediction, np.ones((3,3)), border_value=0)) & safe
    re = (reference & ~ndi.binary_erosion(reference, np.ones((3,3)), border_value=0)) & safe
    p, r = np.argwhere(pe), np.argwhere(re)
    if not len(p) or not len(r):
        return {'boundary_precision':0.0, 'boundary_recall':0.0, 'boundary_f1':0.0,
                'hausdorff_km':None, 'boundary_status':'empty_prediction_or_reference'}
    dp = cKDTree(r).query(p)[0]
    dr = cKDTree(p).query(r)[0]
    bp, br = float((dp <= 3).mean()), float((dr <= 3).mean())
    return {'boundary_precision':bp, 'boundary_recall':br,
            'boundary_f1':2*bp*br/(bp+br) if bp+br else 0.0,
            'hausdorff_km':float(max(dp.max(), dr.max()) * 20 / 1000), 'boundary_status':'ok'}


def metrics(counts):
    tp, fp, fn, tn = map(int, counts)
    divide = lambda a,b: a/b if b else 0.0
    precision, recall = divide(tp,tp+fp), divide(tp,tp+fn)
    lake, background = divide(tp,tp+fp+fn), divide(tn,tn+fp+fn)
    return {'TP':tp, 'FP':fp, 'FN':fn, 'TN':tn, 'precision':precision, 'recall':recall,
            'f1':divide(2*tp,2*tp+fp+fn), 'lake_iou':lake,
            'background_iou':background, 'miou':(lake+background)/2}


def evaluate_case(case, boundary=False):
    data, profile, valid = read(case['prediction'])
    metre_grid(profile)
    prediction = binary(data, case.get('positive_value',1), case.get('rgb_red',False))
    if case.get('validity_source'):
        _,vp,vv = read(case['validity_source'])
        same_grid(profile,vp)
        valid = vv if case.get('rgb_red',False) else (valid & vv)
    reference, rv = reference_mask(case['reference'], profile)
    valid &= rv
    if case.get('evaluation_mask'):
        ed, ep, ev = read(case['evaluation_mask'])
        same_grid(profile, ep)
        valid &= ev & (ed[0] > 0)
    spatial_signature = None
    if case.get('spatial_subset'):
        if case['spatial_subset'] == 'validation' and not case.get('evaluation_mask'):
            raise ValueError('Kekexili validation is sample-level; provide its saved evaluation_mask instead of treating the entire lake district as validation')
        from .spatial_protocol import evaluation_subset_mask
        spatial_valid, spatial_signature = evaluation_subset_mask(case.get('validity_source') or case['prediction'],
            case['spatial_subset'], case.get('spatial_protocol_file'))
        valid &= spatial_valid
    if not valid.any():
        raise ValueError('No valid evaluation pixels')
    counts = [np.sum(prediction & reference & valid), np.sum(prediction & ~reference & valid),
              np.sum(~prediction & reference & valid), np.sum(~prediction & ~reference & valid)]
    result = {'name':case.get('name',Path(case['prediction']).stem), **metrics(counts),
              'valid_pixels':int(valid.sum()), 'area_km2':float((prediction & valid).sum()*0.0004)}
    identity = hashlib.sha256()
    grid = {'crs':profile['crs'].to_string(),'shape':list(valid.shape),
            'transform':list(profile['transform'])[:6]}
    identity.update(json.dumps(grid,sort_keys=True).encode('utf-8'))
    identity.update(np.packbits(valid).tobytes())
    identity.update(np.packbits(reference & valid).tobytes())
    result['evaluation_signature'] = identity.hexdigest()
    result['spatial_subset'] = case.get('spatial_subset')
    result['spatial_protocol_signature'] = spatial_signature
    if boundary:
        result.update(boundary_scores(prediction & valid, reference & valid, valid, profile))
    return result


def evaluate(cases, output, boundary=False):
    rows = [evaluate_case(case, boundary) for case in cases]
    if not rows:
        raise ValueError('No evaluation cases')
    total = np.sum([[row[k] for k in ['TP','FP','FN','TN']] for row in rows], axis=0)
    report = {'region':metrics(total), 'cases':rows}
    if boundary:
        hd = [r['hausdorff_km'] for r in rows if r['hausdorff_km'] is not None]
        report['boundary_mean'] = {'boundary_f1':float(np.mean([r['boundary_f1'] for r in rows])),
                                   'hausdorff_km':float(np.mean(hd)) if hd else None,
                                   'hausdorff_valid_cases':len(hd), 'cases':len(rows), 'tolerance_pixels':3}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main(default_rgb=False, require_validity_source=False, spatial_default=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction')
    parser.add_argument('--reference')
    parser.add_argument('--manifest')
    parser.add_argument('--positive-value', type=int, default=1)
    parser.add_argument('--rgb-red', action='store_true', default=default_rgb)
    parser.add_argument('--boundary', action='store_true')
    parser.add_argument('--evaluation-mask')
    parser.add_argument('--validity-source')
    parser.add_argument('--spatial-subset', choices=['validation','test','external'], default=spatial_default)
    parser.add_argument('--spatial-protocol')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.manifest:
        cases = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    else:
        if not args.prediction or not args.reference:
            parser.error('Provide --prediction and --reference or --manifest')
        if require_validity_source and not args.validity_source:
            parser.error('Supply --validity-source SAR.tif to distinguish valid black background from NoData')
        cases = [{'prediction':args.prediction, 'reference':args.reference, 'positive_value':args.positive_value,
                  'rgb_red':args.rgb_red, 'evaluation_mask':args.evaluation_mask,'validity_source':args.validity_source}]
    for case in cases:
        if args.spatial_subset:
            case['spatial_subset'] = args.spatial_subset
            case['spatial_protocol_file'] = args.spatial_protocol
    print(json.dumps(evaluate(cases, args.output, args.boundary)['region'], indent=2))
