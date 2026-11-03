from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / 'configs/SpatialProtocol.json'


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() or path.exists() else (ROOT / path).resolve()


def load_protocol(path=None):
    path = resolve_path(path or DEFAULT_PROTOCOL)
    config = json.loads(path.read_text(encoding='utf8'))
    roi = Path(config['train_roi'])
    if not roi.is_absolute():
        roi = path.parent / roi
    if not roi.is_file():
        raise FileNotFoundError(f'Kekexili ROI not found: {roi}')
    if config['boundary_policy'] != 'exclude_crossing_windows':
        raise ValueError('Cross-boundary windows must be excluded')
    if not np.isclose(float(config['development_train_fraction']), .8) or not np.isclose(float(config['development_validation_fraction']), .2):
        raise ValueError('Kekexili development samples must be split 80/20 within each class')
    config['train_roi'] = str(roi.resolve())
    with rasterio.open(roi) as src:
        signature = dict(config=config, roi_grid=dict(crs=str(src.crs), transform=list(src.transform), width=src.width,
            height=src.height, nodata=src.nodata), roi_size=roi.stat().st_size, roi_mtime_ns=roi.stat().st_mtime_ns)
    config['signature'] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode('utf8')).hexdigest()
    return config


def split_development(roles, labels, config):
    roles = np.asarray(roles, dtype=object).copy()
    labels = np.asarray(labels)
    pool = roles == 'development'
    if not np.isin(labels[pool], [0, 1]).all():
        raise ValueError('Kekexili development labels must be binary')
    for label in [0, 1]:
        candidates = np.flatnonzero(pool & (labels == label))
        if len(candidates) < 2:
            raise ValueError(f'Kekexili class {label} needs at least two samples for its separate 80/20 split')
        train, validation = train_test_split(candidates, test_size=float(config['development_validation_fraction']),
            random_state=int(config['spatial_seed']), shuffle=True)
        roles[train] = 'train'
        roles[validation] = 'validation'
    return roles


class Partition:
    def __init__(self, source, protocol=None):
        self.config = load_protocol(protocol)
        self.source = rasterio.open(source)
        if self.source.crs is None:
            raise ValueError('Source raster CRS is required')
        if self.source.transform.b != 0 or self.source.transform.d != 0 or self.source.transform.a <= 0 or self.source.transform.e >= 0:
            raise ValueError('A north-up georeferenced source raster is required')
        self.roi = rasterio.open(self.config['train_roi'])
        self.vrt = WarpedVRT(self.roi, crs=self.source.crs, transform=self.source.transform,
                            width=self.source.width, height=self.source.height,
                            resampling=Resampling.nearest, add_alpha=True)
        self.roi_bounds = transform_bounds(self.roi.crs, self.source.crs, *self.roi.bounds, densify_pts=21)

    def close(self):
        self.vrt.close(); self.roi.close(); self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def window_role(self, window):
        if not isinstance(window, Window):
            window = Window(*window)
        if window.width <= 0 or window.height <= 0 or window.col_off < 0 or window.row_off < 0 or window.col_off + window.width > self.source.width or window.row_off + window.height > self.source.height:
            raise ValueError('Sample crop is outside its georeferenced source raster')
        box = bounds(window, self.source.transform)
        intersects = box[0] < self.roi_bounds[2] and box[2] > self.roi_bounds[0] and box[1] < self.roi_bounds[3] and box[3] > self.roi_bounds[1]
        if intersects:
            roi_valid = self.vrt.read_masks(1, window=window) > 0
            if roi_valid.all():
                role = 'development'
            elif roi_valid.any():
                return 'excluded', 'crosses_kekexili_footprint'
            else:
                role = None
        else:
            role = None
        if not (self.source.read_masks(1, window=window) > 0).all():
            return 'excluded', 'crosses_source_nodata'
        if role == 'development':
            return role, 'kekexili'
        return 'test', 'external_tibetan_plateau'

    def pixel_mask(self, window, subset):
        if subset not in ['train', 'validation', 'test', 'external']:
            raise ValueError('Unknown spatial subset')
        inside = self.vrt.read_masks(1, window=window) > 0
        if subset in ['train', 'validation']:
            return inside
        return ~inside


def segmentation_indices(dataset, config):
    rows = []
    with Partition(dataset.tif_path, config.get('spatial_protocol_file')) as partition:
        signature = partition.config['signature']
        for index, window in enumerate(dataset.windows):
            if not hasattr(window, 'col_off'):
                window = Window(*window)
            role, group = partition.window_role(window)
            label = -1
            if role == 'development':
                mask_function = next((getattr(dataset, name) for name in ['_create_mask_from_shp', '_get_mask_for_window', '_create_mask'] if hasattr(dataset, name)), None)
                if mask_function is None:
                    raise ValueError('Dataset needs a reference-mask function to stratify Kekexili patches')
                label = int(np.asarray(mask_function(window)).any())
            rows.append(dict(index=index, subset=role, spatial_group=group, row=int(window.row_off), col=int(window.col_off),
                             width=int(window.width), height=int(window.height), development_class=label))
        development_config = partition.config
    output = Path(config.get('model_path') or config['model_save_path']).parent
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame['subset'] = split_development(frame.subset.to_numpy(), frame.development_class.to_numpy(), development_config)
    indices = tuple(frame.loc[frame.subset == name, 'index'].to_numpy(int) for name in ['train', 'validation', 'test'])
    if any(len(v) == 0 for v in indices):
        raise ValueError('Need Kekexili 80/20 train/validation patches and other-region test patches; supply the full evaluation mosaic')
    frame.to_csv(output / 'split_indices.csv', index=False, encoding='utf-8-sig')
    audit = dict(protocol_signature=signature, counts=frame.subset.value_counts().to_dict(),
                 split_method='Kekexili lake-containing and background nonoverlapping patches split separately 80/20; all external patches are test only',
                 class_counts=frame.groupby(['subset','development_class']).size().reset_index(name='samples').to_dict('records'),
                 ignored_random_ratio_settings=['small_sample_ratio', 'val_ratio_in_small'], source=str(dataset.tif_path))
    (output / 'spatial_split_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    return indices


def contour_indices(frame, config, id_column=None, label_column=None):
    id_column = id_column or frame.columns[0]
    if not config.get('spatial_samples_file'):
        raise ValueError('spatial_samples_file is required: real sample_id,source_raster,col,row,width,height crop mapping. No random split fallback.')
    path = resolve_path(config['spatial_samples_file'])
    mapping = pd.read_csv(path, dtype={'sample_id': str})
    required = ['sample_id', 'source_raster', 'col', 'row', 'width', 'height']
    if set(required) - set(mapping.columns) or mapping[required].isna().any().any():
        raise ValueError(f'Complete spatial sample mapping required: {required}')
    def norm(v):
        return str(int(v)) if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer() else str(v).strip()
    ids = frame[id_column].map(norm)
    mapping['sample_id'] = mapping.sample_id.map(norm)
    if ids.duplicated().any() or mapping.sample_id.duplicated().any() or set(ids) != set(mapping.sample_id):
        raise ValueError('Spatial mapping and contour table must have identical unique sample IDs')
    mapping = mapping.set_index('sample_id').loc[ids].reset_index()
    mapping['source_raster'] = mapping.source_raster.map(lambda v: str((path.parent / v).resolve()))
    roles = np.full(len(frame), 'excluded', dtype=object); groups = np.full(len(frame), '', dtype=object)
    signature = None
    for source, part in mapping.groupby('source_raster', sort=False):
        with Partition(source, config.get('spatial_protocol_file')) as partition:
            signature = partition.config['signature']
            development_config = partition.config
            for index, row in part.iterrows():
                values = [row[k] for k in ['col', 'row', 'width', 'height']]
                if any(float(v) != int(v) for v in values):
                    raise ValueError('Crop positions and dimensions must be integer native source pixels')
                roles[index], groups[index] = partition.window_role(Window(*map(int, values)))
    if 'lake_id' in mapping:
        for _, part in mapping.loc[mapping.lake_id.notna() & mapping.lake_id.astype(str).str.strip().ne('')].groupby('lake_id'):
            if len(set(roles[part.index]) - {'excluded'}) > 1:
                roles[part.index] = 'excluded'; groups[part.index] = 'same_lake_crosses_partitions'
    labels = frame[label_column].to_numpy() if label_column else frame.iloc[:, -1].to_numpy()
    roles = split_development(roles, labels, development_config)
    result = pd.DataFrame(dict(index=np.arange(len(frame)), sample_id=ids.to_list(), subset=roles, spatial_group=groups,
                              label=labels))
    indices = {name: result.loc[result.subset == name, 'index'].to_numpy(int) for name in ['train', 'validation', 'test']}
    if any(len(indices[k]) == 0 for k in ['train','validation']):
        raise ValueError('Need both classes of Kekexili development contours for the separate 80/20 split')
    if config.get('require_external_contour_test', False) and not len(indices['test']):
        raise ValueError('External labelled contour test explicitly required but not supplied')
    audit = dict(protocol_signature=signature, spatial_samples_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 counts=result.subset.value_counts().to_dict(),
                 class_counts=result.groupby(['subset','label']).size().reset_index(name='samples').to_dict('records'),
                 split_method='Kekexili lake and nonlake contours independently split 80/20 then combined; all external contours are test only',
                 external_contour_test_supplied=bool(len(indices['test'])),
                 lake_id_check='development/test crossing lakes excluded before internal contour split; same lake may occur in Kekexili train/validation' if 'lake_id' in mapping else 'lake identity mapping absent; same lake may occur in Kekexili train/validation')
    return indices, result, audit


def evaluation_subset_mask(source, subset='test', protocol=None):
    with Partition(source, protocol) as partition:
        result = np.zeros((partition.source.height, partition.source.width), bool)
        for y in range(0, result.shape[0], 256):
            for x in range(0, result.shape[1], 256):
                window = Window(x, y, min(256, result.shape[1]-x), min(256, result.shape[0]-y))
                result[y:y+int(window.height), x:x+int(window.width)] = partition.pixel_mask(window, subset)
        return result, partition.config['signature']
