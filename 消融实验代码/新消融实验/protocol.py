
from pathlib import Path
import csv
import hashlib
import json

FEATURES = [
    'Edge Pixel Count', 'Bounding Box Length(px)', 'Bounding Box Width(px)',
    'L/W + W/L', 'Curvature Feature', 'Roughness', 'Pixel Mean', 'Pixel Std',
    'Pixel Median', 'DEM Mean', 'DEM Std', 'DEM Median', 'DEM Min', 'DEM Max',
    'Valid DEM Pixels', 'Compactness', 'Extent',
]
GEOMETRY = FEATURES[:6] + FEATURES[15:17]
SAR_STATS = FEATURES[6:9]
DEM = FEATURES[9:15]
                                                                                 
VARIANTS = {
    'A_image_only': {'columns': [], 'attention': False},
    'B_image_dem': {'columns': DEM, 'attention': False},
    'C_image_geometry': {'columns': GEOMETRY, 'attention': False},
    'D_image_dem_geometry': {'columns': [c for c in FEATURES if c not in SAR_STATS], 'attention': False},
    'E_full_attention': {'columns': FEATURES, 'attention': True},
    'F_full_no_attention': {'columns': FEATURES, 'attention': False},
}
SEEDS = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
                                                                               
    for key in ['table', 'image_dir', 'groups_file', 'output_dir', 'pretrained_weights', 'spatial_samples_file', 'spatial_protocol_file', 'external_scene_ids_file']:
        if cfg.get(key):
            cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg['_config_path'] = str(path)
    return cfg


def require_confirmation(cfg):
    if cfg.get('image_content') not in ['sar_grayscale_patch', 'binary_contour']:
        raise ValueError('Confirm image_content: sar_grayscale_patch OR binary_contour')
    if cfg.get('split_mode') != 'kekexili_external':
        raise ValueError('Only kekexili_external split is supported by the updated protocol')
    if not cfg.get('spatial_samples_file'):
        raise ValueError('Provide real contour crop coordinates in spatial_samples_file')


def identifier(value):
    if value is None or str(value).strip() == '':
        raise ValueError('Empty sample identifier')
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def prepare(cfg):

    import numpy as np
    import pandas as pd
    require_confirmation(cfg)
    path = Path(cfg['table'])
    frame = pd.read_excel(path) if path.suffix.lower() in ['.xls', '.xlsx'] else pd.read_csv(path, dtype={cfg['id_column']: str})
    source_names = cfg['feature_columns']
    if set(source_names) != set(FEATURES) or len(set(source_names.values())) != 17:
        raise ValueError('feature_columns must map all 17 canonical names to 17 distinct source columns')
    id_col, label_col = cfg['id_column'], cfg['label_column']
    if id_col == label_col or set(source_names.values()) & {id_col, label_col}:
        raise ValueError('ID, label and feature columns must be distinct')
    required = [id_col, label_col, *source_names.values()]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f'Missing table columns: {missing}')
    if frame[id_col].isna().any() or frame[label_col].isna().any():
        raise ValueError('Sample IDs and labels must be complete')
    data = pd.DataFrame({'sample_id': frame[id_col].map(identifier), 'label': frame[label_col]})
    if data['sample_id'].duplicated().any() or not data['label'].isin([0, 1]).all():
        raise ValueError('Unique sample IDs and binary labels 0/1 are required; do not concatenate scene-local IDs without a unique mapping')
    data['label'] = data['label'].astype(int)
    for canonical, column in source_names.items():
        data[canonical] = pd.to_numeric(frame[column], errors='raise')
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('All 17 source features must be finite; no implicit DEM imputation is permitted')
    image_column = cfg.get('image_path_column')
    if image_column:
        if image_column not in frame or frame[image_column].isna().any():
            raise ValueError('image_path_column is missing or incomplete')
        data['image_path'] = frame[image_column].map(lambda v: str((path.parent / str(v)).resolve()))
    else:
        if not cfg.get('image_dir'):
            raise ValueError('Provide image_dir or image_path_column')
        data['image_path'] = data.sample_id.map(lambda v: str(Path(cfg['image_dir']) / (v + '.png')))
    missing_images = [v for v in data.image_path if not Path(v).is_file()]
    if missing_images:
        raise FileNotFoundError(f'{len(missing_images)} contour images missing; first: {missing_images[:3]}')
    import sys
    repository = next(p for p in Path(__file__).resolve().parents if (p / 'saredge/spatial_protocol.py').is_file())
    sys.path.insert(0, str(repository))
    from saredge.spatial_protocol import contour_indices
    indices, split, spatial_audit = contour_indices(data, cfg, id_column='sample_id', label_column='label')
    mapping = pd.read_csv(cfg['spatial_samples_file'], dtype={'sample_id': str})
    if 'scene_id' in mapping:
        data = data.merge(mapping[['sample_id', 'scene_id']], on='sample_id', validate='one_to_one', sort=False)
    protocol = dict(table_sha256=digest(path), feature_columns=source_names,
        image_content=cfg['image_content'], split_mode='kekexili_external',
        spatial_audit=spatial_audit, require_external_contour_test=bool(cfg.get('require_external_contour_test', False)), image_paths_sha256=hashlib.sha256('\n'.join(data.image_path).encode('utf8')).hexdigest())
    root = Path(cfg['output_dir']); root.mkdir(parents=True, exist_ok=True)
    split_path = root / 'shared_split.csv'; meta_path = root / 'shared_split_metadata.json'
    split = split[['sample_id', 'subset']]
    if split_path.exists() or meta_path.exists():
        if not split_path.exists() or not meta_path.exists():
            raise ValueError('Incomplete shared spatial split; use a fresh output directory')
        saved = json.loads(meta_path.read_text(encoding='utf8'))
        previous = pd.read_csv(split_path, dtype={'sample_id': str})
        if saved['protocol'] != protocol or saved['split_sha256'] != digest(split_path):
            raise ValueError('Protocol/data mismatch; previous random splits cannot be reused')
        if not previous.sort_values('sample_id').reset_index(drop=True).equals(split.sort_values('sample_id').reset_index(drop=True)):
            raise ValueError('Shared spatial split differs')
    else:
        split.sort_values('sample_id').to_csv(split_path, index=False, encoding='utf-8-sig')
        dump(meta_path, dict(protocol=protocol, split_sha256=digest(split_path)))
    data = data.merge(split, on='sample_id', validate='one_to_one', sort=False)
    for subset in ['train', 'validation']:
        if set(data.loc[data.subset == subset, 'label']) != {0, 1}:
            raise ValueError(f'{subset} needs both classes under the real geographic partition')
    manifest = root / 'shared_manifest.csv'
    data.sort_values('sample_id').to_csv(manifest, index=False, encoding='utf-8-sig')
    report = dict(protocol=protocol, split_sha256=digest(split_path), manifest_sha256=digest(manifest),
        counts=data.groupby(['subset', 'label']).size().reset_index(name='samples').to_dict('records'),
        independence='Kekexili lake/nonlake contours separately split 80/20 for train/validation; all external contours are test only; crossing crops excluded',
        image_bytes_hashed=False)
    dump(root / 'data_audit.json', report)
    return data, report
