
from pathlib import Path
import json
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'saredge/spatial_protocol.py').is_file())
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from protocol import digest, dump


def held_out_scenes(cfg):
    root = Path(cfg['output_dir'])
    split = root / 'shared_split.csv'; audit = json.loads((root / 'data_audit.json').read_text(encoding='utf8'))
    if digest(split) != audit['split_sha256'] or digest(root / 'shared_manifest.csv') != audit['manifest_sha256']:
        raise ValueError('Shared data artifacts changed')
    data = pd.read_csv(root / 'shared_manifest.csv', dtype=str)
    column = cfg['pixel_interface']['scene_column']
    if audit['protocol']['split_mode'] != 'kekexili_external':
        raise ValueError('Expected Kekexili 80/20 development split and external-only testing')
    known = set()
    if column in data:
        if data[column].isna().any():
            raise ValueError('Real scene IDs must be complete when provided')
        known = set(data.loc[data.subset == 'test', column])
    if cfg.get('external_scene_ids_file'):
        explicit = pd.read_csv(cfg['external_scene_ids_file'], dtype=str)
        if 'scene_id' not in explicit or explicit.scene_id.isna().any() or explicit.scene_id.duplicated().any():
            raise ValueError('external_scene_ids_file requires unique nonempty real scene_id values')
        external_scenes = set(explicit.scene_id)
        if not known.issubset(external_scenes):
            raise ValueError('External scene inventory omits labelled external test scenes')
    else:
        external_scenes = known
    if not external_scenes:
        raise ValueError('Supply external_scene_ids_file for full pixel tests when no labelled external contour inventory exists; do not fabricate contour labels')
    from saredge.spatial_protocol import load_protocol
    current = load_protocol(cfg.get('spatial_protocol_file'))
    if current['signature'] != audit['protocol']['spatial_audit']['protocol_signature']:
        raise ValueError('Spatial protocol changed after training')
    return external_scenes, audit


def export_requests(cfg, variant_id, seed, scenes_file):

    scenes, audit = held_out_scenes(cfg)
    path = Path(scenes_file).resolve(); table = pd.read_csv(path, dtype=str)
    required = ['scene_id', 'sar', 'dem', 'reference', 'valid_mask', 'candidate_table', 'candidate_pixel_mapping']
    if set(required) - set(table.columns):
        raise ValueError(f'Required scene-input columns: {required}')
    if table[required].isna().any().any() or table.scene_id.duplicated().any() or set(table.scene_id) != scenes:
        raise ValueError('Supply every held-out scene exactly once, without train/validation scenes or blank inputs')
    for column in required[1:]:
        table[column] = table[column].map(lambda v: str((path.parent / v).resolve()))
        for value in table[column]:
            if not Path(value).is_file():
                raise FileNotFoundError(value)
    run = Path(cfg['output_dir']) / variant_id / f'seed_{seed}'
    record = json.loads((run / 'run_record.json').read_text(encoding='utf8'))
    if record['status'] != 'completed' or record['split_sha256'] != audit['split_sha256']:
        raise ValueError('Completed classifier run with the same split is required')
    if digest(run / 'best.pt') != record['checkpoint_sha256']:
        raise ValueError('Checkpoint hash changed')
    if cfg['pixel_interface'].get('frozen_pipeline_configuration') is None:
        raise ValueError('Set pixel_interface.frozen_pipeline_configuration to the actual shared candidate-generation and postprocessing parameter object; do not guess it')
    requests = dict(variant_id=variant_id, seed=seed, checkpoint=str(run / 'best.pt'),
        checkpoint_sha256=record['checkpoint_sha256'], split_sha256=audit['split_sha256'],
        classifier_definition=record['variant'], scaler=str(run / 'scaler_train_only.npz') if record['variant']['columns'] else None,
        scene_inputs=table.to_dict('records'), frozen_pipeline_configuration=cfg['pixel_interface']['frozen_pipeline_configuration'],
        required_adapter='Use this folder networks.Classifier and exact selected/scaled columns. Reconstruct scene masks using real candidate_pixel_mapping. Existing legacy Step6 cannot directly load this new checkpoint format.',
        required_output='One binary prediction TIFF per scene on reference grid; output manifest needs scene_id,prediction,reference,valid_mask,checkpoint_sha256,split_sha256',
        executed=False)
    dump(run / 'external_pipeline_requests.json', requests)
    return requests


def evaluate_pixels(cfg, variant_id, seed, manifest):
    import rasterio
    from rasterio.windows import Window
    from saredge.spatial_protocol import Partition
    scenes, audit = held_out_scenes(cfg)
    run = Path(cfg['output_dir']) / variant_id / f'seed_{seed}'
    record = json.loads((run / 'run_record.json').read_text(encoding='utf8'))
    checkpoint_hash = digest(run / 'best.pt')
    if record['status'] != 'completed' or record['checkpoint_sha256'] != checkpoint_hash or record['split_sha256'] != audit['split_sha256']:
        raise ValueError('Model/split provenance mismatch')
    path = Path(manifest).resolve(); table = pd.read_csv(path, dtype=str)
    required = ['scene_id', 'prediction', 'reference', 'valid_mask', 'checkpoint_sha256', 'split_sha256']
    if set(required) - set(table.columns) or table[required].isna().any().any():
        raise ValueError(f'Complete external manifest required: {required}')
    if table.scene_id.duplicated().any() or set(table.scene_id) != scenes:
        raise ValueError('Pixel manifest must contain all and only held-out scenes, exactly once')
    if not table.checkpoint_sha256.eq(checkpoint_hash).all() or not table.split_sha256.eq(audit['split_sha256']).all():
        raise ValueError('External outputs must declare the exact checkpoint and shared split SHA256')
    for column in ['prediction', 'reference', 'valid_mask']:
        table[column] = table[column].map(lambda v: str((path.parent / v).resolve()))
        for value in table[column]:
            if not Path(value).is_file():
                raise FileNotFoundError(value)
    results = []; total = dict(TN=0, FP=0, FN=0, TP=0)
    foreground = int(cfg['pixel_interface']['foreground_value'])
    if foreground not in [1, 255]:
        raise ValueError('Binary foreground_value must be 1 or 255')
    for row in table.to_dict('records'):
        counts = dict(TN=0, FP=0, FN=0, TP=0)
        with rasterio.open(row['prediction']) as pred, rasterio.open(row['reference']) as ref, rasterio.open(row['valid_mask']) as valid, Partition(row['valid_mask'], cfg.get('spatial_protocol_file')) as partition:
            if any(d.count != 1 or d.crs is None or d.crs != ref.crs or d.transform != ref.transform or d.shape != ref.shape for d in [pred, ref, valid]):
                raise ValueError('Binary prediction, reference and independent valid_mask must have identical CRS/grid/size; no implicit resizing or warping')
            for y in range(0, ref.height, 1024):
                for x in range(0, ref.width, 1024):
                    window = Window(x, y, min(1024, ref.width-x), min(1024, ref.height-y))
                    p = pred.read(1, window=window); r = ref.read(1, window=window); v = valid.read(1, window=window)
                    if not np.isin(v, [0, 1, 255]).all():
                        raise ValueError('valid_mask must use 0 outside and 1 or 255 inside actual evaluation coverage')
                    m = (v > 0) & partition.pixel_mask(window, 'test')
                    if not np.isin(p[m], [0, foreground]).all() or not np.isin(r[m], [0, foreground]).all():
                        raise ValueError('Prediction/reference values disagree with configured binary encoding')
                    p = p[m] == foreground; r = r[m] == foreground
                    counts['TP'] += int((p & r).sum()); counts['FP'] += int((p & ~r).sum())
                    counts['FN'] += int((~p & r).sum()); counts['TN'] += int((~p & ~r).sum())
        if sum(counts.values()) == 0:
            raise ValueError(f'Empty independent evaluation mask for {row["scene_id"]}')
        for key in total:
            total[key] += counts[key]
        results.append(dict(scene_id=row['scene_id'], counts=counts, metrics=pixel_scores(counts)))
    report = dict(evaluation_unit='pixel_segmentation', variant_id=variant_id, seed=seed,
                  split_sha256=audit['split_sha256'], checkpoint_sha256=checkpoint_hash,
                  external_manifest_sha256=digest(path), scenes=results, pooled_counts=total, metrics=pixel_scores(total),
                  note='External full-pipeline masks; exact pixel overlap. No boundary tolerance. Validity must come from imagery/reference coverage, never prediction foreground/alpha.')
    destination = run / 'pixel_metrics.json'
    if destination.exists():
        raise FileExistsError('Preserve existing pixel result; use a new run/output directory')
    dump(destination, report)
    return report


def pixel_scores(c):
    tp, fp, fn, tn = (c[k] for k in ['TP', 'FP', 'FN', 'TN'])
    def div(a, b): return a / b if b else None
    return dict(precision=div(tp, tp+fp), recall=div(tp, tp+fn), f1=div(2*tp, 2*tp+fp+fn),
                iou_lake=div(tp, tp+fp+fn), iou_background=div(tn, tn+fp+fn), accuracy=div(tp+tn, tp+tn+fp+fn))
