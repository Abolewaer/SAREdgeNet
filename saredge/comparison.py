import argparse
import csv
import json
import math
from pathlib import Path

from .evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['precision','recall','f1','lake_iou','background_iou','miou']
ALIASES = {'UNet':'U-Net','AttentionUNet':'Attention U-Net','DeepLabV3Plus':'DeepLabV3+','OSTU':'OTSU'}


def load(path,base):
    path = Path(path)
    path = path if path.is_absolute() else base/path
    return json.loads(path.read_text(encoding='utf-8'))


def collect(manifest,output):
    manifest = Path(manifest)
    config = json.loads(manifest.read_text(encoding='utf-8'))
    base = Path(config.get('base_dir',str(ROOT))).resolve()
    rows = []
    seen = set()
    for entry in config['methods']:
        name = ALIASES.get(entry['method'],entry['method'])
        group = entry.get('evaluation_id',config.get('evaluation_id'))
        if not group:
            raise ValueError('An explicit evaluation_id is required')
        if (name,group) in seen:
            raise ValueError('Duplicate method/evaluation group')
        seen.add((name,group))
        region = load(entry['region_metrics'],base)
        counts = [region['region'][key] for key in ['TP','FP','FN','TN']]
        computed = metrics(counts)
        for metric in METRICS:
            if not math.isclose(region['region'][metric],computed[metric],rel_tol=1e-7,abs_tol=1e-9):
                raise ValueError('Metrics disagree with TP/FP/FN/TN: '+name)
        row = {'method':name,'evaluation_id':group,**computed,
               'valid_pixels':sum(counts),'reference_positive_pixels':counts[0]+counts[2],
               'region_source':entry['region_metrics']}
        signatures = [case.get('evaluation_signature') for case in region.get('cases',[])]
        row['evaluation_signatures'] = sorted(signatures) if signatures and all(signatures) else None
        if entry.get('boundary_metrics'):
            boundary = load(entry['boundary_metrics'],base)
        else:
            boundary = region
        summary = boundary.get('boundary_mean',{})
        row.update(boundary_f1=summary.get('boundary_f1'),hausdorff_km=summary.get('hausdorff_km'),
                   boundary_cases=summary.get('cases'),hausdorff_valid_cases=summary.get('hausdorff_valid_cases'))
        boundary_signatures = [case.get('evaluation_signature') for case in boundary.get('cases',[])] if summary else []
        row['boundary_evaluation_signatures'] = sorted(boundary_signatures) if boundary_signatures and all(boundary_signatures) else None
        row['boundary_tolerance_pixels'] = summary.get('tolerance_pixels')
        if entry.get('seed_range'):
            seed = load(entry['seed_range'],base)
            row['seed_runs'] = seed['runs']
            for metric in METRICS:
                low,high = seed['metrics'][metric]['min'],seed['metrics'][metric]['max']
                if not 0 <= low <= high <= 1:
                    raise ValueError('Invalid seed range')
                row.update({f'{metric}_seed_min':low,f'{metric}_seed_max':high,
                            f'{metric}_lower_offset':low-row[metric],f'{metric}_upper_offset':high-row[metric]})
        if entry.get('efficiency'):
            efficiency = load(entry['efficiency'],base)
            measured_method = efficiency.get('method')
            if measured_method and ALIASES.get(measured_method,measured_method)!=name:
                raise ValueError('Efficiency report belongs to a different method: '+name)
            row.update(model_parameters_millions=efficiency.get('parameters_millions'),
                       runtime_midpoint_seconds=efficiency.get('midpoint_seconds'),
                       runtime_half_range_seconds=efficiency.get('half_range_seconds'),
                       gpu_peak_allocated_gib=efficiency.get('gpu_peak_allocated_gib'),
                       gpu_peak_reserved_gib=efficiency.get('gpu_peak_reserved_gib'),
                       timing_scope=efficiency.get('scope'),timing_device=efficiency.get('device'),
                       timing_repetitions=efficiency.get('repetitions'),efficiency_input_pixels=efficiency.get('input_pixels'),
                       efficiency_input_grid=efficiency.get('input_grid'))
            estimate = efficiency.get('training_time_estimate') or {}
            low,high = estimate.get('minimum_seconds'),estimate.get('maximum_seconds')
            if low is not None and high is not None:
                row['estimated_training_midpoint_minutes'] = (low+high)/120
                row['estimated_training_half_range_minutes'] = (high-low)/120
        rows.append(row)
    if not rows:
        raise ValueError('No methods')
    diagnostics = []
    for group in sorted({r['evaluation_id'] for r in rows}):
        selected = [r for r in rows if r['evaluation_id']==group]
        counts_match = len({(r['valid_pixels'],r['reference_positive_pixels']) for r in selected})==1
        signatures = [r['evaluation_signatures'] for r in selected]
        signatures_known = all(s is not None for s in signatures)
        signatures_match = signatures_known and all(s==signatures[0] for s in signatures)
        status = 'verified' if counts_match and signatures_match else ('unverified_legacy_reports' if counts_match and not signatures_known else 'mismatch')
        diagnostics.append({'evaluation_id':group,'status':status,'methods':[r['method'] for r in selected],
                            'valid_pixel_and_reference_counts_match':counts_match,'pixel_signatures_match':bool(signatures_match)})
        timed = [r for r in selected if r.get('runtime_midpoint_seconds') is not None]
        if timed:
            scopes = {r.get('timing_scope') for r in timed}
            devices = {r.get('timing_device') for r in timed}
            grids = {json.dumps(r.get('efficiency_input_grid'),sort_keys=True) for r in timed}
            diagnostics[-1]['efficiency_comparable'] = len(scopes)==1 and len(devices)==1 and len(grids)==1
            diagnostics[-1]['efficiency_scopes'] = sorted(str(value) for value in scopes)
        boundaries = [r for r in selected if r.get('boundary_f1') is not None]
        if boundaries:
            counts = {(r.get('boundary_cases'),r.get('boundary_tolerance_pixels')) for r in boundaries}
            signatures = [r['boundary_evaluation_signatures'] for r in boundaries]
            signatures_known = all(value is not None for value in signatures)
            diagnostics[-1]['boundary_comparable'] = len(counts)==1 and (not signatures_known or all(s==signatures[0] for s in signatures))
    report = {'rows':rows,'evaluation_checks':diagnostics,
              'metric_units':'region and boundary scores in [0,1]; HD km; time seconds; GPU memory GiB',
              'missing_values':'not measured or not applicable; never replaced with zero'}
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report


def export(source,output_dir,allow_incomparable=False):
    report = json.loads(Path(source).read_text(encoding='utf-8'))
    if any(check['status']=='mismatch' or check.get('efficiency_comparable') is False or check.get('boundary_comparable') is False for check in report['evaluation_checks']) and not allow_incomparable:
        raise ValueError('Evaluation pixels or references differ; use a shared evaluation mask before comparison')
    rows = report['rows']
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    excluded = ['evaluation_signatures','boundary_evaluation_signatures','efficiency_input_grid','region_source']
    columns = list(dict.fromkeys(key for row in rows for key in row if key not in excluded))
    with (output/'comparison_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (output/'comparison_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    base_columns = ['method','evaluation_id',*METRICS,'boundary_f1','hausdorff_km']
    with (output/'accuracy_table.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=base_columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    efficiency_columns = ['method','evaluation_id','model_parameters_millions','runtime_midpoint_seconds',
                          'runtime_half_range_seconds','gpu_peak_allocated_gib','gpu_peak_reserved_gib',
                          'estimated_training_midpoint_minutes','estimated_training_half_range_minutes',
                          'miou','timing_scope','timing_device','timing_repetitions']
    with (output/'efficiency_table.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=efficiency_columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main(stage):
    parser = argparse.ArgumentParser()
    if stage==1:
        parser.add_argument('--manifest',required=True)
        parser.add_argument('--output',required=True)
    else:
        parser.add_argument('--input',required=True)
        parser.add_argument('--output-dir',required=True)
        parser.add_argument('--allow-incomparable',action='store_true')
    args = parser.parse_args()
    if stage==1:
        collect(args.manifest,args.output)
    else:
        export(args.input,args.output_dir,args.allow_incomparable)
