import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ['SAREdgeNet', 'UNet', 'AttentionUNet', 'DeepLabV3Plus', 'FPN', 'PSPNet']
SEEDS = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
METRICS = ['precision', 'recall', 'f1', 'lake_iou', 'background_iou', 'miou']


def path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def read(filename):
    return json.loads(Path(filename).read_text(encoding='utf-8-sig'))


def write(filename, value):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def command(script, *arguments):
    return [sys.executable, str(ROOT / script), *map(str, arguments)]


def normalized(config):
    result = dict(config)
    for key in ['tif_path', 'shp_path', 'dem_path', 'dem_preprocessed', 'pretrain_path',
                'train_table', 'train_image_dir', 'spatial_samples_file', 'spatial_protocol_file']:
        if result.get(key):
            result[key] = str(path(result[key]))
    return result


def build(config):
    if config['seeds'] != SEEDS or set(config['methods']) != set(METHODS):
        raise ValueError('All six comparison methods and the fixed ten seeds are required')
    protocol_path = path(config['spatial_protocol_file'])
    protocol = read(protocol_path)
    if (protocol['spatial_seed'], protocol['development_train_fraction'],
            protocol['development_validation_fraction']) != (42, 0.8, 0.2):
        raise ValueError('Keep classwise 80:20 split with spatial_seed=42')
    output = path(config['output_dir'])
    cases = config['evaluation_cases']
    names = [c['name'] for c in cases]
    if not cases or len(set(names)) != len(names) or any(not n or Path(n).name != n or n in ['.', '..'] for n in names):
        raise ValueError('Evaluation case names must be unique safe folder names')
    required = {str(protocol_path), str(path(protocol['train_roi']))}
    preparation = []
    for case in cases:
        required.update(str(path(case[k])) for k in ['sar', 'reference'])
        if config.get('prepare_candidates', False):
            required.add(str(path(case['dem'])))
            preparation.append(command('主实验代码/SAREdgeNet_Step4.py', '--sar', path(case['sar']),
                                       '--dem', path(case['dem']), '--output', path(case['contour_dir'])))
        else:
            required.update(str(path(case['contour_dir']) / f) for f in
                            ['contour_statistics_enhanced_with_dem.csv', 'candidate_ids.tif', 'step7_filtered_mask3.tif'])
    jobs = []
    for method in METHODS:
        base = normalized(read(path(config['methods'][method])))
        base.update(spatial_protocol_file=str(protocol_path), evaluation_subset='test')
        keys = ['train_table', 'train_image_dir', 'spatial_samples_file'] if method == 'SAREdgeNet' else ['tif_path', 'shp_path']
        required.update(base[k] for k in keys)
        required.update(base[k] for k in ['dem_path', 'dem_preprocessed', 'pretrain_path'] if base.get(k))
        for seed in SEEDS:
            folder = output / method / f'seed_{seed}'
            train = dict(base, training_seed=seed, output_path=str(folder / 'training_prediction.tif'),
                         plot_save_path=str(folder / 'training.png'))
            configs = {}
            commands = []
            if method == 'SAREdgeNet':
                checkpoint = folder / 'classifier/best_classifier.pt'
                commands.append(command('主实验代码/SAREdgeNet_Step5.py', '--table', train['train_table'],
                    '--image-dir', train['train_image_dir'], '--output-dir', checkpoint.parent,
                    '--seed', seed, '--epochs', train.get('epochs', 10000), '--batch-size', train.get('batch_size', 32),
                    '--learning-rate', train.get('learning_rate', 8e-6), '--weight-decay', train.get('weight_decay', 1e-4),
                    '--spatial-protocol', protocol_path, '--spatial-samples', train['spatial_samples_file']))
            else:
                key = 'model_path' if method == 'DeepLabV3Plus' else 'model_save_path'
                train[key] = str(folder / 'model.pth')
                checkpoint = folder / ('model_best.pth' if method == 'DeepLabV3Plus' else 'model.pth')
                commands.append(command(f'对比实验代码/{method}_Step1.py', '--config', folder / 'train_config.json'))
            configs[str(folder / 'train_config.json')] = train
            evaluation_cases = []
            for case in cases:
                target = folder / 'evaluation' / case['name']
                ev = dict(train)
                if method == 'SAREdgeNet':
                    ev.update(checkpoint=str(checkpoint), output_dir=str(target),
                              contour_dir=str(path(case['contour_dir'])), processed_sar=str(path(case['sar'])),
                              support_mask=str(path(case['contour_dir']) / 'step7_filtered_mask3.tif'),
                              reference=str(path(case['reference'])))
                    commands.append(command('Run_Step1.py', '--config', target / 'config.json', '--steps', 6, 7, 8, 9))
                    prediction = target / 'step9/Polish.tif'
                else:
                    ev.update(tif_path=str(path(case['sar'])), shp_path=str(path(case['reference'])),
                              output_path=str(target / 'prediction.tif'))
                    ev[key] = str(checkpoint)
                    for dem_key in ['dem_path', 'dem_preprocessed']:
                        if train.get(dem_key):
                            ev[dem_key] = str(path(case['dem']))
                            required.add(ev[dem_key])
                    commands.append(command(f'对比实验代码/{method}_Step2.py', '--config', target / 'config.json'))
                    prediction = target / 'prediction.tif'
                configs[str(target / 'config.json')] = ev
                evaluation_cases.append(dict(prediction=str(prediction), reference=str(path(case['reference'])),
                    validity_source=str(path(case['sar'])), rgb_red=True, positive_value=1,
                    spatial_subset='test', spatial_protocol_file=str(protocol_path)))
            manifest = folder / 'evaluation_cases.json'
            configs[str(manifest)] = evaluation_cases
            metrics_file = folder / 'pixel_metrics.json'
            commands.append(command('对比实验代码/UNet_Step3.py', '--manifest', manifest,
                                    '--spatial-protocol', protocol_path, '--spatial-subset', 'test', '--output', metrics_file))
            jobs.append(dict(method=method, seed=seed, folder=str(folder), configs=configs, commands=commands,
                             metrics=str(metrics_file), status='planned'))
    for method, entry in config.get('deterministic', {}).items():
        if method not in ['NDVI', 'OSTU']:
            raise ValueError('Only NDVI and OSTU are deterministic entries')
        eval_cases = []
        for case in cases:
            prediction = str(path(entry['predictions'][case['name']]))
            required.add(prediction)
            eval_cases.append(dict(prediction=prediction, reference=str(path(case['reference'])),
                validity_source=str(path(case['sar'])), rgb_red=entry.get('rgb_red', True),
                positive_value=entry.get('positive_value', 1), spatial_subset='test', spatial_protocol_file=str(protocol_path)))
        folder = output / method
        manifest = folder / 'evaluation_cases.json'
        metrics_file = folder / 'pixel_metrics.json'
        jobs.append(dict(method=method, seed=None, folder=str(folder), configs={str(manifest): eval_cases},
            commands=[command('对比实验代码/UNet_Step3.py', '--manifest', manifest, '--output', metrics_file,
                              '--spatial-protocol', protocol_path, '--spatial-subset', 'test')],
            metrics=str(metrics_file), status='planned'))
    source_files = set(ROOT.glob('saredge/*.py'))
    source_files.add(ROOT / 'Run_Step1.py')
    source_files.add(Path(__file__).resolve())
    source_files.update(Path(c[1]) for j in jobs for c in j['commands'])
    source_files.update(ROOT / f'对比实验代码/{m}_Step1.py' for m in METHODS if m != 'SAREdgeNet')
    source_files.update(ROOT / f'主实验代码/SAREdgeNet_Step{i}.py' for i in [4, 5, 6, 7, 8, 9])
    code_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)}
    input_metadata = {}
    for value in sorted(required):
        p = Path(value)
        if p.exists():
            stat = p.stat()
            input_metadata[value] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, directory=p.is_dir())
    return dict(seeds=SEEDS, training_runs=60, spatial_protocol=protocol, code_hashes=code_hashes,
                input_metadata=input_metadata, preparation=preparation, required_inputs=sorted(required), jobs=jobs)


def signatures(report):
    values = [(c['evaluation_signature'], c['spatial_protocol_signature'], c['spatial_subset']) for c in report['cases']]
    if not values or any(not a or not b or s != 'test' for a, b, s in values):
        raise ValueError('Missing evaluation signature or wrong spatial subset')
    return sorted(values)


def collect(config, plan, output):
    rows, all_values, common = [], {}, None
    for job in plan['jobs']:
        record = read(Path(job['folder']) / 'run_record.json')
        if record.get('status') != 'complete' or record.get('job_signature') != digest(job):
            raise ValueError(f'Incomplete or changed run: {job["method"]}/{job["seed"]}')
        report = read(job['metrics'])
        sig = signatures(report)
        if common is None:
            common = sig
        if sig != common:
            raise ValueError('Methods/seeds must use identical reference pixels and coverage')
        row = dict(method=job['method'], seed=job['seed'], **{k: report['region'][k] for k in METRICS})
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in METRICS):
            raise ValueError('Pixel metrics must be finite fractions between zero and one')
        rows.append(row)
        all_values.setdefault(job['method'], []).append(row)
    summary = {}
    for method, values in all_values.items():
        original = None
        if config.get('original_reports', {}).get(method):
            original_report = read(path(config['original_reports'][method]))
            if signatures(original_report) != common:
                raise ValueError('Original point and new runs use different evaluation scopes')
            original = original_report['region']
        summary[method] = dict(n=len(values), metrics={})
        for metric in METRICS:
            series = [v[metric] for v in values]
            low, high = min(series), max(series)
            item = dict(mean=statistics.mean(series), sample_sd=statistics.stdev(series) if len(series) > 1 else None,
                        minimum=low, maximum=high)
            if original is not None:
                item.update(original=original[metric], lower_offset_x1e3=(low-original[metric])*1000,
                            upper_offset_x1e3=(high-original[metric])*1000)
            summary[method]['metrics'][metric] = item
    write(output / 'comparison_summary.json', dict(evaluation_unit='pixel_segmentation', seeds=SEEDS,
                                                   fluctuation='observed_minimum_maximum', methods=summary))
    with (output / 'per_seed_pixel_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'seed', *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def main(default_mode='plan'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(HERE.parent / 'ten_seed_comparison.json'))
    parser.add_argument('--mode', choices=['plan', 'run', 'collect'], default=default_mode)
    args = parser.parse_args()
    config = read(path(args.config))
    plan = build(config)
    output = path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    missing = [p for p in plan['required_inputs'] if not Path(p).exists()]
    if args.mode == 'plan':
        write(output / 'comparison_plan.json', dict(**plan, missing_inputs=missing))
        print(json.dumps(dict(training_runs=60, deterministic_methods=list(config.get('deterministic', {})),
                              missing_inputs=missing, plan=str(output / 'comparison_plan.json')), ensure_ascii=False, indent=2))
        return
    if args.mode == 'collect':
        collect(config, plan, output)
        return
    if missing:
        raise FileNotFoundError('Configure real inputs before training:\n' + '\n'.join(missing))
    lock = output / 'experiment_protocol.json'
    identity = digest(plan)
    if lock.exists() and read(lock)['signature'] != identity:
        raise ValueError('Configuration changed: choose a new output directory')
    if not lock.exists():
        if any(Path(j['folder']).exists() for j in plan['jobs']):
            raise FileExistsError('Existing run folders have no matching experiment protocol')
        write(lock, dict(signature=identity, plan=plan))
    for i, cmd in enumerate(plan['preparation']):
        destination = Path(cmd[-1])
        expected = ['contour_statistics_enhanced_with_dem.csv', 'candidate_ids.tif', 'step7_filtered_mask3.tif']
        preparation_record = output / f'preparation_{i}.json'
        if preparation_record.exists() and read(preparation_record).get('status') == 'complete':
            if not all((destination / f).exists() for f in expected):
                raise FileNotFoundError('Previously prepared candidates are missing')
            continue
        if destination.exists():
            raise FileExistsError(f'Candidate destination already exists without a completed preparation record: {destination}')
        with (output / f'preparation_{i}.log').open('w', encoding='utf-8') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(preparation_record, dict(status='complete', command=cmd))
    for job in plan['jobs']:
        folder = Path(job['folder'])
        record_path = folder / 'run_record.json'
        signature = digest(job)
        if record_path.exists():
            record = read(record_path)
            if record.get('job_signature') != signature:
                raise ValueError(f'Changed job: {folder}')
            if record.get('status') == 'complete' and Path(job['metrics']).exists():
                continue
            raise RuntimeError(f'Incomplete run retained for inspection; use a fresh output directory: {folder}')
        if folder.exists():
            raise FileExistsError(f'Run folder already exists: {folder}')
        for filename, value in job['configs'].items():
            write(filename, value)
        write(record_path, dict(status='running', job_signature=signature))
        try:
            for index, cmd in enumerate(job['commands']):
                environment = dict(os.environ)
                if job['seed'] is not None:
                    environment['PYTHONHASHSEED'] = str(job['seed'])
                with (folder / f'command_{index}.log').open('w', encoding='utf-8') as log:
                    subprocess.run(cmd, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            signatures(read(job['metrics']))
            write(record_path, dict(status='complete', job_signature=signature))
        except Exception as exc:
            write(record_path, dict(status='failed', job_signature=signature, error=str(exc)))
            raise
    collect(config, plan, output)


if __name__ == '__main__':
    main()
