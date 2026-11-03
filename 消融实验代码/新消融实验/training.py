
from pathlib import Path
import hashlib
import json
import platform
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from networks import Classifier
from protocol import VARIANTS, digest, dump, prepare


def seed_everything(seed, deterministic):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


class Samples(Dataset):
    def __init__(self, data, columns, scaler, train, seed):
        self.data = data.reset_index(drop=True)
        self.columns = columns
        self.scaler = scaler
        self.seed = seed
        self.epoch = 0
        self.train = train
        augment = [transforms.RandomHorizontalFlip(.3), transforms.RandomRotation(10)] if train else []
        self.transform = transforms.Compose([transforms.Resize((224, 224)), *augment, transforms.ToTensor(),
                                              transforms.Normalize(mean=[.485], std=[.229])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with Image.open(row.image_path) as original:
            image = original.convert('L').resize((224, 224), Image.Resampling.LANCZOS)
                                                                            
        if self.train:
            key = f'{self.seed}:{self.epoch}:{row.sample_id}'.encode('utf8')
            local_seed = int(hashlib.sha256(key).hexdigest()[:15], 16)
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(local_seed)
                image = self.transform(image)
        else:
            image = self.transform(image)
        features = self.scaler.transform(row[self.columns].to_numpy(dtype=np.float32)[None])[0] if self.columns else np.empty(0, np.float32)
        return image, torch.tensor(features, dtype=torch.float32), int(row.label), str(row.sample_id)


def scores(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist()
    return dict(accuracy=float(accuracy_score(labels, predictions)),
                precision=float(precision_score(labels, predictions, zero_division=0)),
                recall=float(recall_score(labels, predictions, zero_division=0)),
                f1=float(f1_score(labels, predictions, zero_division=0)),
                weighted_f1=float(f1_score(labels, predictions, average='weighted', zero_division=0)),
                TN=tn, FP=fp, FN=fn, TP=tp)


def evaluate(model, loader, weights, device):
    model.eval(); labels = []; predictions = []; records = []; numerator = denominator = 0.
    with torch.no_grad():
        for images, numeric, target, ids in loader:
            target = target.to(device)
            logits = model(images.to(device), numeric.to(device))
            loss = nn.functional.cross_entropy(logits, target, weight=weights, reduction='sum')
            numerator += loss.item(); denominator += weights[target].sum().item()
            pred = logits.argmax(1).cpu().tolist(); true = target.cpu().tolist()
            probabilities = logits.softmax(1)[:, 1].cpu().tolist()
            labels.extend(true); predictions.extend(pred)
            records.extend(dict(sample_id=i, label=y, prediction=p, lake_probability=s) for i, y, p, s in zip(ids, true, pred, probabilities))
    return dict(loss=numerator / denominator, **scores(labels, predictions)), records


def train(cfg, variant_id, seed):
    if variant_id not in VARIANTS:
        raise ValueError('Unknown unique variant ID')
    data, audit = prepare(cfg)
    tr = cfg['training']; variant = VARIANTS[variant_id]; columns = variant['columns']
    if tr['optimizer'] not in ['Adam', 'AdamW'] or tr['selection_metric'] not in ['accuracy', 'f1', 'weighted_f1']:
        raise ValueError('Unsupported optimizer or validation selection metric')
    if tr['scheduler_monitor'] not in ['loss', 'accuracy', 'f1', 'weighted_f1']:
        raise ValueError('Unsupported scheduler monitor')
    batch = int(tr['batch_size']); epochs = int(tr['max_epochs'])
    early_stop = tr['early_stopping_patience']
    if batch < 2 or epochs < 1 or (early_stop is not None and int(early_stop) < 1):
        raise ValueError('batch_size>=2, positive epochs and null or positive early-stopping patience required')
    if not cfg.get('pretrained_weights') or not Path(cfg['pretrained_weights']).is_file():
        raise FileNotFoundError('Provide a local torchvision ResNet-34 ImageNet weight file; no automatic download')
    device_name = tr['device']
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly select cpu to change the experiment device')
    seed_everything(seed, bool(tr['deterministic']))
    out = Path(cfg['output_dir']) / variant_id / f'seed_{seed}'
    if out.exists():
        raise FileExistsError(f'Preserve existing run: {out}. Use a fresh output_dir instead of overwriting.')
    out.mkdir(parents=True)
    run = dict(variant_id=variant_id, seed=seed, variant=variant, configuration=cfg,
               evaluation_unit='contour_classification', image_content=cfg['image_content'],
               split_sha256=audit['split_sha256'], manifest_sha256=audit['manifest_sha256'],
               weights_sha256=digest(cfg['pretrained_weights']), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               python=platform.python_version(), torch=torch.__version__, device=device_name, status='started')
    dump(out / 'run_record.json', run)
    try:
        subsets = {s: data[data.subset == s].copy() for s in ['train', 'validation', 'test']}
        scaler = StandardScaler().fit(subsets['train'][columns].to_numpy(np.float32)) if columns else None
        if scaler is not None:
            np.savez(out / 'scaler_train_only.npz', mean=scaler.mean_, scale=scaler.scale_, columns=np.array(columns))
        datasets = {s: Samples(v, columns, scaler, s == 'train', seed) for s, v in subsets.items()}
        generator = torch.Generator().manual_seed(seed)
        loaders = {s: DataLoader(v, batch_size=batch, shuffle=s == 'train', drop_last=s == 'train',
                                 num_workers=0, generator=generator if s == 'train' else None) for s, v in datasets.items()}
        if not len(loaders['train']):
            raise ValueError('No complete training batch; reduce the shared batch size')
        device = torch.device(device_name)
        model = Classifier(variant, cfg['pretrained_weights']).to(device)
        counts = np.bincount(subsets['train'].label.to_numpy(int), minlength=2)
        class_weights = len(subsets['train']) / (2 * counts)
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        optimizer = getattr(torch.optim, tr['optimizer'])(model.parameters(), lr=float(tr['learning_rate']), weight_decay=float(tr['weight_decay']))
        monitor = tr['scheduler_monitor']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min' if monitor == 'loss' else 'max',
            patience=int(tr['scheduler_patience']), factor=float(tr['scheduler_factor']), min_lr=float(tr['min_lr']))
        best = -float('inf'); stale = 0; history = []; start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            datasets['train'].epoch = epoch
            model.train(); loss_sum = 0.; samples_seen = 0
            for images, numeric, target, _ in loaders['train']:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.to(device), numeric.to(device))
                loss = nn.functional.cross_entropy(logits, target.to(device), weight=weights)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                if tr['gradient_clip_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr['gradient_clip_norm']))
                optimizer.step(); loss_sum += loss.item() * len(target); samples_seen += len(target)
            val, _ = evaluate(model, loaders['validation'], weights, device)
            scheduler.step(val[monitor])
            history.append(dict(epoch=epoch, train_loss=loss_sum / samples_seen, validation=val, learning_rate=optimizer.param_groups[0]['lr']))
            selection = val[tr['selection_metric']]
            if selection > best:
                best = selection; stale = 0
                torch.save(dict(model_state=model.state_dict(), variant_id=variant_id, variant=variant, seed=seed,
                    selected_epoch=epoch, validation=val, class_weights=class_weights.tolist(),
                    split_sha256=audit['split_sha256'], manifest_sha256=audit['manifest_sha256']), out / 'best.pt')
            else:
                stale += 1
            dump(out / 'history.json', history)
            if early_stop is not None and stale >= int(early_stop):
                break
        checkpoint = torch.load(out / 'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'], strict=True)
                                                                                 
        if len(subsets['test']):
            result, predictions = evaluate(model, loaders['test'], weights, device)
        else:
            result, predictions = None, []
        pd.DataFrame(predictions, columns=['sample_id','label','prediction','lake_probability']).to_csv(out / 'test_predictions.csv', index=False, encoding='utf-8-sig')
        report = dict(evaluation_unit='contour_classification', variant_id=variant_id, seed=seed,
                      test_size=len(subsets['test']), status='evaluated' if len(subsets['test']) else 'external_contour_test_not_supplied', selected_epoch=checkpoint['selected_epoch'],
                      split_sha256=audit['split_sha256'], manifest_sha256=audit['manifest_sha256'],
                      image_content=cfg['image_content'], metrics=result)
        dump(out / 'test_metrics.json', report)
        run.update(status='completed', elapsed_seconds=time.perf_counter() - start, selected_epoch=checkpoint['selected_epoch'],
                   checkpoint_sha256=digest(out / 'best.pt'), class_counts_train=counts.tolist(), class_weights_train=class_weights.tolist())
        dump(out / 'run_record.json', run)
        return report
    except Exception as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        dump(out / 'run_record.json', run)
        raise


def aggregate(cfg, allow_partial=False):
    rows = []; unavailable = 0; expected = {(v, s) for v in VARIANTS for s in cfg['seeds']}
    for variant_id, seed in sorted(expected):
        path = Path(cfg['output_dir']) / variant_id / f'seed_{seed}' / 'test_metrics.json'
        if path.is_file():
            r = json.loads(path.read_text(encoding='utf8'))
            if r['evaluation_unit'] != 'contour_classification' or r['variant_id'] != variant_id or r['seed'] != seed:
                raise ValueError(f'Result identity mismatch: {path}')
            if r['metrics'] is None:
                unavailable += 1
                continue
            rows.append(dict(variant_id=variant_id, seed=seed, **{k: r[k] for k in ['test_size', 'selected_epoch', 'split_sha256', 'manifest_sha256', 'image_content']}, **r['metrics']))
    if unavailable == len(expected):
        dump(Path(cfg['output_dir']) / 'contour_mean_std.json', dict(evaluation_unit='contour_classification', status='external_contour_test_not_supplied', summary=None))
        aggregate_pixels(cfg, allow_partial)
        return dict(contour_summary=None, status='external_contour_test_not_supplied')
    if not rows or (len(rows) != len(expected) and not allow_partial):
        raise ValueError(f'Found {len(rows)}/{len(expected)} runs; --allow-partial is required for an incomplete summary')
    table = pd.DataFrame(rows)
    for column in ['split_sha256', 'manifest_sha256', 'image_content', 'test_size']:
        if table[column].nunique() != 1:
            raise ValueError(f'Cannot combine runs with differing {column}')
    table.to_csv(Path(cfg['output_dir']) / 'contour_metrics_by_seed.csv', index=False, encoding='utf-8-sig')
    summary = []
    for variant, group in table.groupby('variant_id'):
        row = dict(variant_id=variant, n_seeds=len(group), complete=len(group) == len(cfg['seeds']))
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'weighted_f1']:
            row[metric + '_mean'] = float(group[metric].mean())
            row[metric + '_std_sample'] = float(group[metric].std(ddof=1)) if len(group) > 1 else None
        summary.append(row)
    dump(Path(cfg['output_dir']) / 'contour_mean_std.json', {'evaluation_unit': 'contour_classification', 'complete_matrix': len(rows) == len(expected), 'summary': summary})
    aggregate_pixels(cfg, allow_partial)
    return summary


def aggregate_pixels(cfg, allow_partial):

    rows = []
    expected = {(v, s) for v in VARIANTS for s in cfg['seeds']}
    for variant, seed in sorted(expected):
        path = Path(cfg['output_dir']) / variant / f'seed_{seed}' / 'pixel_metrics.json'
        if path.is_file():
            result = json.loads(path.read_text(encoding='utf8'))
            if result['evaluation_unit'] != 'pixel_segmentation' or result['variant_id'] != variant or result['seed'] != seed:
                raise ValueError('Pixel result identity mismatch')
            rows.append(dict(variant_id=variant, seed=seed, split_sha256=result['split_sha256'],
                             scene_ids=';'.join(sorted(r['scene_id'] for r in result['scenes'])), **result['metrics']))
    if not rows:
        return
    if len(rows) != len(expected) and not allow_partial:
        raise ValueError('Partial pixel matrix found; explicitly use --allow-partial to summarize it')
    table = pd.DataFrame(rows)
    if table.split_sha256.nunique() != 1 or table.scene_ids.nunique() != 1:
        raise ValueError('Cannot combine pixel outputs from different splits or scene lists')
    table.to_csv(Path(cfg['output_dir']) / 'pixel_metrics_by_seed.csv', index=False, encoding='utf-8-sig')
    summary = []
    for variant, group in table.groupby('variant_id'):
        row = dict(variant_id=variant, n_seeds=len(group), complete=len(group) == len(cfg['seeds']))
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'iou_lake', 'iou_background']:
            values = pd.to_numeric(group[metric], errors='raise').dropna()
            row[metric + '_n_defined'] = len(values)
            row[metric + '_mean'] = float(values.mean()) if len(values) else None
            row[metric + '_std_sample'] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(row)
    dump(Path(cfg['output_dir']) / 'pixel_mean_std.json', dict(evaluation_unit='pixel_segmentation', complete_matrix=len(rows) == len(expected), summary=summary))
