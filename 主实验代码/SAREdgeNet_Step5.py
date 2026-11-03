from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saredge.models import SAREdgeNetContourClassifier
from saredge.efficiency import measure_epochs

class ContourDataset(Dataset):

    def __init__(self, frame, indices, image_dir, feature_columns, scaler, train):
        self.frame = frame.iloc[indices].reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.feature_columns = feature_columns
        self.scaler = scaler
        augmentation = [transforms.RandomHorizontalFlip(p=0.3), transforms.RandomRotation(10)] if train else []
        self.transform = transforms.Compose([transforms.Resize((224, 224)), *augmentation, transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])

    def __len__(self):
        return len(self.frame)

    @staticmethod
    def normalize_identifier(value):
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        contour_id = self.normalize_identifier(row.iloc[0])
        image = Image.open(self.image_dir / f'{contour_id}.png').convert('L')
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        features = row[self.feature_columns].to_numpy(dtype=np.float32)[None, :]
        features = self.scaler.transform(features)[0]
        label = int(row.iloc[-1])
        return (self.transform(image), torch.tensor(features), torch.tensor(label))

def read_table(path):
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    return pd.read_csv(path)

def evaluate(model, loader, criterion, device):
    model.eval()
    losses, labels, predictions = ([], [], [])
    with torch.no_grad():
        for images, features, target in loader:
            images, features, target = (images.to(device), features.to(device), target.to(device))
            logits = model(images, features)
            losses.append(criterion(logits, target).item())
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return {'loss': float(np.mean(losses)), 'accuracy': accuracy_score(labels, predictions), 'precision_weighted': precision_score(labels, predictions, average='weighted', zero_division=0), 'recall_weighted': recall_score(labels, predictions, average='weighted', zero_division=0), 'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=8e-06)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--spatial-protocol', default=str(Path(__file__).resolve().parents[1] / 'configs/SpatialProtocol.json'))
    parser.add_argument('--spatial-samples', required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_table(args.table)
    feature_columns = list(frame.columns[1:-1])
    if len(feature_columns) != 17:
        raise ValueError(f'Expected 17 numerical columns between ID and label; got {len(feature_columns)}')
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Features must be finite')
    if not frame.iloc[:, -1].isin([0, 1]).all():
        raise ValueError('Every contour needs a binary label')
    if frame.iloc[:, 0].duplicated().any():
        raise ValueError('Contour identifiers must be unique')
    labels = frame.iloc[:, -1].to_numpy(dtype=np.int64)
    if (np.bincount(labels, minlength=2) < 5).any():
        raise ValueError('Both classes need at least five examples')
    from saredge.spatial_protocol import contour_indices
    indices, spatial_rows, spatial_audit = contour_indices(frame, {'spatial_protocol_file': args.spatial_protocol,
        'spatial_samples_file': args.spatial_samples})
    train, val, test = (indices[k] for k in ['train', 'validation', 'test'])
    scaler = StandardScaler().fit(frame.iloc[train][feature_columns].to_numpy(dtype=np.float32))
    np.savez(output_dir / 'feature_scaler.npz', mean=scaler.mean_, scale=scaler.scale_, feature_columns=np.asarray(feature_columns))
    spatial_rows.to_csv(output_dir / 'split_indices.csv', index=False, encoding='utf-8-sig')
    (output_dir / 'spatial_split_audit.json').write_text(json.dumps(spatial_audit, ensure_ascii=False, indent=2), encoding='utf8')
    train_set = ContourDataset(frame, train, args.image_dir, feature_columns, scaler, train=True)
    val_set = ContourDataset(frame, val, args.image_dir, feature_columns, scaler, train=False)
    test_set = ContourDataset(frame, test, args.image_dir, feature_columns, scaler, train=False)
    loaders = {'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True), 'validation': DataLoader(val_set, batch_size=args.batch_size, shuffle=False), 'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False)}
    if len(loaders['train']) == 0:
        raise ValueError('No full training batch; reduce --batch-size')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAREdgeNetContourClassifier(pretrained=not args.no_pretrained).to(device)
    counts = np.bincount(labels[train], minlength=2)
    if (counts == 0).any():
        raise ValueError('Kekexili training set needs both classes')
    class_weights = len(train) / (2.0 * counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5, min_lr=1e-06)
    best_f1, patience = (-1.0, 0)
    checkpoint_path = output_dir / 'best_classifier.pt'
    history = []
    for epoch in measure_epochs(range(1,args.epochs+1),output_dir/'epoch_times.json'):
        model.train()
        train_losses = []
        for images, features, target in loaders['train']:
            images, features, target = (images.to(device), features.to(device), target.to(device))
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, features)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        validation = evaluate(model, loaders['validation'], criterion, device)
        scheduler.step(validation['f1_weighted'])
        history.append({'epoch': epoch, 'train_loss': float(np.mean(train_losses)), **validation, 'learning_rate': optimizer.param_groups[0]['lr']})
        if validation['f1_weighted'] > best_f1:
            best_f1 = validation['f1_weighted']
            patience = 0
            torch.save({'model_state': model.state_dict(), 'feature_columns': feature_columns, 'class_weights': class_weights.tolist(), 'epoch': epoch, 'validation_f1': best_f1}, checkpoint_path)
        else:
            patience += 1
        if patience >= 15 or optimizer.param_groups[0]['lr'] < 1e-06:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    test_metrics = evaluate(model, loaders['test'], criterion, device) if len(test) else None
    with open(output_dir / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'evaluation_unit':'external_contour_classification','test_size':len(test),
                   'status':'evaluated' if len(test) else 'external_contour_test_not_supplied',
                   'metrics':test_metrics}, f, indent=2)
    print(json.dumps({'best_epoch': checkpoint['epoch'], 'best_validation_f1': checkpoint['validation_f1'], 'test': test_metrics}, indent=2))
if __name__ == '__main__':
    main()
