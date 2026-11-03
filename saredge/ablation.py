import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import ImageBranch,NumericalBranch
from .training import seed_training

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'主实验代码'))
from SAREdgeNet_Step5 import ContourDataset,read_table

VARIANTS = {
    'Baseline':('resnet34',True,'both'),
    'ResNet18':('resnet18',True,'both'),
    'NoPretrain':('resnet34',False,'both'),
    'ContourOnly':('resnet34',True,'image'),
    'NumericOnly':('resnet34',False,'numeric'),
    'ResNet18NoPretrain':('resnet18',False,'both')}


class AblationClassifier(nn.Module):
    def __init__(self,variant,initialize=True):
        super().__init__()
        backbone,pretrained,mode = VARIANTS[variant]
        self.mode = mode
        if mode!='numeric':
            self.image_branch = ImageBranch(pretrained=pretrained and initialize,backbone_name=backbone)
        if mode!='image':
            self.numerical_branch = NumericalBranch()
        dimension = 96 if mode=='both' else (64 if mode=='image' else 32)
        self.classifier = nn.Sequential(nn.Linear(dimension,64),nn.BatchNorm1d(64),nn.ReLU(inplace=True),nn.Dropout(.5),
                                        nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(inplace=True),nn.Dropout(.3),nn.Linear(32,2))

    def forward(self,image,features):
        branches = []
        if self.mode!='numeric':
            branches.append(self.image_branch(image))
        if self.mode!='image':
            branches.append(self.numerical_branch(features))
        return self.classifier(torch.cat(branches,dim=1))


def prepare(config):
    frame = read_table(config['table'])
    if frame.shape[1]!=19 or not frame.iloc[:,-1].isin([0,1]).all() or frame.iloc[:,0].duplicated().any():
        raise ValueError('Expected unique IDs, 17 features and binary labels')
    if not np.isfinite(frame.iloc[:,1:-1].to_numpy(dtype=np.float32)).all():
        raise ValueError('Features must be finite')
    split_path = Path(config['split_file'])
    labels = frame.iloc[:,-1].to_numpy(dtype=np.int64)
    fingerprint = hashlib.sha256(Path(config['table']).read_bytes()).hexdigest()
    meta = split_path.with_suffix('.json')
    from .spatial_protocol import contour_indices
    indices, split, spatial_audit = contour_indices(frame, config)
    if split_path.exists():
        saved = pd.read_csv(split_path)
        if not meta.exists() or json.loads(meta.read_text(encoding='utf-8')).get('spatial_audit') != spatial_audit or json.loads(meta.read_text(encoding='utf-8'))['table_sha256']!=fingerprint:
            raise ValueError('Split metadata does not match the contour table')
        if not saved[['index','subset']].equals(split[['index','subset']]):
            raise ValueError('Existing random split cannot be reused for the Kekexili protocol; use a fresh split_file')
    else:
        split_path.parent.mkdir(parents=True,exist_ok=True)
        split.to_csv(split_path,index=False)
        meta.write_text(json.dumps({'table_sha256':fingerprint,'spatial_audit':spatial_audit},indent=2),encoding='utf-8')
    if split['index'].duplicated().any() or sorted(split['index'].tolist())!=list(range(len(frame))):
        raise ValueError('Split must cover each contour exactly once')
    if not split['subset'].isin(['train','validation','test','excluded']).all():
        raise ValueError('Invalid split names')
    indices = {name:split.loc[split['subset']==name,'index'].to_numpy(dtype=int) for name in ['train','validation','test']}
    if any(not len(indices[key]) for key in ['train','validation']):
        raise ValueError('All subsets must be nonempty')
    expected = config.get('expected_test_size')
    if expected is not None and len(indices['test'])!=expected:
        raise ValueError(f"Expected {expected} held-out contours, got {len(indices['test'])}")
    return frame,indices,fingerprint


def scores(labels,predictions):
    return {'accuracy':float(accuracy_score(labels,predictions)),
            'precision':float(precision_score(labels,predictions,zero_division=0)),
            'recall':float(recall_score(labels,predictions,zero_division=0)),
            'f1':float(f1_score(labels,predictions,zero_division=0)),
            'weighted_f1':float(f1_score(labels,predictions,average='weighted',zero_division=0)),
            'confusion_matrix_TN_FP_FN_TP':confusion_matrix(labels,predictions,labels=[0,1]).ravel().tolist()}


def validation(model,loader,criterion,device):
    model.eval()
    total,weight = 0.0,0
    labels,predictions = [],[]
    with torch.no_grad():
        for image,features,target in loader:
            logits = model(image.to(device),features.to(device))
            loss = criterion(logits,target.to(device))
            total += loss.item()*len(target)
            weight += len(target)
            labels.extend(target.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return {'loss':total/weight,**scores(labels,predictions)},labels,predictions


def train(config,variant):
    seed_training(config)
    frame,indices,fingerprint = prepare(config)
    features = list(frame.columns[1:-1])
    scaler = StandardScaler().fit(frame.iloc[indices['train']][features].to_numpy(dtype=np.float32))
    output = Path(config['output_dir'])/variant
    output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'feature_scaler.npz',mean=scaler.mean_,scale=scaler.scale_,feature_columns=np.asarray(features))
    batch = int(config.get('batch_size',32))
    loaders = {name:DataLoader(ContourDataset(frame,value,config['image_dir'],features,scaler,name=='train'),
                              batch_size=batch,shuffle=name=='train',drop_last=name=='train') for name,value in indices.items()}
    if not len(loaders['train']):
        raise ValueError('Training has no full batch; reduce batch_size')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AblationClassifier(variant).to(device)
    counts = np.bincount(frame.iloc[indices['train'],-1].to_numpy(dtype=int),minlength=2)
    if (counts==0).any():
        raise ValueError('Training needs both classes')
    weights = len(indices['train'])/(2*counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    optimizer = torch.optim.Adam(model.parameters(),lr=float(config.get('learning_rate',.001)),weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,factor=.5)
    best,history = -1.0,[]
    epochs = int(config.get('epochs',80))
    if epochs<1:
        raise ValueError('epochs must be positive')
    for epoch in range(1,epochs+1):
        model.train()
        losses = []
        for image,numeric,target in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image.to(device),numeric.to(device)),target.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        result,_,_ = validation(model,loaders['validation'],criterion,device)
        scheduler.step(result['loss'])
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':result,
                        'learning_rate':optimizer.param_groups[0]['lr']})
        if result['accuracy']>best:
            best = result['accuracy']
            torch.save({'model_state':model.state_dict(),'variant':variant,'epoch':epoch,
                        'feature_columns':features,'class_weights':weights.tolist(),'table_sha256':fingerprint,
                        'split_sha256':hashlib.sha256(Path(config['split_file']).read_bytes()).hexdigest()},output/'best.pt')
    (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')


def test(config,variant):
    frame,indices,fingerprint = prepare(config)
    output = Path(config['output_dir'])/variant
    if not len(indices['test']):
        report = {'variant':variant,'evaluation_unit':'external_contour_classification','test_size':0,
                  'status':'external_contour_test_not_supplied','metrics':None}
        (output/'test_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        return report
    saved = np.load(output/'feature_scaler.npz',allow_pickle=False)
    features = list(frame.columns[1:-1])
    if features!=saved['feature_columns'].tolist():
        raise ValueError('Feature order differs from training')
    scaler = StandardScaler()
    scaler.mean_,scaler.scale_,scaler.n_features_in_ = saved['mean'],saved['scale'],17
    dataset = ContourDataset(frame,indices['test'],config['image_dir'],features,scaler,False)
    loader = DataLoader(dataset,batch_size=int(config.get('batch_size',32)),shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(output/'best.pt',map_location=device,weights_only=False)
    split_hash = hashlib.sha256(Path(config['split_file']).read_bytes()).hexdigest()
    if checkpoint['table_sha256']!=fingerprint or checkpoint['split_sha256']!=split_hash or checkpoint['variant']!=variant:
        raise ValueError('Checkpoint/data/split mismatch')
    model = AblationClassifier(variant,initialize=False).to(device)
    model.load_state_dict(checkpoint['model_state'])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(checkpoint['class_weights'],dtype=torch.float32,device=device))
    result,labels,predictions = validation(model,loader,criterion,device)
    report = {'variant':variant,'evaluation_unit':'contour','test_size':len(labels),'metrics':result,
              'table_sha256':fingerprint,'split_sha256':split_hash,'selected_epoch':checkpoint['epoch']}
    (output/'test_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame({'id':frame.iloc[indices['test'],0].to_numpy(),'label':labels,'prediction':predictions}).to_csv(output/'test_predictions.csv',index=False)
    return report


def main(stage,variant):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if stage==1:
        prepare(config)
    elif stage==2:
        train(config,variant)
    else:
        print(json.dumps(test(config,variant),indent=2))
