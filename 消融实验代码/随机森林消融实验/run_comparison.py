from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,average_precision_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torchvision
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'消融实验代码'/'新消融实验'))
from networks import NumericBranch,Classifier


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def metrics(y,p):
    prediction=np.asarray(p)>=.5
    tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel().tolist()
    return dict(TN=tn,FP=fp,FN=fn,TP=tp,accuracy=float(accuracy_score(y,prediction)),
        precision=float(precision_score(y,prediction,zero_division=0)),recall=float(recall_score(y,prediction,zero_division=0)),
        f1=float(f1_score(y,prediction,zero_division=0)),weighted_f1=float(f1_score(y,prediction,average='weighted',zero_division=0)),
        pr_auc_AP=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)))


class NumericClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.numeric=NumericBranch(17,attention=True)
        self.classifier=nn.Sequential(nn.Linear(32,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(.5),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,2))
    def forward(self,image,numeric):return self.classifier(self.numeric(numeric))


class Data(Dataset):
    def __init__(self,frame,values,image_cache,train,seed):
        self.frame=frame.reset_index(drop=True);self.values=values.astype(np.float32)
        self.images=image_cache;self.train=train;self.seed=seed;self.epoch=0
        self.transform=transforms.Compose(([transforms.RandomHorizontalFlip(.3),transforms.RandomRotation(10)] if train else [])+
            [transforms.ToTensor(),transforms.Normalize([.485],[.229])])
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        if self.images is None:image=torch.empty(0)
        else:
            im=Image.fromarray(self.images[str(row.sample_id)])
            if self.train:
                token=f'{self.seed}:{self.epoch}:{row.sample_id}'.encode('utf8')
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(int(hashlib.sha256(token).hexdigest()[:15],16))
                    image=self.transform(im)
            else:image=self.transform(im)
        return image,torch.tensor(self.values[i]),int(row.label)


def evaluate(model,loader,device,weights):
    model.eval();prob=[];truth=[];numerator=denominator=0.
    with torch.no_grad():
        for image,numeric,target in loader:
            target=target.to(device);logits=model(image.to(device),numeric.to(device))
            numerator+=nn.functional.cross_entropy(logits,target,weight=weights,reduction='sum').item()
            denominator+=weights[target].sum().item()
            prob.extend(logits.softmax(1)[:,1].cpu().tolist());truth.extend(target.cpu().tolist())
    return np.array(prob),dict(loss=numerator/denominator,**metrics(truth,prob))


def save_predictions(folder,subset,frame,prob):
    out=frame[['sample_id','label']].copy()
    out['lake_probability']=prob;out['prediction']=(np.asarray(prob)>=.5).astype(int)
    out.to_csv(folder/f'{subset}_predictions.csv',index=False,encoding='utf-8-sig')
    return metrics(out.label.to_numpy(),prob)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=['random_forest','numeric_mlp','full_dual_branch'],default=['random_forest'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'))
    audit=json.loads((HERE/'data/data_audit.json').read_text(encoding='utf8'))
    frame=pd.read_csv(HERE/'data/shared_manifest.csv',dtype={'sample_id':str})
    # Historical CSV bytes are retained for their recorded hashes. Resolve
    # relocated image files without rewriting the archived feature/label table.
    frame['image_path'] = frame.apply(lambda row: str(ROOT/'data/contours'/f'{row.sample_id}.png'), axis=1)
    features=audit['feature_columns']
    assert len(features)==17 and np.isfinite(frame[features].to_numpy(float)).all()
    if sha(HERE/'data/shared_manifest.csv')!=audit['manifest_sha256'] or sha(HERE/'data/shared_split.csv')!=audit['split_sha256']:
        raise ValueError('Real data or fixed split changed')
    parts={s:frame[frame.subset==s].copy() for s in ['train','validation','test']}
    if any(set(p.label)!={0,1} for p in parts.values()):raise ValueError('Both classes required in each real comparison subset')
    scaler=StandardScaler().fit(parts['train'][features].to_numpy(float))
    output=HERE/'outputs';output.mkdir(exist_ok=True)
    joblib.dump(scaler,output/'scaler_train_only.joblib')
    values={s:scaler.transform(p[features].to_numpy(float)).astype(np.float32) for s,p in parts.items()}
    counts=np.bincount(parts['train'].label.to_numpy(int),minlength=2)
    weights_array=len(parts['train'])/(2*counts)
    protocol=dict(config=cfg,config_sha256=sha(HERE/'config.json'),feature_columns=features,
        manifest_sha256=audit['manifest_sha256'],split_sha256=audit['split_sha256'],
        source_table_sha256=audit['source_table_sha256'],class_counts={s:p.label.value_counts().to_dict() for s,p in parts.items()},
        class_weights_train=weights_array.tolist(),evaluation_unit='contour_classification',
        pr_auc_definition='sklearn average_precision_score: noninterpolated average precision (AP), not trapezoidal integration',
        threshold=.5,external_scope=audit['external_scope'],independent_scene_test=False,
        environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__,
            torchvision=torchvision.__version__,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    existing=output/'protocol.json'
    if existing.exists():
        old=json.loads(existing.read_text(encoding='utf8'))
        if old!=json.loads(json.dumps(protocol)):raise ValueError('Comparison protocol differs from existing outputs')
    else:dump(existing,protocol)
    for name in args.models:
        destination=output/name
        if (destination/'metrics.json').exists():
            print(f'Reusing completed {name}',flush=True);continue
        if destination.exists():raise FileExistsError(f'Incomplete run retained, do not silently overwrite: {destination}')
        destination.mkdir();start=time.perf_counter()
        seed=int(cfg['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        dump(destination/'run_record.json',dict(status='started',model=name,seed=seed,protocol=protocol))
        print(f'Starting {name}',flush=True)
        if name=='random_forest':
            model=RandomForestClassifier(random_state=seed,**cfg['random_forest'])
            model.fit(values['train'],parts['train'].label.to_numpy(int))
            vp=model.predict_proba(values['validation'])[:,list(model.classes_).index(1)]
            tp=model.predict_proba(values['test'])[:,list(model.classes_).index(1)]
            joblib.dump(model,destination/'model.joblib')
            selected_epoch=None
        else:
            if not torch.cuda.is_available():raise RuntimeError('CUDA required for the declared deep-control run')
            device=torch.device('cuda:0');tr=cfg['deep_training'];image_cache=None
            if name=='numeric_mlp':model=NumericClassifier().to(device)
            else:
                weights_path=Path(cfg['imagenet_weights'])
                if not weights_path.is_file():raise FileNotFoundError(weights_path)
                model=Classifier(dict(columns=features,attention=True),str(weights_path)).to(device)
                image_cache={}
                for row in frame.loc[frame.subset.isin(parts)].itertuples(index=False):
                    with Image.open(row.image_path) as im:image_cache[str(row.sample_id)]=np.array(im.convert('L').resize((224,224),Image.Resampling.LANCZOS))
                dump(destination/'image_initialization.json',dict(image_content='binary_contour',source='Original manually labelled contour PNGs',
                    resize='224x224 LANCZOS; cached before identical train-only augmentation',imagenet_weights_sha256=sha(weights_path)))
            datasets={s:Data(p,values[s],image_cache,s=='train',seed) for s,p in parts.items()}
            generator=torch.Generator().manual_seed(seed)
            loaders={s:DataLoader(d,batch_size=int(tr['batch_size']),shuffle=s=='train',drop_last=s=='train',num_workers=0,
                generator=generator if s=='train' else None) for s,d in datasets.items()}
            weights=torch.tensor(weights_array,dtype=torch.float32,device=device)
            optimizer=torch.optim.Adam(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay'])
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,factor=.5,min_lr=1e-6)
            best=-1.;history=[]
            for epoch in range(1,int(tr['epochs'])+1):
                tick=time.perf_counter();datasets['train'].epoch=epoch;model.train();losses=[]
                for image,numeric,target in loaders['train']:
                    optimizer.zero_grad(set_to_none=True)
                    loss=nn.functional.cross_entropy(model(image.to(device),numeric.to(device)),target.to(device),weight=weights)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss.item())
                _,val=evaluate(model,loaders['validation'],device,weights);scheduler.step(val['loss'])
                history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation=val,epoch_seconds=time.perf_counter()-tick,
                    learning_rate=optimizer.param_groups[0]['lr']))
                if val['weighted_f1']>best:
                    best=val['weighted_f1'];selected_epoch=epoch
                    torch.save(dict(model_state=model.state_dict(),selected_epoch=epoch,variant=name,columns=features,
                        split_sha256=audit['split_sha256']),destination/'best.pt')
                dump(destination/'history.json',history)
                if epoch==1 or epoch%5==0:print(f'{name} epoch {epoch}/{tr["epochs"]} val F1={val["f1"]:.4f} weightedF1={val["weighted_f1"]:.4f} seconds={history[-1]["epoch_seconds"]:.1f}',flush=True)
            checkpoint=torch.load(destination/'best.pt',map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model_state']);selected_epoch=checkpoint['selected_epoch']
            vp,_=evaluate(model,loaders['validation'],device,weights)
            tp,_=evaluate(model,loaders['test'],device,weights)
        report=dict(model=name,seed=seed,n_runs=1,evaluation_unit='contour_classification',threshold=.5,selected_epoch=selected_epoch,
            validation=save_predictions(destination,'validation',parts['validation'],vp),
            spatial_test_same_scene=save_predictions(destination,'test',parts['test'],tp),
            independent_scene_test=False,external_scope=audit['external_scope'],
            elapsed_seconds=time.perf_counter()-start,split_sha256=audit['split_sha256'],manifest_sha256=audit['manifest_sha256'])
        dump(destination/'metrics.json',report)
        dump(destination/'run_record.json',dict(status='completed',**report))
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    reports=[json.loads(p.read_text(encoding='utf8')) for p in output.glob('*/metrics.json')]
    rows=[]
    for r in reports:
        for subset in ['validation','spatial_test_same_scene']:
            rows.append(dict(model=r['model'],subset=subset,seed=r['seed'],n_runs=1,selected_epoch=r['selected_epoch'],**r[subset]))
    pd.DataFrame(rows).to_csv(output/'comparison_metrics.csv',index=False,encoding='utf-8-sig')
    dump(output/'comparison_metrics.json',dict(evaluation_unit='contour_classification',independent_scene_test=False,results=reports))


if __name__=='__main__':main()
