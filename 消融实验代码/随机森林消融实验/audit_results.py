from pathlib import Path
import ast
import argparse
import hashlib
import io
import json
import tokenize
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,confusion_matrix,f1_score,precision_score,recall_score,accuracy_score,roc_auc_score

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-checkpoints',action='store_true')
    parser.add_argument('--require-source-table',action='store_true')
    args=parser.parse_args()
    audit=json.loads((HERE/'data/data_audit.json').read_text(encoding='utf8'))
    manifest=pd.read_csv(HERE/'data/shared_manifest.csv',dtype={'sample_id':str})
    assert sha(HERE/'data/shared_manifest.csv')==audit['manifest_sha256']
    assert sha(HERE/'data/shared_split.csv')==audit['split_sha256']
    source=Path(audit['source_table'])
    if args.require_source_table or source.is_file():
        assert sha(source)==audit['source_table_sha256']
    records=[];models=['random_forest','numeric_mlp','full_dual_branch']
    log=[];checkpoints=[]
    for model in models:
        folder=HERE/'outputs'/model
        result=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
        assert result['split_sha256']==audit['split_sha256'] and result['manifest_sha256']==audit['manifest_sha256']
        assert result['seed']==42 and result['n_runs']==1 and result['independent_scene_test'] is False
        checkpoint=folder/('model.joblib' if model=='random_forest' else 'best.pt')
        if args.require_checkpoints and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoints.append(dict(model=model,path=str(checkpoint),available=checkpoint.is_file(),
                                sha256=sha(checkpoint) if checkpoint.is_file() else None))
        log.append(f"{model}: seed=42, selected_epoch={result['selected_epoch']}, elapsed_seconds={result['elapsed_seconds']:.6f}")
        history=folder/'history.json'
        if history.exists():
            for epoch in json.loads(history.read_text(encoding='utf8')):
                log.append(f"epoch={epoch['epoch']}, train_loss={epoch['training_loss']:.9f}, validation_F1={epoch['validation']['f1']:.9f}, validation_weighted_F1={epoch['validation']['weighted_f1']:.9f}, seconds={epoch['epoch_seconds']:.6f}")
        for subset,key in [('validation','validation'),('test','spatial_test_same_scene')]:
            prediction=pd.read_csv(folder/f'{subset}_predictions.csv',dtype={'sample_id':str})
            expected=manifest.loc[manifest.subset==subset,['sample_id','label']]
            assert not prediction.sample_id.duplicated().any()
            assert set(prediction.sample_id)==set(expected.sample_id)
            joined=expected.merge(prediction,on='sample_id',validate='one_to_one',suffixes=('_true','_saved'))
            assert np.array_equal(joined.label_true,joined.label_saved)
            p=joined.lake_probability.to_numpy(float);y=joined.label_true.to_numpy(int);pred=(p>=.5).astype(int)
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.array_equal(pred,joined.prediction)
            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
            recomputed=dict(TN=tn,FP=fp,FN=fn,TP=tp,precision=precision_score(y,pred,zero_division=0),
                recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),
                accuracy=accuracy_score(y,pred),weighted_f1=f1_score(y,pred,average='weighted',zero_division=0),
                pr_auc_AP=average_precision_score(y,p),roc_auc=roc_auc_score(y,p))
            for metric,value in recomputed.items():assert abs(value-result[key][metric])<1e-12,(model,subset,metric)
            records.append(dict(model=model,subset=subset,n=len(prediction),predictions_sha256=sha(folder/f'{subset}_predictions.csv'),
                                shared_ids_and_labels_verified=True,all_metrics_recomputed=True))
    checks=[]
    for path in HERE.glob('*.py'):
        source=path.read_text(encoding='utf8');tree=ast.parse(source);compile(source,str(path),'exec')
        # Syntax is checked without imposing a style restriction on comments.
        checks.append(dict(file=path.name,syntax='passed',sha256=sha(path)))
    report=dict(status='passed',same_samples_same_labels_same_split=True,prediction_checks=records,source_checks=checks,
        checkpoint_hashes=checkpoints,
        models=models,n_runs_per_model=1,pr_auc_definition='noninterpolated Average Precision (AP)',
        independent_scene_test=False,scope=audit['external_scope'],synthetic_data_used=False)
    (HERE/'outputs/verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (HERE/'outputs/run_log_from_saved_history.txt').write_text('Exported from actual metrics.json and history.json; not a reconstructed raw console transcript.\n\n'+'\n'.join(log),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
