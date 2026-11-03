import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import ndimage as ndi

from .evaluation import evaluate,reference_mask
from .raster import read,binary,metre_grid

ROOT = Path(__file__).resolve().parents[1]


def tiny_lakes(prediction,reference):
    data,profile,valid = read(prediction)
    metre_grid(profile)
    truth,rv = reference_mask(reference,profile)
    valid &= rv
    target,count = ndi.label(truth & valid,np.ones((3,3)))
    pred,_ = ndi.label(binary(data,rgb_red=True) & valid,np.ones((3,3)))
    predicted_sizes = np.bincount(pred.ravel())
    rows = []
    for identifier,slices in enumerate(ndi.find_objects(target),1):
        region = target[slices]==identifier
        size = int(region.sum())
        if size*.0004 >= 1:
            continue
        ids,overlaps = np.unique(pred[slices][region],return_counts=True)
        positive = ids>0
        intersection = int(overlaps[positive].sum())
        predicted_area = int(predicted_sizes[ids[positive]].sum())
        union = size+predicted_area-intersection
        iou = intersection/union if union else 0.0
        status = 'missed' if intersection==0 else ('correct' if iou>=.95 else 'partial')
        rows.append({'lake_id':identifier,'area_km2':size*.0004,'iou':iou,'status':status})
    return rows


def main(stage,kind):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(config['output_dir']).resolve()/kind
    out.mkdir(parents=True,exist_ok=True)
    cases = config['cases']
    names = [c['name'] for c in cases]
    if not cases or len(names)!=len(set(names)) or any(Path(n).name!=n or n in ['.','..'] for n in names):
        raise ValueError('Cases need unique, simple folder names')
    if stage in [1,2]:
        for case in cases:
            directory = out/case['name']
            directory.mkdir(parents=True,exist_ok=True)
            current = {**case,'checkpoint':config['checkpoint'],'output_dir':str(directory)}
            path = directory/'case_config.json'
            path.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
            steps = [1,2,3,4] if stage==1 else [6,7,8,9]
            with (directory/f'stability_step{stage}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(ROOT/'Run_Step1.py'),'--config',str(path),'--steps',*map(str,steps)],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    else:
        evaluation = [{'name':case['name'],'prediction':str(out/case['name']/'step9/Polish.tif'),
                       'reference':case['reference'],'rgb_red':True} for case in cases if case.get('reference')]
        report = evaluate(evaluation,out/'metrics.json') if evaluation else {'region':None,'cases':[]}
        report['qualitative_cases'] = []
        for case in cases:
            if not case.get('reference'):
                path = out/case['name']/'step9/Polish.tif'
                data,profile,valid = read(path)
                metre_grid(profile)
                area = float((binary(data,rgb_red=True) & valid).sum()*.0004)
                report['qualitative_cases'].append({'name':case['name'],'overlay':str(path),'area_km2':area,
                                                    'accuracy_metrics':None})
        if kind=='TinyLakes':
            if len(evaluation)!=len(cases):
                raise ValueError('Tiny-lake omissions need a reference for every case')
            rows = [{'case':case['name'],**row} for case in evaluation for row in tiny_lakes(case['prediction'],case['reference'])]
            report['tiny_lakes'] = rows
            report['tiny_counts'] = {status:sum(r['status']==status for r in rows) for status in ['correct','partial','missed']}
        if kind=='Dunes' and report['region']:
            r = report['region']
            report['false_positive_rate'] = r['FP']/(r['FP']+r['TN']) if r['FP']+r['TN'] else None
        report['kind'] = kind
        report['training'] = 'fixed checkpoint; no retraining'
        (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
