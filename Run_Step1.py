import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def commands(config,steps):
    out = Path(config['output_dir']).resolve()
    sar = out/'step2/sar.tif'
    dem = out/'step3/dem.tif'
    contours = out/'step4'
    classifier = Path(config.get('classifier_dir',str(out/'step5'))).resolve()
    checkpoint = config.get('checkpoint',str(classifier/'best_classifier.pt'))
    selected = out/'step6/positive_contours.tif'
    source_contours = Path(config.get('contour_dir',str(contours))).resolve()
    source_sar = config.get('processed_sar',str(sar))
    source_support = config.get('support_mask',str(source_contours/'step7_filtered_mask3.tif'))
    specs = {
        1:['--input',config['sar'],'--output',out/'step1/sar.tif'],
        2:['--input',out/'step1/sar.tif','--output',sar,'--dx',config.get('dx',0),'--dy',config.get('dy',0)],
        3:['--dem',config['dem'],'--reference',sar,'--output',dem],
        4:['--sar',sar,'--dem',dem,'--output',contours],
        5:['--table',config.get('train_table',''),'--image-dir',config.get('train_image_dir',''),
           '--output-dir',classifier,'--epochs',config.get('epochs',10000),'--batch-size',config.get('batch_size',32),
           '--spatial-protocol',config.get('spatial_protocol_file',str(ROOT/'configs/SpatialProtocol.json')),
           '--spatial-samples',config.get('spatial_samples_file','')],
        6:['--table',source_contours/'contour_statistics_enhanced_with_dem.csv','--image-dir',source_contours,
           '--checkpoint',checkpoint,'--output',out/'step6/predictions.csv','--candidate-ids',source_contours/'candidate_ids.tif',
           '--projected-mask',selected],
        7:['--contours',selected,'--support',source_support,'--source',source_sar,'--output-dir',out/'step7',
           '--seed-max-intensity',config.get('seed_max_intensity',50),'--seed-tolerance',config.get('seed_tolerance',30)],
        8:['--contours',selected,'--step7-dir',out/'step7','--seed-grow',config.get('seed_grow',str(out/'step7/SeedGrow.tif')),
           '--source',source_sar,'--output-dir',out/'step8'],
        9:['--input',out/'step8/Final.tif','--output',out/'step9/Polish.tif'],
        10:['--input',out/'step9/Polish.tif','--output-dir',out/'step10']}
    if config.get('reference'):
        specs[10] += ['--reference',config['reference'],'--validity-source',source_sar,
                      '--spatial-protocol',config.get('spatial_protocol_file',str(ROOT/'configs/SpatialProtocol.json')),
                      '--spatial-subset',config.get('evaluation_subset','test')]
    if config.get('target_crs'):
        specs[2] += ['--target-crs',config['target_crs'],'--resolution',config.get('resolution',20)]
    if 5 in steps and (not config.get('train_table') or not config.get('train_image_dir')):
        raise ValueError('Step5 needs a separately labelled training table and contour folder')
    return [(step,[sys.executable,str(ROOT/f'主实验代码/SAREdgeNet_Step{step}.py'),*map(str,specs[step])]) for step in steps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--steps',type=int,nargs='+',default=list(range(1,11)))
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    if len(set(args.steps)) != len(args.steps) or any(s not in range(1,11) for s in args.steps):
        parser.error('Steps must be unique and between 1 and 10')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    plan = commands(config,args.steps)
    out = Path(config['output_dir'])
    out.mkdir(parents=True,exist_ok=True)
    report = []
    for step,command in plan:
        if not args.dry_run:
            print(f'Running Step{step}',flush=True)
            with (out/f'step{step}.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report.append({'step':step,'command':command,'status':'planned' if args.dry_run else 'complete'})
        (out/'run_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
