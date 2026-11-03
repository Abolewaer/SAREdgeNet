import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def collect():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--workflow')
    source.add_argument('--config')
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.repeats<1:
        parser.error('--repeats must be positive')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    samples,runs,wall = [],[],[]
    if args.workflow:
        workflow = json.loads(Path(args.workflow).read_text(encoding='utf-8'))
        if not workflow.get('commands'):
            raise ValueError('Workflow commands are empty')
    else:
        spec = json.loads(Path(args.config).read_text(encoding='utf-8'))
        spec_path = output.parent/(output.stem+'_benchmark_spec.json')
        spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    for repeat in range(args.repeats):
        log_path = output.parent/f'{output.stem}_run_{repeat}.log'
        start = time.perf_counter()
        with log_path.open('w',encoding='utf-8') as log:
            if args.workflow:
                for command in workflow['commands']:
                    subprocess.run(command,cwd=workflow.get('cwd'),stdout=log,stderr=subprocess.STDOUT,check=True)
            else:
                report_path = output.parent/f'{output.stem}_run_{repeat}.json'
                work = output.parent/f'{output.stem}_work_{repeat}'
                subprocess.run([sys.executable,str(ROOT/'saredge/efficiency_worker.py'),'--spec',str(spec_path),
                                '--work-dir',str(work),'--output',str(report_path),'--device',args.device],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                run = json.loads(report_path.read_text(encoding='utf-8'))
                runs.append(run)
                samples.append(run['seconds'])
        external = time.perf_counter()-start
        wall.append(external)
        if args.workflow:
            samples.append(external)
        report = {'seconds':samples,'repetitions':len(samples),'model_warmup':False,'runs':runs,
                  'external_process_wall_seconds':wall,
                  'scope':'legacy complete child-command wall time' if args.workflow else spec.get('scope','end_to_end'),
                  'method':None if args.workflow else spec['method'],
                  'startup_included_in_primary_seconds':bool(args.workflow),
                  'training_time_estimate':None}
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(output))


def summarize():
    from .efficiency import summary
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epoch-times')
    parser.add_argument('--planned-epochs',type=int)
    args = parser.parse_args()
    raw = json.loads(Path(args.samples).read_text(encoding='utf-8'))
    durations = json.loads(Path(args.epoch_times).read_text(encoding='utf-8')) if args.epoch_times else None
    report = summary(raw,durations,args.planned_epochs)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
