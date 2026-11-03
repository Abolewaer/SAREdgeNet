import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto')
    args = parser.parse_args()
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif args.device.startswith('cuda:'):
        ordinal = args.device.split(':',1)[1]
        if not ordinal.isdigit():
            parser.error('Expected cpu, auto or cuda:N')
        os.environ['CUDA_VISIBLE_DEVICES'] = ordinal
    elif args.device!='auto':
        parser.error('Expected cpu, auto or cuda:N')
    from saredge.efficiency import benchmark_one
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    result = benchmark_one(spec,args.work_dir)
    if args.device.startswith('cuda:') and result['device']=='cpu':
        raise RuntimeError('Requested CUDA device is unavailable')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
