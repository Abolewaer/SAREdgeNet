
import argparse
import json
import os
from protocol import SEEDS, VARIANTS, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['plan', 'prepare', 'train', 'matrix', 'aggregate', 'pipeline-requests', 'pixels'], default='plan')
    parser.add_argument('--variant', choices=list(VARIANTS))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--manifest', help='External scene input or binary pixel prediction manifest CSV')
    parser.add_argument('--run', action='store_true', help='Explicitly authorize requested processing stage')
    parser.add_argument('--allow-batch', action='store_true', help='Additional opt-in for all configured variants/seeds')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg['seeds'] != SEEDS:
        raise ValueError(f'Expected prespecified training seeds {SEEDS}; split_seed remains fixed separately')
    if args.stage == 'plan' or not args.run:
        print(json.dumps(dict(executed=False, stage=args.stage, variants=VARIANTS, seeds=SEEDS,
            note='Plan only. No table/images read, no split generated, no model initialized. --run is required.'), ensure_ascii=False, indent=2))
        return
                                                                       
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.stage == 'prepare':
        from protocol import prepare
        _, result = prepare(cfg)
    elif args.stage == 'train':
        if args.variant is None or args.seed is None:
            parser.error('train requires --variant and --seed')
        from training import train
        result = train(cfg, args.variant, args.seed)
    elif args.stage == 'matrix':
        if not args.allow_batch:
            parser.error('matrix requires both --run and --allow-batch; it launches 60 runs')
        from training import train
        result = [train(cfg, variant, seed) for seed in SEEDS for variant in VARIANTS]
    elif args.stage == 'aggregate':
        from training import aggregate
        result = aggregate(cfg, args.allow_partial)
    else:
        if args.variant is None or args.seed is None or args.manifest is None:
            parser.error('pipeline-requests/pixels require --variant --seed --manifest')
        from pixel_interface import export_requests, evaluate_pixels
        result = (export_requests if args.stage == 'pipeline-requests' else evaluate_pixels)(cfg, args.variant, args.seed, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
