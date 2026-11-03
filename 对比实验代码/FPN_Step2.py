import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT/'.runtime'),str(_ROOT)]
from saredge.training import seed_training, segmentation_split
import argparse
import json
from pathlib import Path
from FPN_Step1 import predict_large_tif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    key = 'model_path' if 'model_path' in config else 'model_save_path'
    Path(config['output_path']).parent.mkdir(parents=True, exist_ok=True)
    predict_large_tif(config[key], config['tif_path'], config['output_path'], config)
    from saredge.raster import preserve_prediction_validity
    preserve_prediction_validity(config['output_path'],config['tif_path'])


if __name__ == '__main__':
    main()
