import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
import argparse
import json
from UNet_Step1 import predict_large_tif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    Path(config['output_path']).parent.mkdir(parents=True,exist_ok=True)
    predict_large_tif(config['model_save_path'],config['tif_path'],config['output_path'],config)
    from saredge.raster import preserve_prediction_validity
    preserve_prediction_validity(config['output_path'],config['tif_path'])


if __name__ == '__main__':
    main()
