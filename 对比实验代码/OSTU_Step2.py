import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
from saredge.evaluation import main

if __name__ == '__main__':
    main(default_rgb=True,require_validity_source=True,spatial_default='test')
