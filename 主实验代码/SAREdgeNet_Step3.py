import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
import argparse
from saredge.spatial import aligned_dem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dem', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    aligned_dem(args.dem,args.reference,args.output)


if __name__ == '__main__':
    main()
