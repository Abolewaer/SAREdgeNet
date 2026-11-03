import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
import argparse
from saredge.spatial import geolocation, attach_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dx',type=float,default=0.0)
    parser.add_argument('--dy',type=float,default=0.0)
    parser.add_argument('--reference')
    parser.add_argument('--copy-grid-only',action='store_true')
    parser.add_argument('--target-crs')
    parser.add_argument('--resolution',type=float,default=20.0)
    args = parser.parse_args()
    if args.copy_grid_only:
        if not args.reference or args.dx or args.dy or args.target_crs:
            parser.error('--copy-grid-only requires --reference and zero dx/dy')
        attach_grid(args.input,args.reference,args.output)
    else:
        geolocation(args.input,args.output,args.dx,args.dy,args.target_crs,args.resolution)


if __name__ == '__main__':
    main()
