import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
from saredge.raster import align
from saredge.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--reference-grid',required=True)
    parser.add_argument('--reference-label',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--metrics',required=True)
    parser.add_argument('--positive-value',type=int,default=1)
    parser.add_argument('--boundary',action='store_true')
    parser.add_argument('--spatial-protocol')
    parser.add_argument('--spatial-subset',choices=['validation','test','external'],default='test')
    args = parser.parse_args()
    align(args.input,args.reference_grid,args.output,categorical=True)
    evaluate([{'prediction':args.output,'reference':args.reference_label,
               'positive_value':args.positive_value,'validity_source':args.reference_grid,
               'spatial_subset':args.spatial_subset,'spatial_protocol_file':args.spatial_protocol}],args.metrics,args.boundary)


if __name__ == '__main__':
    main()
