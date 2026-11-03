import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from saredge.raster import read, binary, metre_grid
from saredge.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--reference')
    parser.add_argument('--binary-input',action='store_true')
    parser.add_argument('--positive-value',type=int,default=1)
    parser.add_argument('--boundary',action='store_true')
    parser.add_argument('--validity-source')
    parser.add_argument('--spatial-protocol')
    parser.add_argument('--spatial-subset',choices=['validation','test','external'],default='test')
    args = parser.parse_args()
    data,profile,valid = read(args.input)
    metre_grid(profile)
    mask = binary(data,args.positive_value,not args.binary_input) & valid
    labels,count = ndi.label(mask,np.ones((3,3)))
    sizes = np.bincount(labels.ravel())[1:]
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'component_id':np.arange(1,count+1),'pixels':sizes,
                  'area_km2':sizes*0.0004}).to_csv(output/'areas.csv',index=False)
    summary = {'lake_pixels':int(mask.sum()),'area_km2':float(mask.sum()*0.0004),
               'connected_components':count,'pixel_area_m2':400}
    (output/'area_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if args.reference:
        if not args.validity_source:
            raise ValueError('Reference evaluation requires --validity-source pointing to the independent SAR coverage raster')
        evaluate([{'prediction':args.input,'reference':args.reference,'rgb_red':not args.binary_input,
                   'positive_value':args.positive_value,'validity_source':args.validity_source,
                   'spatial_subset':args.spatial_subset,'spatial_protocol_file':args.spatial_protocol}],output/'metrics.json',args.boundary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
