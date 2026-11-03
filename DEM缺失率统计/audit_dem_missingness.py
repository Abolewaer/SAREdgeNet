import os
"""Read-only raster completeness audit; never imports project/training modules."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv,json,time,warnings
import numpy as np
import rasterio
from rasterio.windows import Window,from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

ROOT=Path(os.environ['SAREDGE_ARCHIVE_ROOT'])
OUT=ROOT/'湖泊论文/Water投稿/外审返修_20261022/第10点_DEM缺失核查_20261024'
OUT.mkdir(exist_ok=True)
TILE_ROOT=ROOT/'WorkDir/After_Step3DEM'
RAW=[ROOT/'WorkDir/DEM/Merge/Merge.tif',ROOT/'湖泊论文/LakeDetection/DEM/QZGY/QZGY.tif']
def windows(width,height,chunk=1024):
    for y in range(0,height,chunk):
        for x in range(0,width,chunk):
            yield Window(x,y,min(chunk,width-x),min(chunk,height-y))
def classify(a,m,nodata):
    nf=~np.isfinite(a)
    nd=np.zeros(a.shape,bool) if nodata is None else (np.isnan(a) if np.isnan(nodata) else a==nodata)
    return {'nan':np.isnan(a),'inf':np.isinf(a),'nodata_equal':nd,'mask_invalid':m==0,'missing_union':nf|nd|(m==0),'zero':a==0}
def summarize(counts):
    d=dict(counts)
    for k in ['nan','inf','nodata_equal','mask_invalid','missing_union','zero']:
        d[k+'_pct']=100.0*d.get(k,0)/d['total'] if d.get('total') else None
    return d
def profile(path):
    c=Counter()
    with rasterio.open(path) as s:
        for w in windows(s.width,s.height):
            a=s.read(1,window=w);m=s.read_masks(1,window=w)
            c['total']+=a.size
            for k,v in classify(a,m,s.nodata).items():c[k]+=int(np.count_nonzero(v))
        return {'file':str(path),'width':s.width,'height':s.height,'dtype':s.dtypes[0],
                'nodata':s.nodata,'crs':str(s.crs),'transform':list(s.transform),**summarize(c)}

report={'created':datetime.now().isoformat(),'scope':'Local archived real DEM mosaics and 18 processed DEM tiles; not a claim covering all historical scenes',
        'definitions':{'missing_union':'Nonfinite OR dataset NoData equality OR dataset validity mask = 0',
          'raw_dem_in_tile_footprint':'Source DEM cells within archived processed-tile bounding box, clipped to source extent',
          'sar_reference_footprint':'SAR nearest-sampled on the DEM grid: finite, nonzero, valid mask; this is a coarse-grid eligibility definition, not full-resolution SAR pixel coverage',
          'aggregation':'Per-file pixel observations summed; overlapping geographic cells are not deduplicated'},
        'raw_mosaics':[],'processed_tiles':[],'feature_tables':[],
        'cautions':['Archived processed DEMs are uint8 normalized, not elevation metres.',
          'A normalized value of zero may coincide with valid minima or clipped values; processed NoData is not identical to original DEM acquisition gaps.',
          'Recovered original code and organized migration code differ; this audit does not certify historical IQR or NaN handling.',
          'Synthetic, smoke-test and simulation folders excluded. No project modules imported or executed.']}
started=time.perf_counter()
for p in RAW:
    print('Counting raw mosaic:',p,flush=True)
    r=profile(p);report['raw_mosaics'].append(r)
    print('Raw result:',json.dumps({k:r[k] for k in ['total','missing_union','missing_union_pct','nan','zero']},ensure_ascii=False),flush=True)
    (OUT/'audit_progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

with rasterio.open(RAW[0]) as raw:
    for p in sorted(TILE_ROOT.glob('*.tif')):
        print('Counting processed tile:',p.name,flush=True)
        r=profile(p)
        with rasterio.open(p) as dem:
            # Count source-DEM NoData over the same geographic rectangle at its native resolution.
            fw=from_bounds(*dem.bounds,transform=raw.transform).round_offsets().round_lengths()
            fw=fw.intersection(Window(0,0,raw.width,raw.height))
            c=Counter()
            for local in windows(int(fw.width),int(fw.height)):
                w=Window(fw.col_off+local.col_off,fw.row_off+local.row_off,local.width,local.height)
                a=raw.read(1,window=w);m=raw.read_masks(1,window=w);c['total']+=a.size
                for k,v in classify(a,m,raw.nodata).items():c[k]+=int(np.count_nonzero(v))
            r['raw_dem_in_tile_footprint']=summarize(c)
            sar_name=p.name.replace('_dem_processed.tif','.tif')
            sar_path=ROOT/'WorkDir/After_Step1/Over'/sar_name
            if sar_path.exists():
                c=Counter()
                with rasterio.open(sar_path) as sar, WarpedVRT(sar,crs=dem.crs,transform=dem.transform,
                    width=dem.width,height=dem.height,resampling=Resampling.nearest) as sv:
                    r['sar_reference']={'file':str(sar_path),'width':sar.width,'height':sar.height,'nodata':sar.nodata}
                    for w in windows(dem.width,dem.height):
                        a=dem.read(1,window=w);m=dem.read_masks(1,window=w)
                        v=sv.read(1,window=w);vm=sv.read_masks(1,window=w)
                        eligible=np.isfinite(v)&(vm>0)&(v!=0)
                        c['total']+=int(eligible.sum())
                        for k,z in classify(a,m,dem.nodata).items():c[k]+=int(np.count_nonzero(z&eligible))
                r['processed_dem_within_sar_reference']=summarize(c)
        report['processed_tiles'].append(r)
        print('Processed missing %:',r['missing_union_pct'],'native DEM within tile %:',r['raw_dem_in_tile_footprint']['missing_union_pct'],flush=True)
        (OUT/'audit_progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

# Feature-table audit is separate from raster-pixel completeness.
for p in sorted((ROOT/'WorkDir/After_Step4Dog').glob('*/contour_statistics_enhanced_with_dem.csv')):
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames or []
        dem_fields=[s for s in fields if 'DEM' in s]
        c=Counter()
        for row in reader:
            c['rows']+=1;bad=False
            for field in dem_fields:
                try:v=float(row[field]);ok=bool(np.isfinite(v))
                except (ValueError,TypeError):ok=False
                if not ok:c[field+'_missing']+=1;bad=True
            c['rows_with_nonfinite_dem_descriptor']+=int(bad)
            try:c['rows_with_zero_dem_count']+=int(float(row.get('DEM有效像素数','nan'))==0)
            except (ValueError,TypeError):pass
        report['feature_tables'].append({'file':str(p),'dem_fields':dem_fields,**dict(c)})

for key,subkey in [('processed_tile_observations',None),('source_dem_tile_footprint_observations','raw_dem_in_tile_footprint'),('processed_dem_sar_eligible_observations','processed_dem_within_sar_reference')]:
    c=Counter()
    for item in report['processed_tiles']:
        d=item if subkey is None else item.get(subkey,{})
        for k in ['total','nan','inf','nodata_equal','mask_invalid','missing_union','zero']:c[k]+=d.get(k,0)
    report[key]=summarize(c)
c=Counter()
for t in report['feature_tables']:
    for k in ['rows','rows_with_nonfinite_dem_descriptor','rows_with_zero_dem_count']:c[k]+=t.get(k,0)
report['feature_table_totals']=dict(c)
report['elapsed_seconds']=time.perf_counter()-started
(OUT/'DEM_missingness_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
rows=[]
for a in report['processed_tiles']:
    rows.append({'file':a['file'],'pixels':a['total'],'missing':a['missing_union'],'missing_pct':a['missing_union_pct'],
      'raw_footprint_pixels':a['raw_dem_in_tile_footprint']['total'],'raw_footprint_missing':a['raw_dem_in_tile_footprint']['missing_union'],
      'raw_footprint_missing_pct':a['raw_dem_in_tile_footprint']['missing_union_pct'],
      'sar_eligible_pixels':a.get('processed_dem_within_sar_reference',{}).get('total'),
      'sar_eligible_missing_pct':a.get('processed_dem_within_sar_reference',{}).get('missing_union_pct')})
with (OUT/'DEM_missingness_by_tile.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:report[k] for k in ['processed_tile_observations','source_dem_tile_footprint_observations','processed_dem_sar_eligible_observations','feature_table_totals','elapsed_seconds']},ensure_ascii=False,indent=2),flush=True)
