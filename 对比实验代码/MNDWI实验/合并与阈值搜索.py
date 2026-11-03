import os
from pathlib import Path
import zipfile,json,csv,re
import numpy as np
from osgeo import gdal,ogr,osr
gdal.UseExceptions()
ROOT=Path(__file__).parent; R=Path(os.environ['SAREDGE_MNDWI_WORK']); OUT=R/'阈值分析'; OUT.mkdir(parents=True,exist_ok=True)
T=R/'解压分片'; T.mkdir(exist_ok=True)
gdal.SetCacheMax(512*1024*1024)
entries={}; duplicates=[]
for z in sorted(R.glob('*.zip')):
 with zipfile.ZipFile(z) as f:
  for i in f.infolist():
   if not re.fullmatch(r'MNDWI_2022_Int16_tile_\d{2}_of_24.tif',i.filename):continue
   if i.filename in entries:
    assert entries[i.filename]['crc']==i.CRC,'Conflicting duplicate'
    duplicates.append(i.filename); continue
   entries[i.filename]={'zip':z.name,'crc':i.CRC,'bytes':i.file_size}
   dest=T/i.filename
   if not dest.exists() or dest.stat().st_size!=i.file_size:
    print('Extract',i.filename,flush=True); f.extract(i,T)
files=sorted(T.glob('MNDWI_2022_Int16_tile_*_of_24.tif'))
missing=[i for i in range(1,25) if not (T/f'MNDWI_2022_Int16_tile_{i:02d}_of_24.tif').exists()]
audit={'entries':entries,'duplicates':duplicates,'missing_tiles':missing}
(OUT/'输入核查.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
for p in files:
 d=gdal.Open(str(p)); assert d.RasterCount==1 and d.GetRasterBand(1).DataType==gdal.GDT_Int16
 assert d.GetGeoTransform()[1]==20 and d.GetGeoTransform()[5]==-20
 assert d.GetRasterBand(1).GetNoDataValue()==-32768
v=gdal.BuildVRT(str(OUT/'MNDWI.vrt'),list(map(str,files)),srcNodata=-32768,VRTNodata=-32768); v=None
mosaic=OUT/('MNDWI_2022_完整.tif' if not missing else 'MNDWI_2022_现有23片_缺24.tif')
if not mosaic.exists():
 print('Mosaic',flush=True)
 gdal.Translate(str(mosaic),str(OUT/'MNDWI.vrt'),creationOptions=['TILED=YES','COMPRESS=LZW','BIGTIFF=YES','NUM_THREADS=2'])
print('Mosaic done',flush=True)
# Select threshold only on the development region. This is not the formal held-out validation patch split.
sar=Path(os.environ['SAREDGE_ROI'])
s=gdal.Open(str(sar)); gt=s.GetGeoTransform(); w=s.RasterXSize; h=s.RasterYSize
bounds=[gt[0],gt[3]+h*gt[5],gt[0]+w*gt[1],gt[3]]
mp=OUT/'可可西里_MNDWI_对齐SAR.tif'
gdal.Warp(str(mp),str(mosaic),dstSRS=s.GetProjection(),outputBounds=bounds,width=w,height=h,resampleAlg='near',srcNodata=-32768,dstNodata=-32768,creationOptions=['TILED=YES','COMPRESS=LZW','BIGTIFF=YES'])
ref=OUT/'可可西里_参考湖面.tif'
lab=gdal.GetDriverByName('GTiff').Create(str(ref),w,h,1,gdal.GDT_Byte,options=['TILED=YES','COMPRESS=LZW'])
lab.SetGeoTransform(gt);lab.SetProjection(s.GetProjection());lab.GetRasterBand(1).Fill(0)
vec=ogr.Open(os.environ['SAREDGE_REFERENCE']);layer=vec.GetLayer()
target=osr.SpatialReference();target.ImportFromWkt(s.GetProjection());target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
source=layer.GetSpatialRef();source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
trans=osr.CoordinateTransformation(source,target)
mem=ogr.GetDriverByName('Memory').CreateDataSource('');ml=mem.CreateLayer('lakes',srs=target,geom_type=ogr.wkbUnknown)
for feat in layer:
 g=feat.GetGeometryRef()
 if g is None:continue
 g=g.Clone();g.Transform(trans);e=g.GetEnvelope()
 if e[1]<bounds[0] or e[0]>bounds[2] or e[3]<bounds[1] or e[2]>bounds[3]:continue
 f=ogr.Feature(ml.GetLayerDefn());f.SetGeometry(g);ml.CreateFeature(f)
gdal.RasterizeLayer(lab,[1],ml,burn_values=[1]);lab.FlushCache()
m=gdal.Open(str(mp)); pos=np.zeros(65536,np.int64);neg=pos.copy(); eligible=0;validn=0
for y in range(0,h,512):
 n=min(512,h-y); a=m.ReadAsArray(0,y,w,n);sv=s.GetRasterBand(1).ReadAsArray(0,y,w,n); truth=lab.ReadAsArray(0,y,w,n)>0
 eligible+=int(np.count_nonzero(np.isfinite(sv)&(sv!=0)))
 valid=(a!=-32768)&np.isfinite(sv)&(sv!=0);validn+=int(valid.sum());q=a.astype(np.int32)+32768
 pos+=np.bincount(q[valid&truth],minlength=65536);neg+=np.bincount(q[valid&~truth],minlength=65536)
# Strict comparison water = MNDWI > threshold; all quantized thresholds searched exactly.
tp=pos.sum()-np.cumsum(pos);fp=neg.sum()-np.cumsum(neg);fn=pos.sum()-tp
precision=np.divide(tp,tp+fp,out=np.zeros(65536),where=(tp+fp)>0);recall=tp/max(1,pos.sum())
f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(65536),where=(2*tp+fp+fn)>0)
ious=np.divide(tp,tp+fp+fn,out=np.zeros(65536),where=(tp+fp+fn)>0)
ix=np.arange(22768,42769);best=int(ix[np.argmax(f1[ix])]); good=ix[f1[ix]>=.95*f1[best]];balanced=int(good[np.argmin(abs(precision[good]-recall[good]))])
def row(i):return {'threshold':(i-32768)/10000,'precision':float(precision[i]),'recall':float(recall[i]),'F1':float(f1[i]),'IoU':float(ious[i]),'TP':int(tp[i]),'FP':int(fp[i]),'FN':int(fn[i])}
with (OUT/'阈值搜索.csv').open('w',newline='',encoding='utf-8-sig') as f:
 writer=csv.DictWriter(f,fieldnames=list(row(best)));writer.writeheader();writer.writerows(row(int(i)) for i in ix)
result={'scope':'Kekexili development region, not independent test; nearest aligned to native SAR grid, pixel micro-average; no boundary tolerance','missing_tiles':missing,'valid_pixels':validn,'sar_eligible_pixels':eligible,'valid_coverage':validn/max(1,eligible),'best_F1':row(best),'closest_precision_recall_with_F1_at_least_95percent_of_best':row(balanced),'threshold_zero':row(32768),'reference':os.environ['SAREDGE_REFERENCE']}
(OUT/'阈值结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
