import os
from pathlib import Path
import json,csv,time
import numpy as np
from osgeo import gdal,ogr,osr
gdal.UseExceptions();gdal.SetCacheMax(512*1024*1024)
P=Path(os.environ['SAREDGE_MNDWI_WORK'])/'阈值分析';O=P/'全区域_阈值0.1';O.mkdir(exist_ok=True)
m=gdal.Open(str(P/'MNDWI_2022_现有23片_缺24.tif'));gt=m.GetGeoTransform();w=m.RasterXSize;h=m.RasterYSize
bounds=[gt[0],gt[3]+h*gt[5],gt[0]+w*gt[1],gt[3]]
kw=dict(format='VRT',dstSRS=m.GetProjection(),outputBounds=bounds,width=w,height=h,resampleAlg='near',dstNodata=0)
s=gdal.Warp(str(O/'SAR_valid.vrt'),os.environ['SAREDGE_SAR'],**kw)
k=gdal.Warp(str(O/'Kekexili.vrt'),os.environ['SAREDGE_ROI'],**kw)
ref=O/'参考湖面_20m.tif'
if not ref.exists():
 print('Rasterizing full reference',flush=True)
 dst=gdal.GetDriverByName('GTiff').Create(str(ref),w,h,1,gdal.GDT_Byte,options=['TILED=YES','COMPRESS=LZW','BIGTIFF=YES','SPARSE_OK=YES']);dst.SetGeoTransform(gt);dst.SetProjection(m.GetProjection());dst.GetRasterBand(1).Fill(0)
 vec=ogr.Open(os.environ['SAREDGE_REFERENCE']);ly=vec.GetLayer();source=ly.GetSpatialRef();source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
 target=osr.SpatialReference();target.ImportFromWkt(m.GetProjection());target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER);tx=osr.CoordinateTransformation(source,target)
 mem=ogr.GetDriverByName('Memory').CreateDataSource('');out=mem.CreateLayer('lakes',srs=target,geom_type=ogr.wkbUnknown)
 for f in ly:
  geo=f.GetGeometryRef()
  if geo is None:continue
  geo=geo.Clone();geo.Transform(tx);n=ogr.Feature(out.GetLayerDefn());n.SetGeometry(geo);out.CreateFeature(n)
 gdal.RasterizeLayer(dst,[1],out,burn_values=[1]);dst=None
r=gdal.Open(str(ref));counts={x:np.zeros(4,np.int64) for x in ['all','external_test','development']};eligible=0;validn=0;reference_missing=0
mask=gdal.GetDriverByName('GTiff').Create(str(O/'MNDWI_全区域_水体_0.1.tif'),w,h,1,gdal.GDT_Byte,options=['TILED=YES','COMPRESS=LZW','BIGTIFF=YES','SPARSE_OK=YES']);mask.SetGeoTransform(gt);mask.SetProjection(m.GetProjection());mask.GetRasterBand(1).SetNoDataValue(255)
start=time.time()
for y in range(0,h,1024):
 hh=min(1024,h-y)
 for x in range(0,w,2048):
  ww=min(2048,w-x);a=m.ReadAsArray(x,y,ww,hh);sv=s.GetRasterBand(1).ReadAsArray(x,y,ww,hh);rr=r.ReadAsArray(x,y,ww,hh)>0
  sarok=np.isfinite(sv)&(sv!=0);ok=sarok&(a!=-32768);inside=k.GetRasterBand(1).ReadAsArray(x,y,ww,hh)!=0;pred=a>1000
  eligible+=int(sarok.sum());validn+=int(ok.sum());reference_missing+=int((sarok&~ok&rr).sum())
  for name,vm in [('all',ok),('external_test',ok&~inside),('development',ok&inside)]:
   counts[name]+=np.array([np.count_nonzero(vm&pred&rr),np.count_nonzero(vm&pred&~rr),np.count_nonzero(vm&~pred&rr),np.count_nonzero(vm&~pred&~rr)])
  mask.GetRasterBand(1).WriteArray(np.where(ok,pred.astype('uint8'),255),x,y)
 print(f'rows {y+hh}/{h}; {time.time()-start:.0f}s',flush=True)
mask=None
assert np.array_equal(counts['all'],counts['external_test']+counts['development'])
def metrics(c):
 tp,fp,fn,tn=map(int,c);div=lambda a,b:a/b if b else 0;lake=div(tp,tp+fp+fn);bg=div(tn,tn+fp+fn)
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,precision=div(tp,tp+fp),recall=div(tp,tp+fn),F1=div(2*tp,2*tp+fp+fn),lake_IoU=lake,background_IoU=bg,mIoU=(lake+bg)/2,accuracy=div(tp+tn,sum(c)))
result=dict(threshold=.1,rule='Int16 > 1000',grid='EPSG32646 20m; nearest resampling SAR and Kekexili footprint; reference rasterized by pixel centres; no tolerance',missing_tiles=[24],sar_valid_pixels=eligible,common_valid_pixels=validn,coverage=validn/eligible,reference_lake_pixels_in_missing_observations=reference_missing,metrics={name:metrics(c) for name,c in counts.items()})
(O/'全区域指标.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (O/'全区域指标.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wr=csv.DictWriter(f,fieldnames=['region']+list(metrics(counts['all'])));wr.writeheader();wr.writerows(dict(region=n,**metrics(c)) for n,c in counts.items())
print(json.dumps(result,ensure_ascii=False),flush=True)
