import os
from pathlib import Path
import json,csv,time
import numpy as np
import cv2
from scipy.spatial import cKDTree
from osgeo import gdal,ogr,osr

gdal.UseExceptions();gdal.SetCacheMax(128*1024*1024)
ROOT=Path(os.environ['SAREDGE_ARCHIVE_ROOT'])
OUT=ROOT/'湖泊论文/Water投稿/外审返修_20261022/可可西里_后处理前后指标_20261027';OUT.mkdir(exist_ok=True)
CFG=ROOT/'Dog召回率前置代码/定稿流程/配置与配准记录.json'
cfg=json.loads(CFG.read_text(encoding='utf-8'))
meta=json.loads((ROOT/'Dog召回率前置代码/最终结果筛选step3轮廓/筛选规则与数量.json').read_text(encoding='utf-8'))
x0,y0,w,h=meta['native_window']
raw=gdal.Open(cfg['input_sar']);roi=gdal.Open(cfg['input_roi']);sgt=raw.GetGeoTransform()
gt=(sgt[0]+x0*sgt[1],sgt[1],0,sgt[3]+y0*sgt[5],0,sgt[5])
def srs(wkt):
 s=osr.SpatialReference();s.ImportFromWkt(wkt);s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER);return s
native=srs(raw.GetProjection());metric=srs(cfg['metric_wkt'])
to_metric=osr.CoordinateTransformation(native,metric);to_native=osr.CoordinateTransformation(metric,native)
matrix=np.array(cfg['registration']['matrix_reference_to_sar_m']);inv=np.linalg.inv(np.vstack([matrix,[0,0,1]]))
print('Preparing fixed Kokoxili ROI',w,h,flush=True)
vds=gdal.GetDriverByName('MEM').Create('',roi.RasterXSize,roi.RasterYSize,1,gdal.GDT_Byte)
vds.SetGeoTransform(roi.GetGeoTransform());vds.SetProjection(roi.GetProjection());vds.GetRasterBand(1).WriteArray(roi.GetRasterBand(1).GetMaskBand().ReadAsArray())
valid=gdal.Warp('',vds,format='MEM',dstSRS=raw.GetProjection(),outputBounds=[gt[0],gt[3]+h*gt[5],gt[0]+w*gt[1],gt[3]],width=w,height=h,resampleAlg='near',dstNodata=0).ReadAsArray()>0
valid &= raw.GetRasterBand(1).ReadAsArray(x0,y0,w,h)>0
vds=None
stages={'Initial lake mask':'SeedGrow.tif','After boundary refinement':'Final.tif','After internal-gap filling':'Polish.tif'}
base=ROOT/'湖泊论文/LakeDetection/AfterStep7'
def load_mask(path):
 ds=gdal.Open(str(path));assert (ds.RasterXSize,ds.RasterYSize)==(raw.RasterXSize,raw.RasterYSize)
 mask=np.zeros((h,w),bool)
 for y in range(0,h,256):
  nh=min(256,h-y);rgb=ds.ReadAsArray(x0,y0+y,w,nh,band_list=[1,2,3])
  # Match original Step8, including partially blended red overlay pixels.
  hsv=cv2.cvtColor(rgb.transpose(1,2,0),cv2.COLOR_RGB2HSV)
  mask[y:y+nh]=((hsv[:,:,0]<=10)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>=50)&(hsv[:,:,2]>=50)
 return mask&valid
masks={label:load_mask(base/name) for label,name in stages.items()}
print('Stage masks loaded',[(k,int(v.sum())) for k,v in masks.items()],flush=True)
# Verify archived intermediate relationships where their component files exist.
support=load_mask(base/'final_red_overlay_result_optimized.tif');dilite=load_mask(base/'dilite.tif');burr=load_mask(base/'BurrDelect.tif')
initial,refined,filled=masks.values()
audit={'dilite_vs_initial_minus_support_different_pixels':int(np.count_nonzero(dilite^(initial&~support))),
       'final_vs_dilite_union_burr_different_pixels':int(np.count_nonzero(refined^(dilite|burr))),
       'polish_removes_final_pixels':int(np.count_nonzero(refined&~filled))}
del support,dilite,burr
print('Archive chain audit',audit,flush=True)
def affine_geom(g):
 if g.GetPointCount():
  for i in range(g.GetPointCount()):
   x,y,*_=g.GetPoint(i);v=matrix@np.array([x,y,1]);g.SetPoint_2D(i,float(v[0]),float(v[1]))
 else:
  for i in range(g.GetGeometryCount()):affine_geom(g.GetGeometryRef(i))
def reference(registered):
 src=ogr.Open(cfg['input_shp']);layer=src.GetLayer();ss=layer.GetSpatialRef().Clone();ss.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
 trans=osr.CoordinateTransformation(ss,metric)
 store=ogr.GetDriverByName('Memory').CreateDataSource('');out=store.CreateLayer('lakes',srs=native,geom_type=ogr.wkbUnknown)
 # Only clip by extent; include every lake size in the common valid ROI.
 bx0,bx1=gt[0],gt[0]+w*gt[1];by0,by1=gt[3]+h*gt[5],gt[3]
 count=0
 for f in layer:
  g=f.GetGeometryRef().Clone();g.Transform(trans)
  if registered:affine_geom(g)
  g.Transform(to_native);a,b,c,d=g.GetEnvelope()
  if b<bx0 or a>bx1 or d<by0 or c>by1:continue
  nf=ogr.Feature(out.GetLayerDefn());nf.SetGeometry(g);out.CreateFeature(nf);count+=1
 ds=gdal.GetDriverByName('MEM').Create('',w,h,1,gdal.GDT_Byte);ds.SetGeoTransform(gt);ds.SetProjection(raw.GetProjection())
 gdal.RasterizeLayer(ds,[1],out,burn_values=[1]);truth=ds.ReadAsArray().astype(bool)&valid
 return truth,count
def scores(mask,truth):
 tp=int(np.count_nonzero(mask&truth));fp=int(mask.sum())-tp;fn=int(truth.sum())-tp;tn=int(valid.sum())-tp-fp-fn
 div=lambda a,b:a/b if b else 0.
 p=div(tp,tp+fp);r=div(tp,tp+fn);f=div(2*tp,2*tp+fp+fn);iou=div(tp,tp+fp+fn);bg=div(tn,tn+fp+fn)
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,precision=p,recall=r,f1=f,lake_iou=iou,background_iou=bg,miou=(iou+bg)/2)
interior_valid=cv2.erode(valid.astype('uint8'),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
def boundary_points(mask):
 edge=mask&~cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)&interior_valid
 rr,cc=np.where(edge);coords=np.empty((len(rr),2),dtype=np.float64)
 for start in range(0,len(rr),100000):
  sl=slice(start,min(start+100000,len(rr)));xy=np.column_stack([gt[0]+(cc[sl]+.5)*gt[1],gt[3]+(rr[sl]+.5)*gt[5]])
  transformed=np.array(to_metric.TransformPoints(xy.tolist()))[:,:2]
  coords[sl]=np.column_stack([transformed,np.ones(len(transformed))])@inv[:2,:].T
 return coords
allrows=[];transitions=[]
for registered in [False,True]:
 mode='Fixed existing 43-point registration' if registered else 'Original georeferencing only'
 truth,count=reference(registered);print('Reference',mode,'pixels',int(truth.sum()),'bbox-intersecting objects',count,flush=True)
 tree=None;q=None
 if registered:
  q=boundary_points(truth);tree=cKDTree(q);print('Reference boundary points',len(q),flush=True)
 for label,mask in masks.items():
  row=dict(reference_alignment=mode,stage=label,valid_pixels=int(valid.sum()),reference_pixels=int(truth.sum()),**scores(mask,truth))
  if registered:
   points=boundary_points(mask)
   pd=tree.query(points,k=1,workers=2)[0];qd=cKDTree(points).query(q,k=1,workers=2)[0]
   bp=float((pd<=60).mean());br=float((qd<=60).mean());bf=2*bp*br/(bp+br) if bp+br else 0.
   row.update(boundary_precision_60m=bp,boundary_recall_60m=br,boundary_f1_60m=bf,predicted_boundary_pixels=len(points),reference_boundary_pixels=len(q))
   del points,pd,qd
  allrows.append(row);print('RESULT',json.dumps(row),flush=True)
 for (l1,m1),(l2,m2) in zip(list(masks.items())[:-1],list(masks.items())[1:]):
  added=m2&~m1;removed=m1&~m2
  transitions.append(dict(reference_alignment=mode,from_stage=l1,to_stage=l2,added_reference_water=int(np.count_nonzero(added&truth)),added_reference_nonwater=int(np.count_nonzero(added&~truth)),removed_reference_water=int(np.count_nonzero(removed&truth)),removed_reference_nonwater=int(np.count_nonzero(removed&~truth))))
 del truth,q,tree
def savecsv(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
savecsv(OUT/'三个阶段_整湖区指标.csv',allrows);savecsv(OUT/'步骤增删像素_参考水体对应.csv',transitions)
report=dict(scope='Whole valid Kokoxili ROI from fixed existing native window; all reference lake sizes included.',input_roi=cfg['input_roi'],reference_shp=cfg['input_shp'],native_window=[x0,y0,w,h],native_geotransform=gt,stage_paths={k:str(base/v) for k,v in stages.items()},
 baseline_definition='Initial lake mask is AFTER support processing and seed growth; this compares subsequent boundary refinement and gap filling, NOT all postprocessing vs no postprocessing.',
 reference_alignment='Both original georeferencing and frozen pre-existing 43-point reference-to-SAR registration reported; no new fitted alignment or metric optimization.',
 boundary_definition='One-pixel inner 3x3-erosion boundary including island boundaries; omit boundary pixels on valid-ROI edge; physical coordinates in fixed reference metric frame; 60m symmetric matching.',
 registration_caveat='Existing 43-point holdout median residual112.8m exceeds60m tolerance; absolute BF1 subject to alignment uncertainty. Reference date mismatch also applies.',
 archive_chain_audit=audit,controlled_ablation=False,qualification='Archived-stage comparison. If chain checks are nonzero, these are stage-labelled snapshots and not a verified same-run causal ablation. Pixel values unchanged; no dilation, largest-component removal or lake-size filtering.',
 results=allrows,transitions=transitions)
(OUT/'计算口径与来源.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('SAVED',OUT,flush=True)
