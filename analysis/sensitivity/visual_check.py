import os
from pathlib import Path
import time,json,csv
import numpy as np
import cv2
from scipy import ndimage as ndi
from osgeo import gdal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

for name in ['times.ttf','timesbd.ttf']:
 p=Path('C:/Windows/Fonts')/name
 if p.exists():font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','pdf.fonttype':42})
root=Path(os.environ['SAREDGE_ARCHIVE_ROOT'])
out=root/'湖泊论文/Water投稿/外审返修_20261022/参数目视对照_五档_20261027';out.mkdir(exist_ok=True)
rawpath=root/'WorkDir/Finally/U-Net/tif/Tibet1.tif';ds=gdal.Open(str(rawpath));gt=ds.GetGeoTransform()
roi=gdal.Open(str(root/'KeKexili/Kekexili.tif'));rg=roi.GetGeoTransform()
def window(bounds):
 xmin,ymin,xmax,ymax=bounds
 l=max(0,int(np.floor((xmin-gt[0])/gt[1])));r=min(ds.RasterXSize,int(np.ceil((xmax-gt[0])/gt[1])))
 t=max(0,int(np.floor((ymax-gt[3])/gt[5])));b=min(ds.RasterYSize,int(np.ceil((ymin-gt[3])/gt[5])))
 return l,t,r-l,b-t
stats_path=out/'固定区域灰度统计.json'
if stats_path.exists():stats=json.loads(stats_path.read_text(encoding='utf-8'))
else:
 bx=(rg[0],rg[3]+roi.RasterYSize*rg[5],rg[0]+roi.RasterXSize*rg[1],rg[3]);l,t,w,h=window(bx)
 n=0;s=0.;ss=0.;lo=float('inf');hi=-float('inf')
 for off in range(0,h,256):
  v=ds.GetRasterBand(1).ReadAsArray(l,t+off,w,min(256,h-off));v=v[np.isfinite(v)&(v!=0)].astype('float64')
  n+=v.size;s+=float(v.sum());ss+=float(np.square(v).sum());lo=min(lo,float(v.min()));hi=max(hi,float(v.max()))
 mean=s/n;std=np.sqrt(max(0,ss/n-mean**2));stats=dict(source=str(rawpath),region='Fixed rectangular source window covering Kokoxili.tif extent, not the entire plateau mosaic',source_window=[l,t,w,h],nonzero_count=n,mean=mean,std=float(std),minimum=lo,maximum=hi,exclude='Nonfinite and exactly zero, following preprocessing implementation')
 stats_path.write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
print('STATS',stats,flush=True)
# Use a lake within the Kokoxili ROI, not the earlier figure outside that ROI.
# This is the wide central lake in the native archived scene.
scene=gdal.Open(str(root/'湖泊论文/LakeDetection/Contours/step0_original.tif'));sg=scene.GetGeoTransform()
display_native=(6736,4984,3016,1064)
xx,yy,ww,hh=display_native
bounds=(sg[0]+xx*sg[1],sg[3]+(yy+hh)*sg[5],sg[0]+(xx+ww)*sg[1],sg[3]+yy*sg[5])
l,t,w,h=window(bounds);pad=96
L,T=max(0,l-pad),max(0,t-pad);W,H=w+2*pad,h+2*pad
image=ds.GetRasterBand(1).ReadAsArray(L,T,W,H).astype(np.float32);valid=np.isfinite(image)&(image!=0)
sy,sx=t-T,l-L
view=(slice(sy,sy+h),slice(sx,sx+w))
def preprocess(k):
 cap=stats['mean']+k*stats['std'];upper=min(cap,stats['maximum']);low=stats['minimum']
 a=np.zeros(image.shape,np.uint8)
 a[valid]=np.clip(255*(np.minimum(image[valid],cap)-low)/(upper-low),0,255).astype('uint8')
 a=ndi.gaussian_filter(a,1.).astype('uint8');a=ndi.median_filter(a,size=3).astype('uint8');a=ndi.grey_erosion(a,size=(3,3));a=ndi.grey_dilation(a,size=(3,3));a[~valid]=0
 return a
def dog(a,s1,s2):
 z=np.abs(ndi.gaussian_filter(a.astype('float32'),s1)-ndi.gaussian_filter(a.astype('float32'),s2));maximum=float(z.max())
 return (z/maximum*255).astype('uint8') if maximum>0 else z.astype('uint8')
def filt(a,n):
 nlab,labels,st,_=cv2.connectedComponentsWithStats(a,connectivity=8)
 keep=np.zeros(nlab,bool)
 if nlab>1:keep[1:]=st[1:,cv2.CC_STAT_AREA]>=n
 return keep[labels].astype('uint8')*255
def candidates(edge,threshold,area_scale):
 a=(edge>threshold).astype('uint8')*255
 a=filt(a,round(200*area_scale));a=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((4,4),np.uint8))
 a=filt(a,round(200*area_scale));a=cv2.erode(a,np.ones((2,2),np.uint8));a=filt(a,round(500*area_scale))
 # Stop at the same pre-thinning processing stage for every setting.
 return a
base_pre=preprocess(2);base_edge=dog(base_pre,2,6)
groups=[('Clipping coefficient k',[1,1.5,2,2.5,3],'clip'),('DoG small scale sigma1',[1,1.5,2,2.5,3],'s1'),('DoG large scale sigma2',[4,5,6,7,8],'s2'),('DoG response threshold',[30,40,50,60,70],'threshold'),('Area-filter joint multiplier',[.25,.5,1,1.5,2],'area')]
blue=np.array([.15,.56,.77]);records=[];group_results=[]
for title,values,key in groups:
 print('GROUP',title,flush=True);panels=[]
 for val in values:
  pre=preprocess(val) if key=='clip' else base_pre
  edge=dog(pre,val if key=='s1' else 2,val if key=='s2' else 6) if key in ['clip','s1','s2'] else base_edge
  mask=candidates(edge,val if key=='threshold' else 50,val if key=='area' else 1)
  # Same fixed baseline SAR background to make edge changes comparable.
  bg=base_pre[view].astype('float32')/255;rgb=np.repeat(bg[:,:,None],3,axis=2);m=mask[view]>0;rgb[m]=blue
  panels.append(rgb)
  row=dict(parameter=key,value=val,current_setting=val==values[2],foreground_pixels=int(m.sum()),window_width=w,window_height=h,
           clip_k=val if key=='clip' else 2,sigma1=val if key=='s1' else 2,sigma2=val if key=='s2' else 6,response_threshold=val if key=='threshold' else 50,area_limits=[round(v*(val if key=='area' else 1)) for v in [200,200,500]])
  records.append(row)
  np.savez_compressed(out/f'{key}_{val:g}_pre_thinning_mask.npz',mask=m)
 group_results.append((title,values,key,panels))
fig,axes=plt.subplots(5,5,figsize=(17,8.5),layout='constrained')
for row,(title,values,key,panels) in enumerate(group_results):
 for j,(val,rgb) in enumerate(zip(values,panels)):
  ax=axes[row,j];ax.imshow(rgb,interpolation='nearest',aspect='equal');ax.set_xticks([]);ax.set_yticks([])
  ax.set_title(f'{val:g}'+('  [current]' if j==2 else ''),fontsize=12,color='#1d4b6a' if j==2 else '#222',weight='bold' if j==2 else 'normal')
  for spine in ax.spines.values():spine.set_color('#1f658b' if j==2 else '#c6cdd2');spine.set_linewidth(1.7 if j==2 else .5)
  if j==0:ax.set_ylabel(title,fontsize=10,labelpad=8)
fig.savefig(out/'五类参数_各五档_候选边缘目视对照.png',dpi=200);fig.savefig(out/'五类参数_各五档_候选边缘目视对照.pdf');plt.close(fig)
for title,values,key,panels in group_results:
 fig,axes=plt.subplots(1,5,figsize=(16,2.4),layout='constrained')
 for j,(val,rgb) in enumerate(zip(values,panels)):
  axes[j].imshow(rgb,interpolation='nearest',aspect='equal');axes[j].axis('off');axes[j].set_title(f'{val:g}'+('  [current]' if j==2 else ''),fontsize=14,color='#1f658b' if j==2 else '#222')
 fig.suptitle(title,fontsize=15);fig.savefig(out/(key+'_五档对照.png'),dpi=220);plt.close(fig)
with (out/'参数与局部前景像素记录.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
note=dict(source_statistics=stats,input_source_window=[L,T,W,H],display_source_window=[l,t,w,h],parameter_design='Five settings per family, current setting at center. Other conditions fixed; area-filter experiment scales all three cutoffs together and is not an independent test of each cutoff.',implementation='Matches preprocessing operations and DoG/three area filters in saredge.spatial.preprocess and SAREdgeNet_Step4. Stops BEFORE thinning/pruning/classification/postprocessing. Gaussian and component operations run on fixed padded local window; components crossing local bounds may differ from whole-scene processing.',normalization='Raw [nonzero] source values normalized with statistics of fixed Kokoxili bounding rectangle. DoG max normalization is LOCAL padded-window normalization, not whole-scene max; it is fixed spatial scope across settings. Not a reconstruction of original historical parameter selection.',display='Fixed baseline grayscale background; blue shows retained pre-thinning candidate pixels. No lake reference labels, model training, or final segmentation metrics used.',qualification='New local qualitative parameter diagnostic. Does not establish whole-region robustness or globally optimal parameters; does not prove historical parameter experiments existed.',records=records)
(out/'实验范围与限制.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')
print('SAVED',out,flush=True)
