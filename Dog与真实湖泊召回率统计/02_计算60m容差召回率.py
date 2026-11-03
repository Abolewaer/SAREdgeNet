"""Calculate reference-boundary recall from prepared TIFFs; no registration or dilation."""
from pathlib import Path
import csv,json
import numpy as np
from osgeo import gdal
from scipy.ndimage import distance_transform_edt

gdal.UseExceptions()
BASE=Path(__file__).parent
CFG=json.loads((BASE/'配置与配准记录.json').read_text(encoding='utf8'))
OUT=BASE/'产物'
def read(name):
    ds=gdal.Open(str(OUT/name))
    return ds.ReadAsArray(),ds.GetGeoTransform(),ds.GetProjection()

def main():
    candidate,gt,wkt=read('step3候选_配准后60m_未膨胀.tif')
    ref,rgt,rwkt=read('Shp大于1km2湖泊边界_60m.tif')
    valid,vgt,vwkt=read('湖区有效统计范围_60m.tif')
    assert gt==rgt==vgt and wkt==rwkt==vwkt and candidate.shape==ref.shape==valid.shape
    assert gt[2]==gt[4]==0 and abs(gt[1])==abs(gt[5])==60
    # Retain candidates in the surrounding halo for reference pixels near ROI edges.
    # The candidate TIFF is already invalid outside the SAR footprint. ROI is used
    # to select reference pixels, not to remove neighbouring candidate foreground.
    foreground=candidate>0
    boundary=(ref>0)&(valid>0)
    distance=distance_transform_edt(~foreground,sampling=(60,60))
    matched=boundary&(distance<=CFG['match_tolerance_m'])
    n=int(boundary.sum());tp=int(matched.sum());fn=n-tp
    assert n==CFG['prior_result_validation']['reference_pixels']
    assert tp==CFG['prior_result_validation']['matched_pixels_at60m']
    preparation=json.loads((BASE/'输入准备记录.json').read_text(encoding='utf8'))
    row=dict(grid_resolution_m=60,match_tolerance_m=CFG['match_tolerance_m'],
        min_lake_area_km2_exclusive=1,lake_count=preparation['lake_count'],dilation=False,
        reference_boundary_pixels=n,matched_reference_pixels=tp,unmatched_reference_pixels=fn,
        pixel_recall=tp/n)
    with (BASE/'召回率结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    (BASE/'召回率结果.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
    status=np.zeros(boundary.shape,np.uint8);status[matched]=1;status[boundary&~matched]=2
    ds=gdal.GetDriverByName('GTiff').Create(str(OUT/'参考边界匹配结果_1命中_2未命中.tif'),status.shape[1],status.shape[0],1,gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE','TILED=YES'])
    ds.SetGeoTransform(gt);ds.SetProjection(wkt);ds.GetRasterBand(1).WriteArray(status);ds=None
    print(json.dumps(row,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
