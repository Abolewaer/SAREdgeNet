import os
from pathlib import Path
import hashlib
import json
import sys
import cv2
import numpy as np
import pandas as pd
from osgeo import gdal
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from saredge.spatial_protocol import contour_indices


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    cv2.setNumThreads(8)
    images=Path(os.environ['SAREDGE_CONTOURS_DIR'])
    table=images/'contour_statistics_enhanced_with_dem.xlsx'
    source=images/'step0_original.tif'
    mask_path=images/'step11_remove_small_contours_final_mask.tif'
    output=HERE/'data';output.mkdir(exist_ok=True)
    original=pd.read_excel(table)
    if original.shape[1]!=17 or not original.iloc[:,-1].isin([0,1]).all() or original.iloc[:,0].duplicated().any():
        raise ValueError('Expected unique real IDs, 15 numerical features and binary manual labels')
    names=['Edge Pixel Count','Bounding Box Length(px)','Bounding Box Width(px)','L/W + W/L','Curvature Feature','Roughness',
           'Pixel Mean','Pixel Std','Pixel Median','DEM Mean','DEM Std','DEM Median','DEM Min','DEM Max','Valid DEM Pixels']
    frame=pd.DataFrame({'sample_id':original.iloc[:,0].astype(int).astype(str)})
    for i,name in enumerate(names,1):frame[name]=original.iloc[:,i].to_numpy()
    frame['label']=original.iloc[:,-1].astype(int)
    ds=gdal.Open(str(mask_path));mask=ds.ReadAsArray()>0
    count,components,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    del mask
    mapping=[];compact=[];extent=[];records=[]
    print(f'Real table rows={len(frame)}; source components={count-1}',flush=True)
    for index,row in frame.iterrows():
        identifier=int(row.sample_id)
        if identifier>=count:raise ValueError(f'Contour ID absent from source components: {identifier}')
        x,y,width,height,area=map(int,stats[identifier])
        if [width,height]!=[int(row[names[1]]),int(row[names[2]])]:
            raise ValueError(f'Source component bbox does not reproduce real row {identifier}')
        image_path=images/f'{identifier}.png'
        with Image.open(image_path) as im:array=np.array(im.convert('L'))
        if not np.isin(array,[0,255]).all():raise ValueError(f'Not an original binary contour PNG: {identifier}')
        binary=array>0
        px,py,pw,ph=cv2.boundingRect(binary.astype(np.uint8))
        if int(binary.sum())!=int(row[names[0]]):
            raise ValueError(f'PNG/feature/source identity mismatch: {identifier}')
        half=(max(width,height)+10)//2
        left=max(0,x+width//2-half);top=max(0,y+height//2-half);ih,iw=array.shape
        if left<0 or top<0 or left+iw>ds.RasterXSize or top+ih>ds.RasterYSize:
            raise ValueError(f'Invalid recovered crop for {identifier}')
        if not np.array_equal(binary,components[top:top+ih,left:left+iw]==identifier):
            raise ValueError(f'PNG does not reproduce native source component: {identifier}')
        original_component=(components[y:y+height,x:x+width]==identifier).astype(np.uint8)
        contours,_=cv2.findContours(original_component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        c=max(contours,key=cv2.contourArea);a=cv2.contourArea(c);perimeter=cv2.arcLength(c,True)
        compact.append(float(4*np.pi*a/perimeter**2) if perimeter else 0.)
        extent.append(float(a/(width*height)))
        mapping.append(dict(sample_id=row.sample_id,source_raster=str(source),col=left,row=top,width=iw,height=ih,scene_id='Kekexili_original_SAR_scene'))
        records.append(dict(sample_id=row.sample_id,image_path=str(image_path),image_sha256=sha(image_path),native_component_id=identifier,
                            bbox_and_crop_pixel_count_verified=True,pixelwise_native_crop_verified=True,
                            source_component_pixels=area,png_pixels=int(binary.sum()),crop_truncated=area!=int(binary.sum())))
        if (index+1)%500==0:print(f'Verified {index+1}/{len(frame)}',flush=True)
    frame['Compactness']=compact;frame['Extent']=extent
    features=names+['Compactness','Extent']
    frame=frame[['sample_id',*features,'label']]
    frame['image_path']=[v['image_path'] for v in records]
    frame.to_csv(output/'features17_manual_labels.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mapping).to_csv(output/'contour_spatial_mapping.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(output/'image_identity_audit.csv',index=False,encoding='utf-8-sig')
    config=dict(spatial_samples_file=str(output/'contour_spatial_mapping.csv'),spatial_protocol_file=str(ROOT/'configs/SpatialProtocol.json'))
    indices,split,audit=contour_indices(frame,config,id_column='sample_id',label_column='label')
    split.to_csv(output/'shared_split.csv',index=False,encoding='utf-8-sig')
    frame.merge(split[['sample_id','subset']],on='sample_id',validate='one_to_one').to_csv(output/'shared_manifest.csv',index=False,encoding='utf-8-sig')
    summary=dict(source_table=str(table),source_table_sha256=sha(table),source_mask=str(mask_path),source_mask_sha256=sha(mask_path),
        raw_features=15,derived_features=['Compactness=4*pi*A/P^2','Extent=A/(bbox_width*bbox_height)'],
        derivation='Original binary PNGs verified pixelwise against native source crops; extra geometry derived from full numbered component, as in formal Step4; no synthetic samples or predicted labels',
        feature_columns=features,manual_labels=frame.label.value_counts().to_dict(),spatial_audit=audit,
        manifest_sha256=sha(output/'shared_manifest.csv'),split_sha256=sha(output/'shared_split.csv'),
        source_scene_count=1,external_scope='Outside the predefined Kekexili.tif ROI within this same original SAR scene; not an independent-scene test',
        contour_metrics_only=True)
    (output/'data_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
