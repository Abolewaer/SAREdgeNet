"""Prepare aligned TIFF inputs only. No dilation or recall calculation.

Reuses the saved whole-region affine registration, without refitting on recall.
All permanent paths are relative to this script or recorded raw-data inputs.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape
import json,hashlib
import numpy as np
from osgeo import gdal,ogr,osr

gdal.UseExceptions()
BASE=Path(__file__).parent
CFG=json.loads((BASE/'配置与配准记录.json').read_text(encoding='utf8'))
OUT=BASE/'产物';OUT.mkdir(exist_ok=True)

def write(path,array,gt,wkt):
    ds=gdal.GetDriverByName('GTiff').Create(str(path),array.shape[1],array.shape[0],1,gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE','TILED=YES'])
    ds.SetGeoTransform(gt);ds.SetProjection(wkt);ds.GetRasterBand(1).WriteArray(array);ds=None

def crs(wkt):
    s=osr.SpatialReference();s.ImportFromWkt(wkt)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER);return s

def corrected_vrt(source,path,matrix):
    ds=gdal.Open(str(source));gt=ds.GetGeoTransform()
    transform=np.array([[gt[1],gt[2],gt[0]],[gt[4],gt[5],gt[3]],[0,0,1.]])
    corrected=np.linalg.inv(np.vstack((matrix,[0.,0.,1.])))@transform
    cg=[corrected[0,2],corrected[0,0],corrected[0,1],corrected[1,2],corrected[1,0],corrected[1,1]]
    path.write_text(f'''<VRTDataset rasterXSize="{ds.RasterXSize}" rasterYSize="{ds.RasterYSize}">
<SRS>{escape(ds.GetProjection())}</SRS><GeoTransform>{','.join(map(str,cg))}</GeoTransform>
<VRTRasterBand dataType="Byte" band="1"><NoDataValue>0</NoDataValue><SimpleSource>
<SourceFilename relativeToVRT="0">{escape(str(source.resolve()))}</SourceFilename><SourceBand>1</SourceBand>
<SrcRect xOff="0" yOff="0" xSize="{ds.RasterXSize}" ySize="{ds.RasterYSize}"/>
<DstRect xOff="0" yOff="0" xSize="{ds.RasterXSize}" ySize="{ds.RasterYSize}"/>
</SimpleSource></VRTRasterBand></VRTDataset>''',encoding='utf8')

def rings(g):
    if g.GetGeometryName()=='POLYGON':
        for i in range(g.GetGeometryCount()):yield g.GetGeometryRef(i).GetPoints()
    else:
        for i in range(g.GetGeometryCount()):yield from rings(g.GetGeometryRef(i))

def main():
    wkt=CFG['metric_wkt'];mgt=CFG['metric_grid_gt'];mw,mh=CFG['metric_grid_size']
    gt=CFG['evaluation_gt'];w,h=CFG['evaluation_size']
    mb=[mgt[0],mgt[3]+mh*mgt[5],mgt[0]+mw*mgt[1],mgt[3]]
    eb=[gt[0],gt[3]+h*gt[5],gt[0]+w*gt[1],gt[3]]
    matrix=np.array(CFG['registration']['matrix_reference_to_sar_m'])
    with TemporaryDirectory(prefix='dog60_') as tmp:
        tmp=Path(tmp)
        sar_band=tmp/'sar_band.vrt'
        gdal.Translate(str(sar_band),CFG['input_sar'],format='VRT',bandList=[1])
        grids={}
        for label,source,algorithm in [('sar',sar_band,'bilinear'),('candidate',CFG['input_candidate'],'near')]:
            metric_path=tmp/f'{label}_metric.vrt'
            gdal.Warp(str(metric_path),str(source),format='VRT',dstSRS=wkt,outputBounds=mb,
                width=mw,height=mh,resampleAlg=algorithm,srcNodata=0,dstNodata=0)
            aligned=tmp/f'{label}_aligned.vrt';corrected_vrt(metric_path,aligned,matrix)
            grids[label]=gdal.Warp('',str(aligned),format='MEM',dstSRS=wkt,outputBounds=eb,
                width=w,height=h,resampleAlg=algorithm,dstNodata=0).ReadAsArray()
        roi=gdal.Open(CFG['input_roi'])
        native_valid=roi.GetRasterBand(1).GetMaskBand().ReadAsArray()
        footprint=tmp/'roi.tif';write(footprint,native_valid,roi.GetGeoTransform(),roi.GetProjection())
        roi60=gdal.Warp('',str(footprint),format='MEM',dstSRS=wkt,outputBounds=eb,width=w,height=h,
            resampleAlg='near',dstNodata=0).ReadAsArray()>0
    valid=roi60&(grids['sar']>0)
    # Keep the exact candidate sampling used in the preceding step3 evaluation.
    candidate=grids['candidate']
    assert hashlib.sha256(candidate.tobytes()).hexdigest()==CFG['prior_result_validation']['candidate_sha256'], 'Candidate grid differs from prior step3 run'
    candidate=np.where(grids['sar']>0,candidate,0).astype(np.uint8)
    write(OUT/'step3候选_配准后60m_未膨胀.tif',candidate,gt,wkt)
    write(OUT/'湖区有效统计范围_60m.tif',valid.astype(np.uint8)*255,gt,wkt)
    reference=ogr.Open(CFG['input_shp']);layer=reference.GetLayer()
    ss=layer.GetSpatialRef().Clone();ss.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    metric=crs(wkt);ct=osr.CoordinateTransformation(ss,metric)
    mem=ogr.GetDriverByName('Memory').CreateDataSource('')
    lines=mem.CreateLayer('shorelines',srs=metric,geom_type=ogr.wkbLineString)
    lines.CreateField(ogr.FieldDefn('OBJECTID',ogr.OFTInteger))
    layer.SetAttributeFilter(f"Area > {CFG['min_lake_area_km2_exclusive']}")
    for feature in layer:
        geometry=feature.GetGeometryRef().Clone();geometry.Transform(ct)
        a,b,c,d=geometry.GetEnvelope()
        if b<eb[0] or a>eb[2] or d<eb[1] or c>eb[3]:continue
        for points in rings(geometry):
            ring=ogr.Geometry(ogr.wkbLineString)
            for x,y,*_ in points:ring.AddPoint_2D(x,y)
            f=ogr.Feature(lines.GetLayerDefn());f.SetField('OBJECTID',int(feature.GetField('OBJECTID')))
            f.SetGeometry(ring);lines.CreateFeature(f)
    dst=gdal.GetDriverByName('MEM').Create('',w,h,1,gdal.GDT_UInt32)
    dst.SetGeoTransform(gt);dst.SetProjection(wkt)
    gdal.RasterizeLayer(dst,[1],lines,options=['ATTRIBUTE=OBJECTID','ALL_TOUCHED=FALSE'])
    ids=dst.ReadAsArray();boundary=(ids>0)&valid
    binary=boundary.astype(np.uint8)*255
    assert hashlib.sha256(binary.tobytes()).hexdigest()==CFG['prior_result_validation']['reference_sha256'], 'Reference boundary differs from prior run'
    write(OUT/'Shp大于1km2湖泊边界_60m.tif',binary,gt,wkt)
    audit=dict(lake_count=len(np.unique(ids[boundary])),reference_pixels=int(boundary.sum()),
        lake_OBJECTID=np.unique(ids[boundary]).tolist(),valid_region_pixels=int(valid.sum()),
        valid_region_area_km2=int(valid.sum())*60**2/1e6,dilation=False,
        candidate_resampling='nearest neighbour',registration_refitted=False)
    (BASE/'输入准备记录.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
