// SAREdgeNet: MNDWI baseline, 2022 June-August.
// ROI is the exact outer extent of the SAR TIFF, also supplied as SHP.
// It includes SAR NoData areas: exclude them during local evaluation.
// Inline identical geometry avoids uploading the SHP to use this script.
var geometry = ee.Geometry.Polygon([[[78.17067737185428, 29.520302119443265], [101.22014500519761, 29.520302119443265], [101.22014500519761, 38.65023967806297], [78.17067737185428, 38.65023967806297], [78.17067737185428, 29.520302119443265]]], 'EPSG:4326', false);
var config = {start:'2022-06-01', end:'2022-09-01', cloud:40,
  crs:'EPSG:32646', grid:[20,0,0,0,-20,0], threshold:0.1};
// Same dates, SCL cloud mask, median composite and export grid as NDVI_Step1.js.
// B3 10 m -> B11 native 20 m by area averaging BEFORE compositing/index.
function prepare(image) {
  var scl=image.select('SCL');
  var clear=scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  var swir=image.select('B11').updateMask(clear).multiply(0.0001);
  var green=image.select('B3').updateMask(clear).multiply(0.0001)
    .reduceResolution({reducer:ee.Reducer.mean(),maxPixels:64})
    .reproject({crs:image.select('B11').projection()});
  return green.rename('Green').addBands(swir.rename('SWIR1'))
    .copyProperties(image,['system:time_start']);
}
var collection=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(geometry).filterDate(config.start,config.end)
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',config.cloud)).map(prepare);
var composite=collection.median().clip(geometry);
var green=composite.select('Green'), swir=composite.select('SWIR1');
var denominator=green.add(swir);
var mndwi=green.subtract(swir).divide(denominator)
  .updateMask(denominator.neq(0)).rename('MNDWI').toFloat();
var water=mndwi.gt(config.threshold).rename('water').toUint8();
var count=collection.select('Green').count().clip(geometry).rename('clear_count');
print('Number of source granules',collection.size());
print('Period: June-August 2022; water if MNDWI > 0.1; 20 m output.');
print('SAR TIFF outer extent; not the spatial test mask.');
Map.centerObject(geometry,5);
Map.addLayer(geometry,{color:'orange'},'SAR TIFF export extent',false);
Map.addLayer(mndwi,{min:-1,max:1,palette:['8c510a','f5f5f5','01665e']},'MNDWI');
Map.addLayer(water.selfMask(),{palette:['0088ff']},'MNDWI water');
Map.addLayer(count,{min:0,max:20,palette:['red','yellow','green']},'Clear observations',false);
// Single-band continuous Float32 only; thresholding remains local.
// One batch export, split into manageable GeoTIFF files. Not a download URL.
// Large plateau computation can take time. Do not treat missing pixels as land.
Export.image.toDrive({image:mndwi
  .unmask(-9999), description:'MNDWI_2022_SAR_extent_20m_Float32_only',folder:'SAREdgeNet_MNDWI_Float32_only',
  region:geometry,crs:config.crs,crsTransform:config.grid,maxPixels:1e13,
  shardSize:256,fileDimensions:8192,skipEmptyTiles:true,fileFormat:'GeoTIFF',
  formatOptions:{cloudOptimized:true,noData:-9999}});
