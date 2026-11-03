// Exactly ONE export task. Run the next file only after this task completes.
// Tile 16/24; same original study extent, dates, 20m grid and algorithm.
// Int16 stored values: MNDWI*10000 rounded; NoData=-32768.
var tile=ee.Geometry.Rectangle([176320, 3545180, 550220, 3824720],'EPSG:32646',false);
// SAREdgeNet: MNDWI baseline, 2022 June-August.
// ROI is the exact outer extent of the SAR TIFF, also supplied as SHP.
// It includes SAR NoData areas: exclude them during local evaluation.
// Inline identical geometry avoids uploading the SHP to use this script.
var geometry = ee.Geometry.Polygon([[[78.17067737185428, 29.520302119443265], [101.22014500519761, 29.520302119443265], [101.22014500519761, 38.65023967806297], [78.17067737185428, 38.65023967806297], [78.17067737185428, 29.520302119443265]]], 'EPSG:4326', false);
var config = {start:'2022-06-01', end:'2022-09-01', cloud:40,
  crs:'EPSG:32646', grid:[20,0,0,0,-20,0], threshold:0};
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
  .filterBounds(geometry).filterBounds(tile.buffer(200,1)).filterDate(config.start,config.end)
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',config.cloud)).map(prepare);
var composite=collection.median().clip(geometry);
var green=composite.select('Green'), swir=composite.select('SWIR1');
var denominator=green.add(swir);
var mndwi=green.subtract(swir).divide(denominator)
  .updateMask(denominator.neq(0)).rename('MNDWI').toFloat();
print('Tile', '16/24');
print('Source granules for this tile',collection.size());
print('Estimated raw MiB (before compression/COG overviews)',498.39022636413574);
print('Decode: exclude -32768 first, then divide by 10000.');
Map.centerObject(tile,7);
Map.addLayer(tile,{color:'orange'},'Export tile');
Export.image.toDrive({image:mndwi.multiply(10000).round().toInt16().unmask(-32768),
 description:'MNDWI_2022_Int16_tile_16_of_24',fileNamePrefix:'MNDWI_2022_Int16_tile_16_of_24',folder:'SAREdgeNet_MNDWI_Int16_24tiles',
 region:tile,crs:config.crs,crsTransform:config.grid,maxPixels:1e13,
 shardSize:256,fileDimensions:32768,skipEmptyTiles:true,fileFormat:'GeoTIFF',
 formatOptions:{cloudOptimized:true,noData:-32768}});
