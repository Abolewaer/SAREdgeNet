var settings = {
  start: '2022-06-01',
  end: '2022-09-01',
  threshold: 0.0,
  maxCloudPercent: 40,
  crs: 'EPSG:32646',
  gridTransform: [20, 0, 0, 0, -20, 0],
  folder: 'SAREdgeNet_NDVI'
};

function maskClouds(image) {
  var scl = image.select('SCL');
  var clear = scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return image.updateMask(clear).select(['B4', 'B8']).multiply(0.0001)
    .copyProperties(image, ['system:time_start']);
}

function buildNDVI(region, config) {
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region).filterDate(config.start, config.end)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', config.maxCloudPercent))
    .map(maskClouds);
  var composite = collection.median().clip(region);
  var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var water = ndvi.lt(config.threshold).rename('water').toUint8();
  return {collection: collection, ndvi: ndvi, water: water};
}

if (typeof geometry === 'undefined') {
  throw new Error('Draw/import an ROI named geometry before running. Set CRS and gridTransform to the SAR evaluation grid.');
}

var result = buildNDVI(geometry, settings);
print('Image count', result.collection.size());
Map.centerObject(geometry);
Map.addLayer(result.ndvi, {min: -1, max: 1, palette: ['blue', 'white', 'green']}, 'NDVI');
Map.addLayer(result.water.selfMask(), {palette: ['red']}, 'NDVI water');

Export.image.toDrive({
  image: result.ndvi.unmask(-9999),
  description: 'NDVI_2022',
  folder: settings.folder,
  region: geometry,
  crs: settings.crs,
  crsTransform: settings.gridTransform,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {noData: -9999}
});

Export.image.toDrive({
  image: result.water.unmask(255),
  description: 'NDVI_water_2022',
  folder: settings.folder,
  region: geometry,
  crs: settings.crs,
  crsTransform: settings.gridTransform,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {noData: 255}
});
