# DEM completeness audit

`audit_dem_missingness.py` counts NoData, validity-mask exclusions, NaN/Inf and zeros separately in the archived inputs. Set `SAREDGE_ARCHIVE_ROOT` to the recovered external project tree before executing. This is a potentially long read of real rasters, not a training task; it was not run during packaging.

The [recorded audit](../results/dem_missingness/audit.json) covers 18 processed DEM tiles and their contour tables. Within the audit's SAR-eligible sample footprint, processed zero/NoData observations were about 0.543%. A normalized zero may be a valid minimum or clipped value, so this is not a direct estimate of original DEM acquisition gaps. The 405,320 checked contour records had finite DEM descriptors. Overlapping raster observations are not geographically deduplicated.

The main image branch receives contour masks, not DEM rasters. DEM invalid samples are excluded before descriptors enter the numerical branch. The audit does not imply that convolutions accept NaN values.
