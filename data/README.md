# Packaged manual contour data

`contours/` contains 5,636 original masks from the saved manual-label manifest. `training/features17_labelled.csv` is a positional-format copy of the same values: ID, the 17 descriptors in their original order, then label. It removes only the image-path column for compatibility with the main Step 5 reader; no numeric values or labels are changed.

`training/contour_spatial_mapping.csv` retains sample IDs and crop coordinates but uses the portable placeholder `data/source_scene.tif`. Supply the actual corresponding source scene and `development_roi.tif` to regenerate spatial splits. These source rasters are not part of the package. This conversion does not claim to recover a historical 5,640-object ablation batch or historical main-training manifest.

The unchanged feature, identity, split and manifest records used for the simple-classifier experiment remain under `消融实验代码/随机森林消融实验/data/`. Its frozen runner uses those saved subsets and resolves packaged masks by sample ID. See the data/splits guide for the difference between manifest count, excluded samples, internal validation and same-scene spatial test.
