# Candidate reference-boundary coverage

The recorded analysis uses pre-classification candidates, a frozen registration, a 60 m analysis grid and a 60 m Euclidean matching tolerance. Candidate rasters are not dilated. For 42 lakes with reference area >1 km², 48,960 of 50,018 eligible shoreline pixels match a candidate, giving 97.88% pooled coverage. The result is not candidate precision, lake-object detection rate, contour closure or final lake-mask accuracy.

Step 01 prepares aligned rasters from the external paths in `配置与配准记录.json`; Step 02 calculates coverage from them. Prepared TIFFs are outside the source package. Hash/count checks preserve this particular archived analysis. For a new dataset, create and document a new configuration rather than forcing its results to match the old counts.

The analysis grid differs from the main 20 m evaluation grid: 60 m is one analysis pixel here and three main-grid pixels. `输入准备记录.json` records the included lake IDs and the saved preparation scope.
