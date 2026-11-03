# Data and spatial splits

## Included contour data

[data/contours](../data/contours) contains 5,636 original binary PNGs matching the saved manual-label manifest. Candidate IDs name the files. The companion [feature table](../消融实验代码/随机森林消融实验/data/features17_manual_labels.csv), [manifest](../消融实验代码/随机森林消融实验/data/shared_manifest.csv), [split](../消融实验代码/随机森林消融实验/data/shared_split.csv) and [identity audit](../消融实验代码/随机森林消融实验/data/image_identity_audit.csv) retain their original bytes and hashes.

Historical tables contain original local image paths. The simple-classifier runner now resolves the included PNGs by sample ID under `data/contours`, leaving the archived table unchanged. Coordinate-mapping files refer to the original source scene; supply that scene and adjust mapping paths when generating a new spatial split. Do not substitute arbitrary imagery with the same file name.

Labels were assigned manually from Kokoxili candidate contours viewed with SAR context. `1` means a lake-shoreline candidate; `0` means non-lake. Candidate IDs and display colors do not encode the label. Cropping and manual selection avoided visibly truncated lakes where possible. This reduces crop fragments but does not prove every same-lake candidate belongs to one subset.

The manifest includes records excluded by the fixed ROI-boundary policy. Its total is not the test size: the saved simple-classifier comparison uses 1,898 training, 475 internal-validation and 2,652 same-scene spatial-test contours. Read `subset` rather than inferring membership from file order. The historical 5,640-contour ablation counts are a different archived batch; no extra four PNGs are fabricated to make the totals match.

## Main development and regional evaluation

The development area is the Kokoxili lake region in Yushu, Qinghai. Split its lake contours 80:20 and its non-lake contours 80:20, both with split seed 42, then combine the two training portions and the two validation portions. Training fits model parameters, feature scaling and current class weights. Validation selects checkpoints and configured learning-rate/early-stopping decisions.

Regional pixel evaluation uses the other covered areas of the Tibetan Plateau. The evaluator applies the spatial test mask; it does not discard every source scene that intersects the development region. Input images are mosaicked and evaluation tiles do not overlap, so each eligible pixel contributes once.

For segmentation baselines, the unit is a patch rather than a contour. The current spatial protocol uses non-overlapping training patches wholly within the development region, splitting lake-containing and background patches 80:20 separately. It excludes boundary-crossing and invalid patches. Overlapping inference is a different operation and remains enabled.

This protocol is not the old random 64:16:20 description. Nor does it assert lake-ID-disjoint internal validation or independent source-scene testing for every local experiment. The random-forest comparison specifically holds out the ROI exterior **within the same source scene**.

## Public sources and temporal limits

- Sentinel-1 SAR: calibrated, terrain-corrected inputs obtained through ASF or GEE.
- SRTM: nominal 30 m terrain observations acquired in 2000, aligned to the processing grid.
- 2022 lake reference: the public [Lake_TP data release](https://doi.org/10.5281/zenodo.10686952) and its [data paper](https://doi.org/10.1016/j.isprsjprs.2024.08.012). Communication with its authors additionally confirmed Landsat-based NDVI extraction and manual correction. Summer observations are combined to reduce cloud obstruction; an annual reference is not a same-day SAR shoreline.

The uncorrected automatic NDVI baseline and the manually refined public lake product are distinct. Shared optical-index ancestry can still create correlated errors. MNDWI uses its own common optical–SAR valid mask and cannot be treated as a paired comparison against methods evaluated on other pixels.

[Catalog date queries](../results/acquisition_candidates/README.md) identify spatially intersecting candidate source products. They do not recover the exact historical GEE export recipe. Do not replace an unknown acquisition date with a TIFF creation time or choose an arbitrary catalog match.

## Data outside this package

Restricted project-wide boundary data and the research-use 2022 monthly wind-speed extension cannot be redistributed. Full-scene SAR, optical downloads, raw DEM mosaics and trained binaries are kept outside the Git source directory. The package provides input schemas, public source citations, selected permitted contour data and recorded evaluation outputs without claiming that every historical input is publicly supplied.
