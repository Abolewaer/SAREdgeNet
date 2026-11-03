# Setup and main workflow

## Environment

Use Python 3 with mutually compatible PyTorch and torchvision versions. Install GDAL, rasterio, pyproj, geopandas and their native dependencies from a consistent environment, such as one Conda channel; avoid mixing incompatible GDAL binary builds. Then install the remaining [requirements](../requirements.txt). The requirements list records required libraries, not a locked reconstruction of every historical machine.

Scripts retain some Chinese file and directory names. Use UTF-8 and quote paths containing spaces. Commands below assume this source directory is the working directory, regardless of where it sits in a Git checkout. No `organized/` parent directory is required.

## Required inputs

Prepare calibrated, terrain-corrected single-band SAR, a georeferenced DEM in metres, a development-region raster defining its valid footprint, labeled contour data for training, and lake reference polygons for evaluation. The workflow does not perform Sentinel-1 radiometric calibration or terrain correction.

Edit [SAREdgeNet.json](../configs/SAREdgeNet.json) and [SpatialProtocol.json](../configs/SpatialProtocol.json). The latter expects `data/development_roi.tif`; supply the actual Kokoxili development footprint rather than an arbitrary rectangle. If the SAR has a geographic CRS, choose a suitable projected CRS and 20 m resolution before interpreting pixel area. Setting `resolution` alone does not reproject when `target_crs` is null.

| Step | Script | Main operation |
| --- | --- | --- |
| 1 | `主实验代码/SAREdgeNet_Step1.py` | Clip and normalize SAR, Gaussian/median filtering, grayscale opening |
| 2 | `主实验代码/SAREdgeNet_Step2.py` | Apply recorded grid correction and optional reprojection |
| 3 | `主实验代码/SAREdgeNet_Step3.py` | Align raw DEM; clip and normalize within the reference extent |
| 4 | `主实验代码/SAREdgeNet_Step4.py` | Generate candidate IDs, original mask PNGs and numerical descriptors |
| 5 | `主实验代码/SAREdgeNet_Step5.py` | Train on manually labeled contours; save classifier, scaler and split |
| 6 | `主实验代码/SAREdgeNet_Step6.py` | Classify candidate contours and map accepted IDs back to the source grid |
| 7 | `主实验代码/SAREdgeNet_Step7.py` | Filter boundary support and reconstruct water by constrained seed growth |
| 8 | `主实验代码/SAREdgeNet_Step8.py` | Refine boundaries and remove unsupported branches |
| 9 | `主实验代码/SAREdgeNet_Step9.py` | Fill eligible internal gaps |
| 10 | `主实验代码/SAREdgeNet_Step10.py` | Produce lake polygons and, when references are supplied, regional metrics |

## Running the workflow

Inspect the commands first:

```powershell
python Run_Step1.py --config configs/SAREdgeNet.json --steps 1 2 3 4 --dry-run
```

When the paths and grid are correct, remove `--dry-run` to prepare candidates. Step 4 does not invent labels. Assign lake/non-lake labels, provide their spatial mapping, and set `train_table` and `train_image_dir` before training:

```powershell
python Run_Step1.py --config configs/SAREdgeNet.json --steps 5
```

For a new prepared scene, keep the trained checkpoint and scaler at the configured Step 5 location, then run:

```powershell
python Run_Step1.py --config configs/SAREdgeNet.json --steps 6 7 8 9 10
```

Do not overwrite one run with a different protocol. Use separate output directories for different seeds, modalities or input scenes. `run_manifest.json` records the requested commands; it is a plan, not evidence that a command completed.

The `--table` mode of Step 4 upgrades an old ID + 15-feature + label table by deriving Compactness and Extent from the matching original binary PNGs. A final model still uses 17 descriptors. Resized images or SAR patches cannot substitute for the original contour masks in this conversion.
