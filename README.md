# SAREdgeNet

SAREdgeNet extracts lakes from large SAR images by separating full-image candidate generation from contour-level classification. It combines a ResNet-34 contour-mask branch with 17 geometric, SAR-intensity and DEM descriptors, then reconstructs lake masks from accepted contours.

This repository provides code and permitted supporting data for the Water revision of **SAREdgeNet: Lightweight Parallel Deep Learning via Task Decoupling for Large-Scale SAR Lake Segmentation** (water-4645099).

The 3 November 2026 update replaces the early top-level scripts with the organized workflow, comparison experiments, ablations, evaluation tools and supporting records below. Earlier files remain accessible through Git history. The previous public version was not synchronized after an office move and migration between experimental computers; [the version notes](VERSION_NOTES.md) explain the recovered files, subsequent repairs and remaining limits of historical reproduction.

## Start here

| Guide | What it explains |
| --- | --- |
| [Setup and workflow](docs/WORKFLOW.md) | Dependencies, inputs, Steps 1–10 and commands |
| [Method and features](docs/METHOD.md) | Mask input, 17-feature order, DEM preparation and postprocessing |
| [Data and splits](docs/DATA_AND_SPLITS.md) | Manual labels, included PNGs, development/test geography and restrictions |
| [Experiments](docs/EXPERIMENTS.md) | Baselines, ablations, repeated training, timing and case workflows |
| [Evaluation](docs/EVALUATION.md) | Positive class, aggregation, boundary tolerance, units and score interpretation |
| [Results](results/README.md) | Files supporting manuscript tables and response supplements |
| [Version notes](VERSION_NOTES.md) | Recovered files, migration changes and this cleanup |

## Directory guide

| Directory | Contents |
| --- | --- |
| `主实验代码/` | Main Steps 1–10 |
| `saredge/` | Shared models, raster operations, evaluation and spatial protocol |
| `configs/` | Editable examples; check paths and training settings before running |
| `对比实验代码/` | Segmentation comparisons, optical indices, repeated-run and timing entry points |
| `消融实验代码/` | Backbone, input/attention and simple-classifier comparisons |
| `稳定性实验代码/` | Fixed-model case workflows |
| `DEM缺失率统计/` | Archived-data completeness audit |
| `Dog与真实湖泊召回率统计/` | Registered 60 m candidate-boundary coverage analysis |
| `analysis/` | Postprocessing and parameter-sensitivity analysis scripts |
| `data/contours/` | 5,636 manually labeled candidate-mask PNGs |
| `results/` | Recorded tables and selected explanatory figures; no raw imagery |

## First use

Run commands from this directory. Install a compatible PyTorch/torchvision build and GDAL environment, then the remaining packages in [requirements.txt](requirements.txt). See the setup guide before installing geospatial dependencies.

```powershell
python Run_Step1.py --config configs/SAREdgeNet.json --steps 1 2 3 4 --dry-run
```

This prints commands and writes a plan; it does **not** process imagery or train a model. Edit the input paths and spatial protocol before removing `--dry-run`. Training, inference and all long analyses require an explicit separate command; no experiment is launched merely by opening the package.

## Read results in their own context

The main regional pixel evaluation, ten-seed summaries, added SAR+DEM comparisons and local contour ablations are separate records. Their sample units and coverage differ. [The results index](results/README.md) identifies each source, so local classification scores are not confused with regional segmentation accuracy.

Main model development uses Kokoxili in Yushu, Qinghai. Lake and non-lake contours are each split 80:20 into training and validation; regional pixel testing excludes this development area. This does not make the internal contour split lake-ID-disjoint. Five additional lake cases are qualitative applications of the original model, not global accuracy tests.

The source package excludes installed environments, synthetic smoke outputs, downloaded satellite scenes, large intermediate rasters and training binaries. Restricted project-wide boundaries and the non-public 2022 wind-speed extension are not included. The included labeled contour masks are separate from those restricted regional datasets.

[FILE_MANIFEST.csv](FILE_MANIFEST.csv) records the path, byte size and SHA-256 of every packaged file except the manifest itself.
