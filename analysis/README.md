# Supplementary analysis scripts

These scripts preserve the completed local investigations described in the responses. They are separate from model training and require the corresponding external source files. They were not executed during this cleanup.

| Script | Inputs and output |
| --- | --- |
| [Postprocessing](postprocessing/evaluate_stages.py) | Archived RGB stage rasters, ROI, public reference, fixed registration; produces stage counts and pixel-change records |
| [Sensitivity](sensitivity/visual_check.py) | Archived raw SAR, Kokoxili extent and source window; produces local five-setting candidate figures |
| [DEM completeness](../DEM缺失率统计/audit_dem_missingness.py) | Raw DEM mosaics, 18 archived tiles and feature tables; produces missingness counts |
| [Candidate coverage](../Dog与真实湖泊召回率统计/02_计算60m容差召回率.py) | Prepared 60 m candidate, reference-boundary and valid masks; produces pooled boundary coverage |

The first three use `SAREDGE_ARCHIVE_ROOT` to identify the external recovered project tree. Their historical relative layout is preserved under that root. They contain top-level processing and are scripts, not importable utility libraries. Set the variable explicitly, inspect inputs and output paths, then invoke a script only when computation is intended. They do not make private raw data part of this code package.

The candidate-coverage preparation uses its saved registration configuration and checks reference/candidate hashes. Supply its external inputs or prepared rasters before running; it is a frozen case reproduction, not a universal evaluator that should force every input to yield 97.88%.
