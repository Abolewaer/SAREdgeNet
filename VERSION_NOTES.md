# Version notes

## Recovered project and public version

The previously inspected public repository exposed an early classifier with 15 numerical inputs and a shorter MLP. Project files were later migrated between computers after an office move, and the public copy was not synchronized with the working implementation. This revision release assembles recovered files, migration interfaces and revision analyses. The replacement preserves the existing repository URL and earlier public commits.

The classifier documented here uses 17 descriptors, a five-linear-layer numerical branch, residual feature attention, LayerNorm and GELU. A 15-input checkpoint is not interchangeable with this model. [来源记录.json](来源记录.json) preserves recorded source-file provenance. Provenance records and current source hashes have different roles: editing a migration interface does not make the new file the original historical script.

## Local preparation on 3 November 2026

- Consolidated the duplicate ablation trees under `消融实验代码/`. Retained the improvement permitting null early stopping for fixed 80-epoch comparisons.
- Corrected the nested ten-seed runner's repository root and default JSON location, wrapper imports and the simple-classifier import path.
- Supplied 5,636 matching original contour PNGs and retained the label/feature/split table bytes and historical hashes. The simple-classifier runner resolves image paths by ID in the packaged data directory.
- Completed the migrated Step 3 DEM preparation: raw metre-valued inputs are aligned, clipped at μ±2σ and normalized to [0,255] over the processing extent, with a separate validity mask. The former migrated CLI only aligned. This repair implements the documented input preparation; it is not a new training run or a retroactive verification of historical scores.
- Made archival analysis inputs explicit through environment variables and corrected null references in qualitative cross-region examples.
- Kept completed comparison/ablation CSVs, per-sample simple-classifier predictions, postprocessing and sensitivity supplements. Invalidated boundary diagnostics were removed from the publication tree.
- Moved local environments, synthetic smoke outputs, downloaded scenes, large generated rasters, model binaries and superseded task notes to a separate local archive. Only disposable Python caches were deleted.

No training, inference, full-raster analysis or smoke experiment was executed for this preparation. Syntax, configuration structure, packaged file correspondence and recorded numeric consistency were checked statically. Those checks do not establish that every migrated workflow reproduces a historical run.

## Records still needed for an exact historical reproduction

The existing result tables are not accompanied by all original full-run metadata. In particular, the five-network added-DEM batch still needs its exact per-method trained configurations/checkpoints; historical 5,640-contour batches lack complete ID/probability manifests; original timing needs the paired image dimensions and candidate count; the historical SAR export needs its exact task recipe. These gaps are kept explicit rather than filled with defaults. The recorded results remain unchanged.
