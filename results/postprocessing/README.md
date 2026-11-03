# Postprocessing stage comparison

[stage_metrics.csv](stage_metrics.csv) compares the seed-grown initial lake mask, boundary refinement, and internal-gap filling in the same Kokoxili ROI. [pixel_changes.csv](pixel_changes.csv) separates added/removed pixels by reference water membership. [stage_example.png](stage_example.png) illustrates both cleanup and loss of local shoreline structures.

After refinement, Precision rose from 74.94% to 94.58%, F1 from 82.94% to 91.69%, and Recall fell from 92.85% to 88.97%. Filling raised Recall to 89.89% and F1 to 92.10%, with Precision 94.42%. These results show a substantial contribution and a tradeoff, not negligible postprocessing effects.

The initial mask already includes seed growth: this is not an experiment disabling every postprocessing operation. Comparisons use one frozen 43-control-point registration, the same ROI, reference and valid grid, no per-stage registration fitting and no largest-component filter. All lake sizes are included. Boundary scores use 60 m tolerance and need caution because the frozen registration's median validation residual exceeds that distance. These are local supplementary results, not replacements for the external regional test.

Code: [evaluate_stages.py](../../analysis/postprocessing/evaluate_stages.py). Restricted source rasters and registration inputs must be supplied separately.
