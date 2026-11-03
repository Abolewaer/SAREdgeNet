# Recorded results and provenance

These files preserve completed records; the package cleanup did not retrain models or replace results with synthetic data. Numeric CSVs are kept in their original units unless a derived summary explicitly says otherwise.

| Manuscript or response material | Source in this package | Scope |
| --- | --- | --- |
| Abstract and ten-run Figure 3 summaries | [十种子.csv](../对比实验代码/十次对比实验/十种子.csv), [derived summary](ten_seed/summary.csv) | Six neural methods × ten seeds; regional pixel metrics |
| Added SAR+DEM comparison, Table 3 | [comparison_results_withDEM.csv](../对比实验代码/对比实验结果/comparison_results_withDEM.csv) | Separate supplied comparison batch, not ten new runs per model |
| Original reference comparison values | [paper_comparison_results.csv](../对比实验代码/对比实验结果/paper_comparison_results.csv) | Historical single-result record |
| Backbone/pretraining ablations, Table 6 | [original counts](../消融实验代码/新消融实验/原消融实验/原稿消融_图9原始数据.csv) | 5,640 contours; lake-positive scores derived from counts |
| Modality/attention ablations, Table 7 | [新消融.csv](../消融实验代码/新消融实验/新消融.csv) | A–F and archived configuration counts; `F1_macro` is not lake-positive F1 |
| Simple classifiers, Table 8 | [comparison_metrics.csv](../消融实验代码/随机森林消融实验/outputs/comparison_metrics.csv), [predictions](../消融实验代码/随机森林消融实验/outputs) | Seed 42; separate internal validation and 2,652 same-scene spatial-test contours |
| MNDWI supplement | [external and other scopes](mndwi/全区域_阈值0.1/全区域指标.csv) | External common optical–SAR valid pixels, threshold 0.1 |
| Candidate reference-boundary coverage | [召回率结果.csv](../Dog与真实湖泊召回率统计/召回率结果.csv) | 42 reference lakes >1 km²; 60 m analysis grid and tolerance |
| Postprocessing | [stage metrics and example](postprocessing/README.md) | Kokoxili ROI, all lake sizes; frozen registration |
| Parameter comparisons | [images and settings](sensitivity/README.md) | Local candidate stage only; five settings per parameter |
| Contour resizing | [native and 224 example](contour_resize/native_and_224.png) | Same original mask and resized input; no accuracy claim |
| DEM completeness | [by-tile CSV](dem_missingness/by_tile.csv), [audit](dem_missingness/audit.json) | Archived 18 processed DEMs; zero is not necessarily an original data gap |
| Acquisition-date evidence | [catalog records](acquisition_candidates/README.md) | Spatial/date candidates, not confirmed historical source lineage |
| Early U-Net development result | [original output](development/Kekexili_UNet_30percent.png) | Motivation evidence, not controlled current-model benchmarking |

The 5,636 included labeled PNGs are not a one-to-one reconstruction of the historical 5,640-object ablation batches. Do not silently add or duplicate samples. Similarly, the added DEM batch, MNDWI footprint and ten-run batch are not interchangeable evaluation datasets.

Some records contain historical workstation paths and hashes. Those document their original source, not a promise those external files are included. Raw imagery, project boundaries, generated TIFFs and trained model binaries were moved out of the source package. Their omission does not turn new configuration templates into verified historical settings.

Additional supplements: [three-lake HD95](boundary/README.md), [small-lake pooled counts](small_lakes/README.md), and [four illustrative lake sizes](lake_size_examples/README.md). These retain their own source and evaluation limits.
