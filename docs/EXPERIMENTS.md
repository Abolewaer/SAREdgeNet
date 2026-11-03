# Experiment entry points

## Baseline training and inference

Each neural baseline has `Step1` training, `Step2` prediction and `Step3` evaluation under `对比实验代码/`. For example:

```powershell
python 对比实验代码/UNet_Step1.py --config configs/UNet.json
python 对比实验代码/UNet_Step2.py --config configs/UNet.json
```

These are long-running commands. They are examples, not actions performed during packaging. Configure real paths first and use an output directory separate from recorded results.

U-Net and Attention U-Net have custom convolutional encoders. DeepLabV3+, FPN and PSPNet use ResNet-50 in the current capacity versions. The parameter-count record is [模型参数核算.json](../模型参数核算.json). It describes specified implementations, not the original architecture names in isolation.

The supplied UNet/AttentionUNet configurations remain one-channel SAR examples. DeepLabV3+, FPN and PSPNet templates use two channels with their method-specific DEM preparation. Do not describe this mixed set of examples as the exact five-network SAR+DEM result batch. The added DEM result CSV is preserved separately; its corresponding final per-method run configurations still need to be attached to a release if recovered. Changing `input_channels` in JSON alone does not implement a missing two-channel dataset or restore a historical checkpoint.

For PSPNet/FPN SAR+DEM code, [sar_dem_baselines.py](../saredge/sar_dem_baselines.py) aligns raw elevation in metres by bilinear resampling, clips to [−500,9000] m and divides by 9000, filling invalid values with zero. This differs from the main classifier's extent-normalized DEM descriptors. DeepLabV3+ has its own preparation in its entry point. Keep input metadata with checkpoints; do not swap normalized descriptors and metre-valued image channels.

Sliding-window inference uses 512 × 512 inputs with 64-pixel overlap and 448-pixel stride. Internal edges are cropped before stitching; probability-weighted overlap fusion is not implemented. Visible seams concern this implementation, not an inevitable property of these architectures.

## Ten training seeds

The fixed training seeds are **2, 12, 22, 32, 42, 52, 62, 72, 82 and 92**. Split seed 42 remains fixed separately.

```powershell
python 对比实验代码/SeedRuns_Step1.py --mode plan --config 对比实验代码/ten_seed_comparison.json
```

`plan` writes the 60-job plan and lists missing inputs without launching training. `--mode run` explicitly launches the six methods × ten seeds. `--mode collect` verifies completed run records and aggregates pixel metrics. Use a fresh destination when code or configuration changes.

SAREdgeNet selects by validation weighted-F1; U-Net, Attention U-Net, FPN and PSPNet select by validation accuracy; DeepLabV3+ selects by validation loss. Testing is not used to select the checkpoint. Earlier last-epoch results and signed offsets are superseded by ten-run means and sample standard deviations. The supplied [ten-run CSV](../对比实验代码/十次对比实验/十种子.csv) is an author-provided result record; preparation of the current orchestrator does not establish that all historical jobs were executed by that exact file.

## Contour ablations

Three result batches remain separate:

1. Archived backbone/pretraining comparisons on 5,640 evaluation contours.
2. A–F modality/attention comparisons on 5,640 evaluation contours.
3. Full dual branch, numerical Attention-MLP and random forest on the saved local spatial holdout (2,652 test contours).

The first two use confusion-matrix counts for lake-positive Precision, Recall and F1. They do not supply individual scores from which AP or ten-run uncertainty can be reconstructed. The newer runnable A–F interface does not retroactively supply missing historical IDs.

The A–F variants are defined in [protocol.py](../消融实验代码/新消融实验/protocol.py). A uses the image branch alone; B adds 6 DEM features; C adds 8 geometry features; D adds both (14); E uses all 17 with attention; F uses all 17 without attention. E/F isolate the attention switch. C is not a strict “full model minus DEM” experiment, because it also lacks SAR statistics and attention.

```powershell
python 消融实验代码/新消融实验/run_ablation.py --config 消融实验代码/新消融实验/config.template.json --stage plan
```

Paths in this particular template resolve relative to its JSON file. The template points to the included 5,636-record dataset as an editable starting point, not the missing original 5,640-object run manifest. Processing requires `--run`; a 60-run matrix additionally requires `--allow-batch`.

The saved simple-classifier comparison has its own fixed manifest and predictions:

```powershell
python 消融实验代码/随机森林消融实验/run_comparison.py --models random_forest
```

This command trains a model. It defaults to random forest; explicitly request `numeric_mlp` or `full_dual_branch` for neural comparisons. Both neural models use Adam, learning rate 0.001 and 80 epochs, selecting by internal validation weighted-F1. The saved run uses seed 42. The `audit_results.py` script can recompute metrics from saved probabilities; `--require-checkpoints` and `--require-source-table` additionally require the original external files.

## Timing

[Efficiency_Step1.py](../对比实验代码/Efficiency_Step1.py) collects repeated complete-workflow timing; Step 2 summarizes. Use `--config`, `--repeats 30` and `--output` as defined in [efficiency_cli.py](../saredge/efficiency_cli.py). The benchmark does not perform a model warm-up. Primary timing and external child-process wall time are recorded separately. Read the selected scope before comparing measurements.

The paper's 23 ± 1 s is the rounded observed range 22–24 s, expressed as midpoint ± half-range. It is not a standard deviation. The historical RTX 4080 measurement does not have a recovered input-size/candidate-count run record in this package; no synthetic test or new machine timing is substituted for it. Training-time ranges estimated from epoch durations are not measured end-to-end training durations.

## Fixed-model cases and supplementary analyses

The four groups under `稳定性实验代码/` use one trained classifier: wind-background cases, additional regions, dunes and small-waterbody errors. Cross-region configuration references are null because these five cases have no contemporaneous reference polygons; output overlays do not imply quantitative accuracy.

The extra [analysis](../analysis/README.md) scripts reproduce archived postprocessing and candidate-parameter investigations when their required local inputs are supplied. They are not run at import for inspection; execute their files only when processing is intended. The archive scripts include top-level work, so do not import them as libraries.

The legacy `OSTU_Step1.py` retains the original name and historical implementation. Its code estimates intensity statistics within reference polygon regions and applies a mean ±2σ rule; it is not a classic between-class-variance Otsu implementation. It is retained for provenance, not silently redefined as classic Otsu. The manuscript's OTSU-labelled historical result cannot be certified from this legacy file alone.
