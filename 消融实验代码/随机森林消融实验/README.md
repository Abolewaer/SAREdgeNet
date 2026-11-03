# Simple-classifier comparison

The saved comparison uses one fixed manifest and 17 descriptors for random forest, numerical Attention-MLP and the full dual-branch classifier. The full model additionally reads the matching contour masks. Seed is 42; the two neural models use Adam, learning rate 0.001 and 80 epochs. The best checkpoint is chosen by internal validation weighted-F1. Random forest uses the fixed settings in `config.json`, without parameter search.

There are 1,898 training, 475 internal-validation and 2,652 spatial-test contours after the recorded boundary policy. Spatial-test samples lie outside the Kokoxili ROI **within the same original SAR scene**; they are not an independent-scene test. The full manifest also contains excluded samples. Subset IDs and manual labels are retained in `data/`.

The numerical MLP has the highest test AP (93.59%), random forest the highest test F1 (77.66%). Full-model test F1 is 72.64%, numerical-MLP F1 73.00%. All have Recall 96.05%. These are contour-level outcomes, not final reconstructed lake-mask pixel scores. Internal validation is reported separately.

Saved model-wise probabilities, metrics, selected-epoch histories and hashes remain in `outputs/`. Checkpoint binaries are archived outside the source package. Original data-table bytes are unchanged for hash verification; the runner resolves PNG paths by ID under `data/contours`. To train neural models, provide the ImageNet weight file configured under `data/weights/`.

```powershell
python 消融实验代码/随机森林消融实验/run_comparison.py --models random_forest
```

This trains. `audit_results.py` only checks supplied results; checkpoint/source-table verification is optional unless explicitly required with its corresponding flags. `prepare_real_data.py` is an archival preparation script: set `SAREDGE_CONTOURS_DIR` to the original contour/source-mask directory before running. It requires the actual manual-label workbook and source component mask, not fabricated labels.
