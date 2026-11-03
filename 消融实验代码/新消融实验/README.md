# Modality and attention ablations

This is the canonical A–F implementation. It retains support for 80 full training epochs with `early_stopping_patience: null`; that small improvement was merged from the duplicate working folder before its removal.

Read [the experiment guide](../../docs/EXPERIMENTS.md) for variants, commands and distinction between historical 5,640-contour counts and the included 5,636-mask dataset. `config.template.json` is editable and resolves paths relative to itself. A plan performs no training; processing requires `--run`, and the 60-run matrix also requires `--allow-batch`.

`新消融.csv` is the supplied result record. Its `F1_macro` column is macro-F1; manuscript lake-positive F1 is calculated from TP/FP/FN. The record does not provide missing per-sample historical probabilities or IDs. Files under `原消融实验/` preserve the original confusion counts and derived metric definitions.
