# Four illustrative lake sizes

`metrics.csv` and `shoreline_examples.png` show the same prediction mosaic and reference SHP for four selected reference-area classes. The sub-1 km² example was deliberately selected for high IoU. These are illustrative objects, not a representative sample or area-class averages.

IoU uses direct lake-pixel overlap, without distance tolerance or additional registration. Boundary F1 uses one-pixel inner boundaries, including island shores, with boundary centres transformed to local UTM and 60 m bidirectional matching. Predicted components overlapping the target are combined; there is no largest-component filter, dilation or closing. Unassociated nearby predictions are excluded, so this is object-associated diagnosis rather than complete-window false-positive evaluation.

The 0.955 km² example has IoU 94.55% and Boundary F1 100% within 60 m. This illustrates why the 0.95 IoU category is not a necessary or sufficient guarantee of every shoreline detail. The main three-lake boundary protocol has a different component-selection rule.
