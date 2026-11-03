# Small-lake aggregate

The author-supplied count record covers 28,222 reference lakes below 1 km² inside existing prediction coverage. Bounding boxes are expanded by 20 pixels and merged to count each pixel once. Prediction and reference are compared directly, without tolerance or new regional registration. The stored percentages retain the reported rounding.

This is a pooled pixel comparison, not a lake-object detection rate, a median per-lake IoU or a complete set of area-bin evaluations. Source-wide project rasters are not included.
