# Evaluation definitions

## Regional lake segmentation

Lake is positive. Accumulate TP, FP, FN and TN over non-overlapping eligible evaluation pixels before calculating scores. Do not average scene percentages to obtain this pooled result.

| Metric | Definition | Practical meaning |
| --- | --- | --- |
| Precision | TP/(TP+FP) | Reliability of extracted water; penalizes dark-land false detections |
| Recall | TP/(TP+FN) | Completeness; penalizes omitted water |
| F1 | 2TP/(2TP+FP+FN) | Balance between false detections and omissions |
| Lake IoU | TP/(TP+FP+FN) | Overlap of predicted and reference water |
| Background IoU | TN/(TN+FP+FN) | Background classification |
| mIoU | (Lake IoU + Background IoU)/2 | Equal weighting of the two classes |

Apply the same reference, grid and eligibility mask for a paired method comparison. Exclude missing optical observations; do not count them as dry land. The MNDWI supplement and older reported SAR results do not all share the same valid mask.

At 20 m, a cell has area 400 m². Validate a projected metre grid before using this conversion. A 512 × 512 input covers 104.8576 km² on this grid. This is input footprint, not a measured effective receptive field.

## Contour classification

Lake-positive Precision, Recall and F1 answer whether candidate contours are correctly accepted. They are not the same as final lake-mask pixel scores. `weighted-F1` first calculates each class F1 and averages by class support; `macro-F1` gives classes equal weight. Neither should replace the lake-positive scores in the imbalanced ablation presentation.

AP is non-interpolated average precision from saved lake probabilities. It cannot be recovered from one confusion matrix. The historical 5,640-sample tables lack these probabilities; the simple-classifier comparison supplies them.

## Shoreline evaluation

Use one-pixel-wide boundaries. Boundary precision measures predicted boundary pixels within tolerance of reference boundaries; boundary recall reverses the direction; Boundary F1 is their harmonic mean. Main shoreline comparisons use **3 pixels = 60 m on the 20 m grid**. HD is the maximum bidirectional boundary distance, in kilometres. HD95 reduces the influence of the most extreme distances but needs its percentile convention recorded with results.

For the three main lake cases, retain the largest predicted connected component for each target. These scores describe the retained main lake and exclude disconnected fragments. This filtering is not applied to all regional pixel scores. Report per-lake results; an aggregate uses their arithmetic mean. Do not pool boundary cases into the regional pixel confusion matrix.

The four-lake IoU/Boundary-F1 diagnostic uses a different target-association rule and is illustrative, not a size-stratified sample. Record this difference before comparing its scores with the main three-lake table.

## Candidate coverage and small-lake errors

Candidate coverage is reference-boundary recall before classification. The saved analysis uses a registered **60 m analysis grid** and **60 m matching tolerance**, not a one-pixel tolerance on the 20 m grid. Candidate rasters are not dilated. For 42 lakes larger than 1 km², 48,960/50,018 reference boundary pixels are covered (97.88%). This is neither object detection rate nor final segmentation accuracy; it does not assess candidate precision or closure.

The small-lake supplement covers 28,222 reference lakes smaller than 1 km² within available prediction coverage. It merges bounding boxes expanded by 20 pixels, deduplicates pixels, and calculates direct overlap without tolerance or new registration. Precision/Recall/F1/IoU are 64.10/12.39/20.77/11.59%. These are pooled pixel scores, not detected-object counts or medians of individual lake IoUs.

The IoU ≥0.95 category in the error figure means high area overlap. Below it does not imply omission, and above it does not guarantee every shoreline detail. Inspect boundary scores and overlays separately.

## Repetition summaries

Ten-seed uncertainty summaries use mean and sample standard deviation (denominator n−1). They describe training variability on a fixed split, not variability across ten geographic test regions. The 30-repeat efficiency range is a different statistic. Do not label either one a confidence interval or claim statistical significance without the corresponding analysis.
