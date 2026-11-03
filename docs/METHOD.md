# Method and feature definitions

## What is parallel

The two classifier branches process the same candidate independently before fusion. “Parallel” describes this architecture, not measured hardware concurrency. The whole extraction workflow remains staged.

## Image input and classifier

The image branch reads a black-background, white-candidate **contour mask**, not a grayscale SAR patch. The original crop has nominal side length `2 * floor((max(width,height)+10)/2)` with 5-pixel padding; crops may be shorter at source boundaries. Resize to 224 × 224 using PIL LANCZOS, convert to one channel, then apply `ToTensor` and normalization `(x - 0.485) / 0.229`. Training additionally uses horizontal flipping with probability 0.3 and rotations within ±10°. Evaluation has no random augmentation.

ResNet-34 starts from ImageNet weights, averaging RGB first-layer kernels into one channel. Its 512-dimensional output is projected through 256 → 128 → 64. The numerical branch is 17 → 128 → 128 → 64 → 64 → 32, with GELU and dropout. After its second linear layer, a 128 → 16 → 128 sigmoid gate modulates features as `x * (1 + gate(x))`, followed by LayerNorm. It is feature attention, without Q/K/V or multiple attention heads.

The 64- and 32-dimensional outputs concatenate into 96 values. The final classifier is 96 → 64 → 32 → 2. Training uses logits; the predicted class is their argmax. Implementation: [models.py](../saredge/models.py).

## Numerical feature order

The feature vector follows this order. Compactness and Extent were appended to the earlier 15-feature layout; geometry is therefore not stored as eight consecutive columns.

| Position | Feature | Definition or sampling scope |
| --- | --- | --- |
| 1 | Edge Pixel Count | Foreground count in the original candidate mask |
| 2 | Bounding Box Length(px) | First image-axis-aligned bounding-box dimension |
| 3 | Bounding Box Width(px) | Second bounding-box dimension |
| 4 | L/W + W/L | Sum of reciprocal ratios; **one scalar** |
| 5 | Curvature Feature | `(1 - A/Ah) * P²/(4πA)` |
| 6 | Roughness | `P/Ph` |
| 7–9 | Pixel Mean, Pixel Std, Pixel Median | Preprocessed SAR values at candidate-contour pixels |
| 10–14 | DEM Mean, DEM Std, DEM Median, DEM Min, DEM Max | Valid normalized DEM samples at candidate-contour pixels, after IQR filtering |
| 15 | Valid DEM Pixels | Count of retained DEM samples |
| 16 | Compactness | `4πA/P²` |
| 17 | Extent | `A/(bounding-box width * height)` |

`A` and `P` are area and closed perimeter of the main external contour; `Ah` and `Ph` are its convex-hull area and perimeter. The curvature descriptor combines concavity and shape complexity; it is not differential point curvature. Geometry is measured before resizing. These geometric areas are not the final mapped lake area. No latitude or longitude enters the vector.

## DEM preparation and missing data

Step 3 takes raw elevation metres, aligns them by bilinear resampling to the SAR processing grid, and retains a separate validity mask. Across that **reference processing extent**, valid elevations are clipped to mean ± 2 standard deviations and linearly mapped to [0,255]. This is not normalization within each contour crop. Do not pass an already normalized DEM as raw metres and normalize it again.

Step 4 samples the normalized DEM at each contour, discards invalid values, and removes samples outside `Q1 - 1.5*IQR` to `Q3 + 1.5*IQR`. The retained values yield the six DEM descriptors. If none remain, the current implementation returns zeros including a zero valid-count descriptor. A valid minimum that maps to zero remains valid when its mask says so; zero alone does not identify original acquisition gaps.

After descriptors are computed, the classifier standardizes numerical columns using training-set means and standard deviations. Save these parameters and reuse them for validation and inference. This feature standardization is separate from the preceding spatial DEM normalization.

The height statistics express elevation relative to the processing extent, rather than absolute metres. This removes a uniform elevation offset in an otherwise identical landscape, but does not make different landscapes statistically identical. Resampling SRTM from 30 m to a 20 m grid adds no independent terrain detail.

## SAR preparation and candidate generation

Finite, non-zero SAR samples determine the upper clip `μ + 2σ`; clipped values are mapped to [0,255]. Apply Gaussian filtering (σ=1), 3 × 3 median filtering and 3 × 3 grayscale opening (erosion then dilation). These SciPy operations are not Lee or Refined Lee filtering.

The candidate sequence uses DoG scales 2 and 6, response threshold 50, and successive minimum eight-connected-component sizes 200, 200 and 500 pixels. A 4 × 4 opening and 2 × 2 erosion occur between filtering steps. Zhang–Suen thinning is followed by two terminal-branch pruning passes at 30 pixels and a final 20-pixel component filter. No dilation follows thinning. See [Step 4](../主实验代码/SAREdgeNet_Step4.py) for the executable ordering and image outputs.

SAR descriptors retain relative gray-level contrast; they are not σ⁰ in dB. Smooth water often has low returns, while suitable ground–building geometry can cause bright double-bounce scattering. Dark non-water surfaces remain possible, so intensity is used together with geometry and terrain. Scene normalization does not make the same gray value physically equivalent across scenes.

## Postprocessing

| Operation | Rule used by the current entry point |
| --- | --- |
| Boundary support | Retain support near accepted contours; distance 17 pixels, component threshold 1,200 pixels |
| Constrained seed growth | Within closed retained boundaries, use a low-intensity seed; gray value ≤50 and difference from seed ≤30; four-neighbour propagation |
| Boundary refinement | Anchor-distance thresholds 20 and 10 pixels in the Step 8 caller |
| Internal-gap filling | Fill gaps ≤20 pixels directly; larger gaps require mean ≤50 **and** standard deviation ≤30 |

Call-site arguments take precedence over helper defaults. These rules can reduce false detections but can also remove genuine shoreline detail. Internal filling does not restore external branches already removed. [Stage examples and metrics](../results/postprocessing/README.md) document this tradeoff. The constrained-growth CLI is a migration implementation; see the version notes before treating it as an exact historical program.
