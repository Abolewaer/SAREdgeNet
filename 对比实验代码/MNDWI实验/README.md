# MNDWI optical comparison

Use the Int16 GEE script or the 24 sequential tile scripts with your own Earth Engine account. They export continuous MNDWI, quantized as `round(index * 10000)`, NoData −32768, on a 20 m EPSG:32646 grid. They use June–August 2022 Sentinel-2 Level-2A, cloud cover ≤40%, SCL exclusions 0/1/3/8/9/10/11, area-averaged B3 on B11's native grid and per-band temporal medians before calculating the index.

The final reported threshold is **MNDWI >0.1**, or Int16 >1000. Exporting the continuous index does not itself choose a final water threshold. Threshold exploration in the recovered script is confined to Kokoxili development data. It does not tune against external test labels.

Local scripts require explicit environment variables:

| Variable | File or directory |
| --- | --- |
| `SAREDGE_MNDWI_WORK` | External working folder with download ZIPs; generated TIFFs remain here |
| `SAREDGE_ROI` | Actual Kokoxili development raster |
| `SAREDGE_REFERENCE` | Lake_TP_2022 reference SHP |
| `SAREDGE_SAR` | Processed SAR defining valid evaluation coverage |

`合并与阈值搜索.py` extracts downloads and investigates thresholds on the development region. `全区域评价_阈值01.py` evaluates the fixed 0.1 threshold on all/common, development and external-test masks separately. These scripts perform large operations when executed; none was run in preparing this package.

The completed record used 23 of 24 optical tiles. Common valid observations covered 99.25% of SAR-valid pixels before excluding the development region. Missing optical pixels were excluded rather than assigned to land. Recorded outputs are under [results/mndwi](../../results/mndwi). An MNDWI-to-SAR performance difference is not strictly paired unless both use the same valid mask.
