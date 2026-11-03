# Candidate-parameter visual comparisons

[five_level_comparison.png](five_level_comparison.png) shows five settings each for the clip coefficient, DoG small scale, DoG large scale, response threshold and joint component-size multiplier. [parameter_records.csv](parameter_records.csv) preserves settings and local foreground counts; [scope.json](scope.json) records limitations.

Within this window and parameter range, clipping had less effect than DoG scale, response threshold and component filtering. Component thresholds 200/200/500 were scaled together, so the comparison does not isolate each filtering pass.

Outputs stop after the third component filter, before thinning and classification. More candidate pixels are not automatically better: extra texture may be rejected later, while omitted shoreline candidates may be unrecoverable. These images do not establish end-to-end sensitivity, a global optimum or robustness across all SAR scenes.

Code: [visual_check.py](../../analysis/sensitivity/visual_check.py). It uses fixed rectangular-region statistics and a buffered local window rather than reconstructing a historical whole-plateau run.
