# Day 5 Explainability Report — Grad-CAM++ and SHAP

## Purpose
This report documents model explainability for the multimodal Alzheimer's classification model.
It includes Grad-CAM++/Grad-CAM heatmaps with hippocampal ROI annotation and SHAP top-3 tabular feature analysis.

## Important performance context
The model does not meet the target performance metrics. Explainability therefore describes the current learned behaviour, not a clinically validated diagnostic system.

## Grad-CAM anatomical validation
- Number of validated cases: 20
- Best layer used: conv5_block3_3_conv
- Mean hippocampal activation ratio: 1.843
- Median hippocampal activation ratio: 1.387
- Cases with ratio >= 1.2: 13
- Cases with ratio < 1.2: 7

The hippocampal ROI is approximate and based on the 2D extracted MRI slice. Because the project uses single-slice MRI rather than full 3D anatomical registration, this should be treated as a qualitative anatomical sanity check, not clinical localization proof.

## SHAP top-3 tabular features
- EDUC_zscore: mean |SHAP| = 0.003534
- Age_zscore: mean |SHAP| = 0.002365
- MMSE_norm: mean |SHAP| = 0.000339

SHAP was computed on the four tabular features: MMSE_norm, nWBV, Age_zscore, and EDUC_zscore. The MRI input was fixed to a reference image to isolate tabular feature influence.

## MCI label issue
The MCI class is based on the OASIS Converted group. Several MCI-labeled visits have CDR=0.0 and MMSE=30, making them clinically similar to CN at that visit. This makes the task partly future-conversion prediction.

## Day 4 summary
```text
Day 4 — Multimodal Model Summary
================================

Architecture: late-fusion CNN + tabular model.
MRI branch: ResNet50 backbone loaded from Day 3 backbone weights only.
Tabular branch: MMSE_norm, nWBV, Age_zscore, EDUC_zscore.
Imbalance mitigation: real-sample oversampling of aligned MRI+tabular MCI samples, plus soft class weighting.

Final reporting mode: standard argmax predictions.
Reason: validation-based MCI threshold tuning collapsed into predicting all samples as MCI, giving MCI recall=1.0 but specificity=0.0 and accuracy=0.0847. Therefore threshold predictions are diagnostic only.

Multimodal argmax metrics:
- Accuracy: 0.5593 | target: 0.85 | passed: False
- Recall macro: 0.4482 | target: 0.88 | passed: False
- F1 macro: 0.4665 | target: 0.85 | passed: False
- AUC-ROC: 0.5630 | target: 0.90 | passed: False
- Recall MCI: 0.2000 | target: 0.85 | passed: False
- Specificity: 0.7097 | target: 0.82 | passed: False

Threshold-adjusted diagnostic metrics:
- Accuracy: 0.0847 | target: 0.85 | passed: False
- Recall macro: 0.3333 | target: 0.88 | passed: False
- F1 macro: 0.0521 | target: 0.85 | passed: False
- AUC-ROC: 0.5630 | target: 0.90 | passed: False
- Recall MCI: 1.0000 | target: 0.85 | passed: True
- Specificity: 0.0000 | target: 0.82 | passed: False

MCI label issue:
The MCI test cases are from the OASIS Converted group. Several have CDR=0.0 and MMSE=30, making them clinically similar to CN at that visit. This indicates that the task is partly future-conversion prediction, not pure current-state classification.

Conclusion:
Day 4 multimodal fusion improved MCI recall compared with the Day 3 CNN-only baseline, but performance remains below the specification targets. The poor MCI performance is likely driven by small MCI sample size, Converted-label ambiguity, and weak single-slice MRI representation.

```

## MCI diagnostic details
```text
MCI Label Diagnostic Analysis
=============================

The test set contains 5 MCI-labeled samples.
These samples are labeled MCI because they belong to the OASIS 'Converted' group.
However, several Converted visits have CDR=0.0 and MMSE=30, making them clinically similar to CN at that visit.

Observed MCI test cases:
- OAS2_0031_MR1: CDR=0.0, MMSE=30.0, nWBV=0.718, predicted=CN, P_CN=0.430, P_MCI=0.278, P_MA=0.292
- OAS2_0031_MR2: CDR=0.0, MMSE=30.0, nWBV=0.719, predicted=CN, P_CN=0.366, P_MCI=0.353, P_MA=0.280
- OAS2_0031_MR3: CDR=0.5, MMSE=28.0, nWBV=0.696, predicted=MCI, P_CN=0.326, P_MCI=0.357, P_MA=0.316
- OAS2_0144_MR1: CDR=0.0, MMSE=30.0, nWBV=0.716, predicted=MA, P_CN=0.378, P_MCI=0.126, P_MA=0.496
- OAS2_0144_MR2: CDR=0.5, MMSE=30.0, nWBV=0.708, predicted=CN, P_CN=0.438, P_MCI=0.160, P_MA=0.402

Conclusion:
The poor MCI recall is not only a model failure. It is partly caused by the longitudinal Converted label: some visits labeled as MCI appear clinically normal at that visit. This makes the task closer to predicting future conversion rather than current diagnosis.

```
