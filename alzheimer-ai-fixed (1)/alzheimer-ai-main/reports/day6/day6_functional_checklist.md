# Day 6 Functional Checklist — Streamlit App

> **Status legend:** [x] = verified in code / present · [~] = code present, runtime not yet
> verified (needs the real model + a live run) · [ ] = not done / blocked.
>
> **BLOCKER:** `app/models/multimodal_final.h5` and `shap_explainer.pkl` are Git LFS
> *pointer stubs*, not the real binaries. The app now detects this and either downloads
> the model from a configured `MODEL_URL` or stops with a clear message. Predictions are
> impossible until the real 226 MB weights are restored (see README → Deployment).

## L3 Streamlit App
- [~] Streamlit app runs locally (blocked until real model is present)
- [ ] App deploys with public URL (not yet re-verified)
- [x] `app/app.py` is the entrypoint
- [x] `requirements.txt` exists
- [ ] Final model available at `app/models/multimodal_final.h5` (LFS stub only — BLOCKER)

## BF-01 MRI upload
- [~] `.nii` upload works (code present)
- [~] `.nii.gz` upload works (code present)
- [x] Uploaded MRI is read from a temporary file and deleted after preprocessing (code present)

## BF-02 MMSE validation
- [x] MMSE input restricted to 0–30 (number_input min=0, max=30)

## BF-03 Probability vector
- [x] P(CN), P(MCI), P(MA) displayed

## BF-04 Color coding
- [x] CN = green · MCI = orange · MA = red

## BF-05 Grad-CAM
- [x] Grad-CAM is generated automatically after prediction
- [x] Hippocampal ROI annotation is visible
- [x] App states that the ROI is approximate and research-only

## BF-06 SHAP
- [x] Real SHAP KernelExplainer is used (with safe rebuild fallback; LFS stub skipped)
- [x] SHAP runs automatically after prediction
- [x] No placeholder SHAP values

## BF-07 Plain-language summary
- [x] Clinical summary is understandable without AI jargon
- [x] Output does not claim diagnosis

## BF-08 Medical disclaimer
- [x] Research-use-only disclaimer is permanently visible at top of page

## BNF Performance
- [x] `scripts/benchmark_performance.py` is present and imports cleanly
- [~] p95 inference time < 30 s (needs a live run with the real model)

## BNF Availability
- [ ] Streamlit public URL tested
- [ ] UptimeRobot keep-alive configured
- [~] Local backup demo ready with `streamlit run app/app.py` (once model restored)

## Known limitations to disclose
- [x] Model did not meet all target metrics (accuracy 0.56, AUC 0.56, MCI recall 0.20)
- [x] MCI recall remains weak
- [x] Converted/MCI label ambiguity documented
- [x] SHAP top-3 result documented honestly
