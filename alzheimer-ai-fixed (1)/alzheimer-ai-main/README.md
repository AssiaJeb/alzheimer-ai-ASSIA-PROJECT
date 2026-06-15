# AI Early Detection of Alzheimer's Disease

Digital Health Engineering PFA 2024–2025.

Dataset: OASIS-2 Longitudinal  
Model: ResNet50 + multimodal fusion  
Inputs: MRI + MMSE + nWBV + Age + EDUC  
Outputs: CN / MCI / MA classification

Research use only. This project does not replace medical diagnosis.

## Deliverables

- L1: Trained model with metrics and ROC curves
- L2: Reproducible preprocessing pipeline with tests
- L3: Streamlit web app
- L4: Explainability report with Grad-CAM and SHAP
- L5: Ablation study
- L6: Technical report
- L7: Versioned GitHub repo

## Licence / Citation

OASIS-2 data must be cited according to its ODC-by licence.
Marcus et al. (2010) must be cited in public outputs.

## Deployment

### Required: restore the trained model

`app/models/multimodal_final.h5` is tracked by Git LFS. A plain `git clone` or a
downloaded ZIP contains only a small **pointer stub**, not the 226 MB weights, and
the app cannot predict without the real file. Pick one path:

1. **Download at startup (recommended for hosted deploys).** Host the real `.h5`
   somewhere with a direct-download link (Hugging Face Hub `resolve` URL, S3/GCS
   signed URL, or a GitHub Release asset) and set a `MODEL_URL` secret. On first run
   the app downloads and caches it automatically. See `.streamlit/secrets.toml.example`.
2. **Ship the full file.** Run `git lfs install && git lfs pull` so the full-size
   `.h5` is present, then deploy. Note: Streamlit Community Cloud LFS bandwidth is
   limited and can fail on files this large — option 1 is more reliable.
3. **Local run.** `MODEL_URL="https://.../multimodal_final.h5" python scripts/download_model.py`,
   then `streamlit run app/app.py`.

If the model is missing or is still an LFS stub, the app now stops with a clear
on-screen message instead of crashing.

### Performance note (important, not a bug)

This model did **not** meet its target metrics: accuracy ≈ 0.56, AUC-ROC ≈ 0.56,
MCI recall ≈ 0.20 on the OASIS-2 test split — close to chance on a 3-class task.
This is a training/data limitation, not something the app code can fix. The
research-use-only framing and no-diagnosis disclaimers exist precisely because of
this; keep them prominent in any public deployment.
