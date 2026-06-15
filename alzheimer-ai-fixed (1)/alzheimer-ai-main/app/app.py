
import os
import json
import time
import tempfile
import hashlib
import pickle
import urllib.request
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import shap

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="AI Alzheimer Detection",
    page_icon="🧠",
    layout="wide"
)

# BF-08 — mandatory permanent disclaimer
st.error(
    "⚠️ **RESEARCH USE ONLY** · This system does **NOT** replace a medical diagnosis. "
    "Any clinical decision must be made by a qualified healthcare professional. "
    "Digital Health Engineering PFA 2024-2025."
)

st.title("🧠 AI System — Early Detection of Alzheimer's Disease")
st.markdown(
    "*OASIS-2 · ResNet50 · Multimodal Fusion · MRI + MMSE + nWBV + Age + EDUC · Grad-CAM · SHAP*"
)
st.divider()


# ============================================================
# Paths
# ============================================================
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

MODEL_PATH = MODELS_DIR / "multimodal_final.h5"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.npy"
SHAP_REFERENCE_IMAGE_PATH = MODELS_DIR / "shap_reference_image.npy"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"
TABULAR_STATS_PATH = MODELS_DIR / "tabular_stats.json"

CLASSES = [
    "CN — Cognitively Normal",
    "MCI — Mild Cognitive Impairment",
    "MA — Alzheimer Disease"
]
CLS_KEYS = ["CN", "MCI", "MA"]

COLORS = {
    0: "#2E7D32",   # CN green
    1: "#E65100",   # MCI orange
    2: "#C62828",   # MA red
}

HIPPO_X = 80
HIPPO_Y = 130
HIPPO_W = 64
HIPPO_H = 40
GRADCAM_LAYER = "conv5_block3_3_conv"


# ============================================================
# Asset integrity helpers
# ============================================================
def is_lfs_pointer(path: Path) -> bool:
    """
    True if `path` is a Git LFS pointer stub rather than the real binary.
    LFS stubs are tiny text files beginning with the spec header.
    """
    try:
        if path.stat().st_size > 1024:
            return False
        with open(path, "rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com")
    except Exception:
        return False


def file_is_usable(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024 and not is_lfs_pointer(path)


def _resolve_url(secret_key: str):
    """Look up a download URL from Streamlit secrets first, then env vars."""
    try:
        if secret_key in st.secrets:
            return st.secrets[secret_key]
    except Exception:
        pass
    return os.environ.get(secret_key)


def ensure_asset(path: Path, secret_key: str, label: str) -> bool:
    """
    Guarantee a large binary asset is present and real (not an LFS stub).
    If missing/stub and a URL is configured via `secret_key`, download it once.
    Returns True only if a usable file is present afterwards.
    """
    if file_is_usable(path):
        return True

    url = _resolve_url(secret_key)
    if not url:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with st.spinner(f"Downloading {label} (first run only, please wait)…"):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        except Exception as e:
            try:
                tmp.unlink()
            except Exception:
                pass
            st.error(f"Could not download {label}: {e}")
            return False

    return file_is_usable(path)


def require_model():
    """
    Startup gate: ensure the trained model is genuinely available before the
    rest of the app runs. Fails with a clear, actionable message instead of an
    opaque HDF5 traceback when only the LFS pointer is present.
    """
    if ensure_asset(MODEL_PATH, "MODEL_URL", "model weights (multimodal_final.h5)"):
        return

    if is_lfs_pointer(MODEL_PATH):
        detail = (
            "`multimodal_final.h5` is a **Git LFS pointer**, not the real model. "
            "The 226 MB weights file was not pulled into this deployment."
        )
    elif not MODEL_PATH.exists():
        detail = "`multimodal_final.h5` is **missing** from `app/models/`."
    else:
        detail = "`multimodal_final.h5` is present but appears truncated or invalid."

    st.error(
        f"⛔ **Model not available.** {detail}\n\n"
        "**Fix one of the following:**\n"
        "1. Set a `MODEL_URL` secret (Streamlit *Settings → Secrets*) or environment "
        "variable pointing to the real `.h5`; it will be downloaded automatically on first run.\n"
        "2. Or run `git lfs pull` and redeploy so the full-size file ships with the repo.\n\n"
        "The app cannot make predictions until the trained weights are restored."
    )
    st.stop()


# ============================================================
# Cached resources
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_tabular_stats():
    if not TABULAR_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing tabular stats file: {TABULAR_STATS_PATH}")
    with open(TABULAR_STATS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_shap_background_and_reference():
    if not SHAP_BACKGROUND_PATH.exists():
        raise FileNotFoundError(f"Missing SHAP background: {SHAP_BACKGROUND_PATH}")
    if not SHAP_REFERENCE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing SHAP reference image: {SHAP_REFERENCE_IMAGE_PATH}")

    background = np.load(SHAP_BACKGROUND_PATH).astype(np.float32)
    reference_img_raw = np.load(SHAP_REFERENCE_IMAGE_PATH).astype(np.float32)

    return background, reference_img_raw


require_model()
model = load_model()
tabular_stats = load_tabular_stats()


# ============================================================
# Preprocessing helpers
# ============================================================
def prepare_resnet_batch(x_raw):
    """
    Model was trained with ResNet50 preprocess_input applied outside the model.
    Input x_raw must be [0,1], shape (N,224,224,3).
    """
    return preprocess_input(x_raw.astype(np.float32) * 255.0)


def normalize_slice(slice_2d):
    """
    Robust intensity normalization to [0,1].
    """
    s = np.asarray(slice_2d, dtype=np.float32)

    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s, dtype=np.float32)

    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    nonzero = s[s > 0]
    if nonzero.size > 20:
        lo, hi = np.percentile(nonzero, [1, 99])
    else:
        lo, hi = np.min(s), np.max(s)

    s = np.clip(s, lo, hi)
    s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-8)

    return s.astype(np.float32)


def preprocess_nifti_file(uploaded_file):
    """
    BF-01: MRI upload .nii or .nii.gz.
    Returns:
      img_rgb: (224,224,3) float32 in [0,1]
      info: metadata dictionary
    """
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # no patient data is kept; file is deleted after reading
        with open(tmp_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:12]

        nii = nib.load(tmp_path)
        data = nii.get_fdata()

        data = np.squeeze(data)

        if data.ndim == 4:
            data = data[..., 0]

        if data.ndim == 2:
            slice_2d = data
            slice_index = None
        elif data.ndim == 3:
            slice_index = data.shape[2] // 2
            slice_2d = data[:, :, slice_index]
        else:
            raise ValueError(f"Unsupported MRI dimensions: {data.shape}")

        slice_norm = normalize_slice(slice_2d)

        img_2d = np.array(
            Image.fromarray((slice_norm * 255).astype(np.uint8)).resize(
                (224, 224),
                Image.LANCZOS
            ),
            dtype=np.float32
        ) / 255.0

        img_rgb = np.stack([img_2d, img_2d, img_2d], axis=-1).astype(np.float32)

        info = {
            "original_shape": tuple(data.shape),
            "slice_index": slice_index,
            "file_hash": file_hash
        }

        return img_rgb, info

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def build_tabular_vector(mmse, nwbv, age, educ):
    """
    Feature vector expected by the model:
    [MMSE_norm, nWBV, Age_zscore, EDUC_zscore]
    """
    mmse_norm = float(mmse) / 30.0

    age_z = (float(age) - float(tabular_stats["age_mean"])) / (float(tabular_stats["age_std"]) + 1e-8)
    educ_z = (float(educ) - float(tabular_stats["educ_mean"])) / (float(tabular_stats["educ_std"]) + 1e-8)

    return np.array([[mmse_norm, float(nwbv), age_z, educ_z]], dtype=np.float32)


def clinical_summary(pred_cls, proba):
    """
    BF-07: plain-language explanation, no AI jargon.
    """
    confidence = float(proba[pred_cls])

    if pred_cls == 0:
        msg = (
            "The model output is closest to the cognitively normal group. "
            "This does not prove absence of disease; it only means the uploaded data resembles the CN examples in this research dataset."
        )
    elif pred_cls == 1:
        msg = (
            "The model output suggests a possible mild cognitive impairment pattern. "
            "This is a screening-style signal only and should be reviewed by a qualified clinician."
        )
    else:
        msg = (
            "The model output is closest to the Alzheimer disease group in the research dataset. "
            "This is not a diagnosis and must not be used without clinical evaluation."
        )

    return f"{msg}\n\nModel confidence for selected class: {confidence:.1%}."


# ============================================================
# Grad-CAM helpers
# ============================================================
def resize_heatmap_to_224(heatmap):
    h = np.array(heatmap, dtype=np.float32)
    h = np.squeeze(h)

    if h.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {h.shape}")

    h_tf = tf.convert_to_tensor(h[None, :, :, None], dtype=tf.float32)
    h_resized = tf.image.resize(h_tf, (224, 224), method="bilinear")
    h_resized = h_resized.numpy()[0, :, :, 0]

    h_resized = h_resized - np.min(h_resized)
    h_resized = h_resized / (np.max(h_resized) + 1e-8)

    return h_resized.astype(np.float32)


@st.cache_resource
def build_gradcam_model():
    """
    Build Grad-CAM model for nested ResNet50 branch.
    """
    backbone = model.get_layer("resnet50")
    target_layer = backbone.get_layer(GRADCAM_LAYER)

    conv_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output]
    )

    img_input = model.inputs[0]
    tab_input = model.inputs[1]

    conv_output, backbone_output = conv_model(img_input)
    cnn_vec = model.get_layer("cnn_gap")(backbone_output)

    tab = tab_input
    for lname in ["tab_dense_128", "tab_bn_128", "tab_dropout_128", "tab_dense_64", "tab_bn_64"]:
        if lname in [l.name for l in model.layers]:
            tab = model.get_layer(lname)(tab)

    merged = model.get_layer("fusion")([cnn_vec, tab])

    layers_list = list(model.layers)
    fusion_index = layers_list.index(model.get_layer("fusion"))

    x = merged
    for layer in layers_list[fusion_index + 1:]:
        x = layer(x)

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_output, x]
    )


def generate_gradcam(img_raw, tabular, class_idx):
    """
    BF-05: automatic Grad-CAM with hippocampal annotation.
    Uses validated fallback Grad-CAM because Grad-CAM++ failed on nested multimodal model.
    """
    grad_model = build_gradcam_model()
    img_pp = prepare_resnet_batch(img_raw)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            [img_pp, tabular.astype(np.float32)],
            training=False
        )
        loss = predictions[:, int(class_idx)]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = resize_heatmap_to_224(heatmap.numpy())

    return heatmap


def plot_gradcam_overlay(img_raw_single, heatmap, pred_cls):
    overlay = np.clip(
        0.55 * img_raw_single + 0.45 * cm.jet(heatmap)[:, :, :3],
        0,
        1
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)

    rect = patches.Rectangle(
        (HIPPO_X, HIPPO_Y),
        HIPPO_W,
        HIPPO_H,
        linewidth=2,
        edgecolor="yellow",
        facecolor="none",
        linestyle="--"
    )

    ax.add_patch(rect)
    ax.text(
        HIPPO_X,
        HIPPO_Y - 5,
        "Hippocampus ROI",
        color="yellow",
        fontsize=10,
        fontweight="bold",
        backgroundcolor="black"
    )

    ax.set_title(
        f"Grad-CAM — Prediction: {CLS_KEYS[pred_cls]}",
        color=COLORS[pred_cls],
        fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()

    return fig


# ============================================================
# SHAP helpers
# ============================================================
@st.cache_resource
def load_or_create_shap_explainer():
    """
    BF-06: real SHAP KernelExplainer.
    Tries to load Day 5 precomputed explainer. If not portable, recreates KernelExplainer
    using saved background/reference image.
    """
    background, reference_img_raw = load_shap_background_and_reference()
    reference_img_pp = prepare_resnet_batch(reference_img_raw)

    def predict_tabular_for_shap(tabular_batch):
        tabular_batch = np.array(tabular_batch, dtype=np.float32)
        imgs = np.repeat(reference_img_pp, repeats=len(tabular_batch), axis=0)
        return model.predict([imgs, tabular_batch], verbose=0)

    if SHAP_EXPLAINER_PATH.exists() and not is_lfs_pointer(SHAP_EXPLAINER_PATH):
        try:
            with open(SHAP_EXPLAINER_PATH, "rb") as f:
                explainer = pickle.load(f)

            # Quick compatibility test
            _ = explainer.shap_values(background[:1], nsamples=10)
            return explainer, background

        except Exception:
            # Recreate safely if pickled object is not portable
            pass

    explainer = shap.KernelExplainer(
        predict_tabular_for_shap,
        background
    )

    return explainer, background


def compute_shap_values_for_instance(tabular_vector, pred_cls):
    explainer, _ = load_or_create_shap_explainer()

    shap_values = explainer.shap_values(
        tabular_vector.astype(np.float32),
        nsamples=50
    )

    if isinstance(shap_values, list):
        vals = np.array(shap_values[pred_cls])[0]
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            vals = arr[0, :, pred_cls]
        elif arr.ndim == 3 and arr.shape[0] == 3:
            vals = arr[pred_cls, 0, :]
        else:
            raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

    return vals


def plot_shap_bar(shap_vals, pred_cls):
    feature_names = ["MMSE_norm", "nWBV", "Age_zscore", "EDUC_zscore"]

    fig, ax = plt.subplots(figsize=(7, 3.8))

    colors = ["#C00000" if v > 0 else "#2E75B6" for v in shap_vals]
    bars = ax.barh(feature_names, shap_vals, color=colors)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(f"SHAP Explanation — Predicted class: {CLS_KEYS[pred_cls]}")

    for bar, value in zip(bars, shap_vals):
        ax.text(
            value + (0.001 if value >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9
        )

    plt.tight_layout()

    return fig


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "**Dataset**: OASIS-2 Longitudinal MRI\n\n"
        "**Model**: ResNet50 + tabular clinical features\n\n"
        "**Classes**: CN, MCI, MA\n\n"
        "**Licence**: ODC-by — cite Marcus et al. 2010 in public use."
    )

    st.warning(
        "Validated only on the OASIS-2 population. "
        "The model did not reach all target metrics, especially MCI recall."
    )

    st.markdown("### Required app functions")
    st.markdown(
        "- MRI upload\n"
        "- Probability vector\n"
        "- Color-coded result\n"
        "- Grad-CAM hippocampus ROI\n"
        "- SHAP tabular explanation\n"
        "- Plain-language summary"
    )


# ============================================================
# UI inputs
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Input parameters")

    uploaded = st.file_uploader(
        "MRI T1 file (.nii or .nii.gz)",
        type=["nii", "gz"],
        help="Upload a NIfTI T1 MRI file. The file is read temporarily and then deleted."
    )

    mmse = st.number_input(
        "MMSE score (0–30)",
        min_value=0,
        max_value=30,
        value=25,
        step=1
    )

    nwbv = st.slider(
        "nWBV — normalized whole brain volume",
        min_value=0.60,
        max_value=0.90,
        value=0.73,
        step=0.01
    )

    age = st.slider(
        "Age",
        min_value=60,
        max_value=96,
        value=72,
        step=1
    )

    educ = st.slider(
        "Education years",
        min_value=6,
        max_value=23,
        value=12,
        step=1
    )

    run_btn = st.button("Run analysis", type="primary")


with col2:
    st.subheader("🧾 Input preview")

    if uploaded is None:
        st.info("Upload a `.nii` or `.nii.gz` MRI file to start.")
    else:
        st.success(f"Uploaded: `{uploaded.name}`")
        st.write(
            {
                "MMSE": mmse,
                "nWBV": nwbv,
                "Age": age,
                "EDUC": educ
            }
        )


# ============================================================
# Inference
# ============================================================
if run_btn:
    if uploaded is None:
        st.error("Please upload an MRI `.nii` or `.nii.gz` file first.")
        st.stop()

    if not (0 <= mmse <= 30):
        st.error("MMSE must be between 0 and 30.")
        st.stop()

    t0 = time.perf_counter()

    with st.spinner("Preprocessing MRI and running model..."):
        img_rgb, mri_info = preprocess_nifti_file(uploaded)
        img_batch_raw = img_rgb[None, :, :, :].astype(np.float32)
        img_batch_pp = prepare_resnet_batch(img_batch_raw)

        tabular = build_tabular_vector(mmse, nwbv, age, educ)

        proba = model.predict([img_batch_pp, tabular], verbose=0)[0]
        pred_cls = int(np.argmax(proba))

    st.divider()

    result_color = COLORS[pred_cls]

    st.markdown(
        f"""
        <div style="padding:18px;border-radius:12px;background-color:{result_color};color:white;">
            <h2 style="margin-bottom:0;">Prediction: {CLASSES[pred_cls]}</h2>
            <p style="font-size:18px;margin-top:6px;">Confidence: {proba[pred_cls]:.1%}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📊 Probability vector")
    prob_df = {
        "Class": CLS_KEYS,
        "Probability": [float(x) for x in proba]
    }
    st.dataframe(prob_df, use_container_width=True)

    st.bar_chart(
        {
            "CN": [float(proba[0])],
            "MCI": [float(proba[1])],
            "MA": [float(proba[2])]
        }
    )

    st.subheader("🩺 Plain-language clinical summary")
    st.info(clinical_summary(pred_cls, proba))

    st.subheader("🧠 MRI slice preview")
    fig0, ax0 = plt.subplots(figsize=(5, 5))
    ax0.imshow(img_rgb[:, :, 0], cmap="gray")
    ax0.set_title(f"Extracted median slice | Original shape: {mri_info['original_shape']}")
    ax0.axis("off")
    st.pyplot(fig0)
    plt.close(fig0)

    # BF-05: automatic Grad-CAM after prediction
    st.subheader("🔥 Grad-CAM with hippocampal annotation")
    with st.spinner("Generating Grad-CAM heatmap..."):
        try:
            heatmap = generate_gradcam(img_batch_raw, tabular, pred_cls)
            fig_cam = plot_gradcam_overlay(img_rgb, heatmap, pred_cls)
            st.pyplot(fig_cam)
            plt.close(fig_cam)
            st.caption(
                "Hippocampal ROI is approximate and based on the 2D extracted MRI slice. "
                "This is an anatomical sanity check, not clinical localization proof."
            )
        except Exception as e:
            st.warning(f"Grad-CAM unavailable: {e}")

    # BF-06: automatic SHAP after prediction
    st.subheader("📌 SHAP tabular explanation")
    with st.spinner("Computing SHAP values..."):
        try:
            shap_vals = compute_shap_values_for_instance(tabular, pred_cls)
            fig_shap = plot_shap_bar(shap_vals, pred_cls)
            st.pyplot(fig_shap)
            plt.close(fig_shap)
            st.caption(
                "SHAP explains the four tabular inputs while the MRI input is fixed to the saved reference image."
            )
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

    elapsed = time.perf_counter() - t0

    st.caption(f"⏱️ Total inference time: {elapsed:.1f} seconds")

    if elapsed > 30:
        st.warning("Inference exceeded 30 seconds. This may fail the performance requirement.")
    else:
        st.success("Inference completed under 30 seconds.")

    st.caption(f"Temporary file hash: {mri_info['file_hash']} — file not stored.")
else:
    st.info("After uploading an MRI and setting the clinical fields, press **Run analysis**.")
