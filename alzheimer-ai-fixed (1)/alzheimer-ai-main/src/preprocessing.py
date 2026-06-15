"""
MRI preprocessing pipeline for OASIS-2.
Deliverable L2 — Digital Health Engineering PFA 2024–2025.

Steps:
1. Load NIfTI / Analyze MRI file
2. Skull stripping using nilearn brain mask
3. Extract median axial slice
4. Normalize intensity to [0, 1]
5. Resize to 224x224
6. Convert to 3 channels for ResNet50
"""

import os
from typing import Tuple

import nibabel as nib
import numpy as np
from PIL import Image
import nilearn.image as nli
from nilearn.masking import compute_brain_mask


def skull_strip(nii_path: str) -> np.ndarray:
    """
    Remove skull using nilearn brain mask.
    Falls back to intensity thresholding if nilearn fails.
    """
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    data = np.squeeze(data)

    if data.size == 0 or np.max(data) - np.min(data) < 1e-8:
        return np.zeros_like(data, dtype=np.float32)

    non_zero = data[data > 0]

    if non_zero.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    try:
        brain_mask = compute_brain_mask(img, threshold=0.5, connected=True)
        brain_img = nli.math_img("img * mask", img=img, mask=brain_mask)
        brain_data = brain_img.get_fdata().astype(np.float32)
        brain_data = np.squeeze(brain_data)

        if brain_data.size == 0 or np.max(brain_data) - np.min(brain_data) < 1e-8:
            return np.zeros_like(brain_data, dtype=np.float32)

        return brain_data

    except Exception:
        threshold = np.percentile(non_zero, 15)
        return (data * (data > threshold)).astype(np.float32)


def load_nifti_slice(nii_path: str, axis: int = 2) -> np.ndarray:
    """
    Load MRI file, skull-strip, extract median slice, normalize to [0, 1].
    Returns 2D float32 array.
    """
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"MRI file not found: {nii_path}")

    data = skull_strip(nii_path)
    data = np.squeeze(data)

    if data.ndim == 2:
        slice_2d = data

    elif data.ndim == 3:
        mid = data.shape[axis] // 2

        if axis == 2:
            slice_2d = data[:, :, mid]
        elif axis == 1:
            slice_2d = data[:, mid, :]
        else:
            slice_2d = data[mid, :, :]

    else:
        raise ValueError(f"2D or 3D MRI volume expected, got shape {data.shape}")

    slice_2d = np.squeeze(slice_2d)

    if slice_2d.ndim != 2:
        raise ValueError(f"2D slice expected after squeeze, got shape {slice_2d.shape}")

    vmin = slice_2d.min()
    vmax = slice_2d.max()

    if vmax - vmin < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.float32)

    normalized = (slice_2d - vmin) / (vmax - vmin + 1e-8)

    return normalized.astype(np.float32)


def resize_and_encode(
    slice_2d: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Resize to 224x224 and replicate to 3 channels for ResNet50.
    Returns float32 array with shape (224, 224, 3).
    """
    slice_2d = np.squeeze(slice_2d)

    if slice_2d.ndim != 2:
        raise ValueError(f"2D array expected, got shape {slice_2d.shape}")

    image = Image.fromarray((slice_2d * 255).astype(np.uint8))
    image = image.resize(target_size, Image.LANCZOS)

    arr = np.array(image, dtype=np.float32) / 255.0

    return np.stack([arr, arr, arr], axis=-1).astype(np.float32)


def preprocess_mri(nii_path: str, target_size=(224, 224)) -> np.ndarray:
    """
    Full pipeline: MRI path → (224, 224, 3) float32 array.
    """
    slice_2d = load_nifti_slice(nii_path)
    return resize_and_encode(slice_2d, target_size)
