import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from src.preprocessing import load_nifti_slice, resize_and_encode, preprocess_mri


def make_fake_nii(shape=(91, 109, 91), tmpdir=None):
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = os.path.join(tmpdir or tempfile.gettempdir(), "test.nii")
    nib.save(img, path)
    return path


class TestLoadNiftiSlice:

    def test_output_is_2d(self, tmp_path):
        result = load_nifti_slice(make_fake_nii(tmpdir=str(tmp_path)))
        assert result.ndim == 2

    def test_normalized_0_1(self, tmp_path):
        result = load_nifti_slice(make_fake_nii(tmpdir=str(tmp_path)))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_float32(self, tmp_path):
        result = load_nifti_slice(make_fake_nii(tmpdir=str(tmp_path)))
        assert result.dtype == np.float32

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_nifti_slice("/nonexistent.nii")

    def test_constant_volume_returns_zeros(self, tmp_path):
        img = nib.Nifti1Image(np.ones((60, 60, 60), dtype=np.float32), np.eye(4))
        path = str(tmp_path / "const.nii")
        nib.save(img, path)

        result = load_nifti_slice(path)

        assert np.all(result == 0.0)


class TestResizeAndEncode:

    def test_shape_224(self):
        result = resize_and_encode(np.random.rand(91, 109).astype(np.float32))
        assert result.shape == (224, 224, 3)

    def test_three_channels(self):
        result = resize_and_encode(np.random.rand(100, 100).astype(np.float32))
        assert result.shape[2] == 3

    def test_channels_identical(self):
        result = resize_and_encode(np.random.rand(100, 100).astype(np.float32))
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_values_in_0_1(self):
        result = resize_and_encode(np.random.rand(100, 100).astype(np.float32))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            resize_and_encode(np.random.rand(10, 10, 3))


class TestPreprocessMri:

    def test_full_pipeline(self, tmp_path):
        result = preprocess_mri(make_fake_nii(tmpdir=str(tmp_path)))

        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
