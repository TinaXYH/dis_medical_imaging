# tests/test_module2.py
import unittest
import numpy as np
from numpy import fft
from medical_imaging.module2 import denoise_mri
class TestDenoiseMRI(unittest.TestCase):

    def generate_synthetic_kspace(self, image: np.ndarray, n_coils: int, coil_dim: int):
        """
        Generate synthetic k-space data for a given image and number of coils.
        Different coils are simulated by adding slight random noise to the FFT result.
        """
        # Compute FFT of the image
        base_kspace = fft.fft2(image)
        # Create an array for k-space data with additional noise for each coil
        if coil_dim == 0:
            kspace_data = np.empty((n_coils, *image.shape), dtype=np.complex128)
            for i in range(n_coils):
                kspace_data[i] = base_kspace * (1 + 0.05 * np.random.randn(*image.shape))
        elif coil_dim == 1:
            kspace_data = np.empty((image.shape[0], n_coils, image.shape[1]), dtype=np.complex128)
            for i in range(n_coils):
                kspace_data[:, i, :] = base_kspace * (1 + 0.05 * np.random.randn(*image.shape))
        elif coil_dim == 2:
            kspace_data = np.empty((image.shape[0], image.shape[1], n_coils), dtype=np.complex128)
            for i in range(n_coils):
                kspace_data[:, :, i] = base_kspace * (1 + 0.05 * np.random.randn(*image.shape))
        else:
            raise ValueError("coil_dim must be 0, 1, or 2.")
        return kspace_data

    def test_output_shape_and_type_coil0(self):
        # Create a simple synthetic image (e.g., a square in the center)
        image = np.zeros((64, 64))
        image[28:36, 28:36] = 1.0

        n_coils = 3
        kspace_data = self.generate_synthetic_kspace(image, n_coils, coil_dim=0)
        combined = denoise_mri(kspace_data, coil_dim=0)
        # Check that output shape equals image shape
        self.assertEqual(combined.shape, image.shape)
        # Check that output type is a floating point type
        self.assertTrue(np.issubdtype(combined.dtype, np.floating))

    def test_output_shape_and_type_coil1(self):
        # Use a synthetic image with different pattern
        image = np.ones((128, 128))
        image[40:88, 40:88] = 0.5

        n_coils = 4
        kspace_data = self.generate_synthetic_kspace(image, n_coils, coil_dim=1)
        combined = denoise_mri(kspace_data, coil_dim=1)
        self.assertEqual(combined.shape, image.shape)
        self.assertTrue(np.issubdtype(combined.dtype, np.floating))

    def test_output_shape_and_type_coil2(self):
        # Create another synthetic image with a circular pattern
        x, y = np.indices((64, 64))
        center = (32, 32)
        radius = 20
        image = ((x - center[0])**2 + (y - center[1])**2 < radius**2).astype(np.float64)
        
        n_coils = 2
        kspace_data = self.generate_synthetic_kspace(image, n_coils, coil_dim=2)
        combined = denoise_mri(kspace_data, coil_dim=2)
        self.assertEqual(combined.shape, image.shape)
        self.assertTrue(np.issubdtype(combined.dtype, np.floating))

    def test_invalid_coil_dim(self):
        # Test that an invalid coil_dim参数会触发异常
        image = np.zeros((64, 64))
        image[30:34, 30:34] = 1.0
        kspace_data = self.generate_synthetic_kspace(image, 2, coil_dim=0)
        with self.assertRaises(ValueError):
            # 使用不合法的 coil_dim 值，如 3
            denoise_mri(kspace_data, coil_dim=3)

if __name__ == '__main__':
    unittest.main()
