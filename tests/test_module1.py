# tests/test_module1.py
import unittest
import numpy as np
from skimage.transform import radon
from medical_imaging.module1 import correct_ct_sinogram, correct_pet_sinogram, fbp_reconstruction, os_sart_reconstruction, sirt_reconstruction, osem_reconstruction, mlem_reconstruction

class TestModule1(unittest.TestCase):

    def test_ct_correction(self):
        ct_sino = np.full((10, 10), 100.0)
        ct_dark = np.full((10, 10), 10.0)
        ct_flat = np.full((10, 10), 110.0)
        corrected = correct_ct_sinogram(ct_sino, ct_dark, ct_flat)
        expected = (100 - 10) / (110 - 10)
        self.assertTrue(np.allclose(corrected, expected))

    def test_pet_correction(self):
        pet_sino = np.full((10, 10), 50.0)
        pet_calib = np.full((10, 10), 5.0)
        corrected = correct_pet_sinogram(pet_sino, pet_calib)
        expected = 50.0 / 5.0
        self.assertTrue(np.allclose(corrected, expected))
    
    def setUp(self):
        # Create a simple phantom image for reconstruction testing
        x, y = np.indices((64, 64))
        center = (32, 32)
        radius = 20
        self.phantom = ((x - center[0])**2 + (y - center[1])**2 < radius**2).astype(np.float64)
        self.theta = np.linspace(0., 180., max(self.phantom.shape), endpoint=False)
        self.sinogram = radon(self.phantom, theta=self.theta, circle=True)
    
    def test_fbp_reconstruction(self):
        recon = fbp_reconstruction(self.sinogram, self.theta)
        self.assertEqual(recon.shape, self.phantom.shape)
    
    def test_os_sart_reconstruction(self):
        recon = os_sart_reconstruction(self.sinogram, self.theta, gamma=0.1, max_iter=5, num_subsets=5)
        self.assertEqual(recon.shape, self.phantom.shape)
    
    def test_sirt_reconstruction(self):
        recon = sirt_reconstruction(self.sinogram, self.theta, gamma=0.01, max_iter=5)
        self.assertEqual(recon.shape, self.phantom.shape)
        
    def test_osem_reconstruction(self):
        recon = osem_reconstruction(self.sinogram, self.theta, gamma=0.1, max_iter=5, num_subsets=5)
        self.assertEqual(recon.shape, self.phantom.shape)
    
    def test_mlem_reconstruction(self):
        recon = mlem_reconstruction(self.sinogram, self.theta, gamma=0.1, max_iter=5)
        self.assertEqual(recon.shape, self.phantom.shape)

if __name__ == '__main__':
    unittest.main()
