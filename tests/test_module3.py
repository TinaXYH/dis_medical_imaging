# tests/test_module3.py
import unittest
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from medical_imaging.module3 import segmentation, radiomics

class TestModule3Segmentation(unittest.TestCase):
    def setUp(self):
        # Create synthetic 3D scan and mask arrays
        self.scan = np.zeros((50, 50, 50))
        self.mask = np.zeros((50, 50, 50))
        # Create a spherical segmentation in the center
        x, y, z = np.indices((50, 50, 50))
        center = (25, 25, 25)
        radius = 10
        sphere = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 < radius**2
        self.mask[sphere] = 1
        self.scan[sphere] = 100
        
        # Patch nib.load to return synthetic images
        self.original_nib_load = nib.load
        self.nifti_img = nib.Nifti1Image(self.scan, affine=np.eye(4))
        self.nifti_mask = nib.Nifti1Image(self.mask, affine=np.eye(4))
        nib.load = lambda f: self.nifti_img if "mask" not in f else self.nifti_mask
    
    def tearDown(self):
        nib.load = self.original_nib_load
    
    def test_compute_metrics(self):
        pred = self.mask.astype(bool)
        gt = self.mask.astype(bool)
        metrics = segmentation.compute_metrics(pred, gt)
        self.assertAlmostEqual(metrics["dice"], 1.0, places=4)
    
    def test_process_case(self):
        result = segmentation.process_case("case_0", "dummy_folder")
        self.assertIsNotNone(result)
        self.assertIn("metrics", result)
    
    def test_draw_slice_bounding_box(self):
        fig, ax = plt.subplots()
        test_slice = self.mask[:, :, 25]
        segmentation.draw_slice_bounding_box(ax, test_slice, color='r')
        plt.close(fig)

class TestModule3Radiomics(unittest.TestCase):
    def setUp(self):
        # Create a synthetic ROI
        self.roi = np.ones((20, 20, 20)) * 50
        self.roi[5:15, 5:15, 5:15] = 100
    
    def test_compute_energy(self):
        energy = radiomics.compute_energy(self.roi.flatten())
        self.assertGreater(energy, 0)
    
    def test_compute_mad(self):
        mad = radiomics.compute_mad(self.roi.flatten())
        self.assertGreaterEqual(mad, 0)
    
    def test_compute_uniformity(self):
        uniformity = radiomics.compute_uniformity(self.roi.flatten(), num_bins=10, fixed_range=(0, 200))
        self.assertGreaterEqual(uniformity, 0)
    
    def test_compute_radiomic_features(self):
        features = radiomics.compute_radiomic_features(self.roi, num_bins=10, fixed_range=(0, 200))
        self.assertIn("energy", features)
        self.assertIn("mad", features)
        self.assertIn("uniformity", features)
    
    def test_case_str_to_int(self):
        self.assertEqual(radiomics.case_str_to_int("case_0"), 0)
        self.assertEqual(radiomics.case_str_to_int("case_10"), 10)

if __name__ == '__main__':
    unittest.main()
