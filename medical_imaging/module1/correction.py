# medical_imaging/module1/correction.py
import numpy as np

def correct_ct_sinogram(ct_sino: np.ndarray, ct_dark: np.ndarray, ct_flat: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Correct the CT sinogram using dark and flat field measurements.
    
    Correction is performed using:
        corrected = (ct_sino - ct_dark) / max(ct_flat - ct_dark, epsilon)
    
    Parameters
    ----------
    ct_sino : np.ndarray
        Original CT sinogram.
    ct_dark : np.ndarray
        Dark field measurement.
    ct_flat : np.ndarray
        Flat field measurement (already contains I0).
    epsilon : float, optional
        Small value to avoid division by zero.
    
    Returns
    -------
    np.ndarray
        Corrected CT sinogram.
    """
    corrected = (ct_sino - ct_dark) / np.maximum(ct_flat - ct_dark, epsilon)
    return corrected

def correct_pet_sinogram(pet_sino: np.ndarray, pet_calibration: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Correct the PET sinogram using the calibration sinogram.
    
    Correction is performed using:
        corrected = pet_sino / max(pet_calibration, epsilon)
    
    Parameters
    ----------
    pet_sino : np.ndarray
        Original PET sinogram.
    pet_calibration : np.ndarray
        PET calibration sinogram.
    epsilon : float, optional
        Small value to avoid division by zero.
    
    Returns
    -------
    np.ndarray
        Corrected PET sinogram.
    """
    corrected = pet_sino / np.maximum(pet_calibration, epsilon)
    return corrected
