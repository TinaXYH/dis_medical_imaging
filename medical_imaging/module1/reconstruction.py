# medical_imaging/module1/reconstruction.py
import numpy as np
from skimage.transform import radon, iradon, iradon_sart
import logging

logger = logging.getLogger(__name__)

def fbp_reconstruction(sinogram: np.ndarray, theta: np.ndarray, filter_name: str = "ramp", circle: bool = True) -> np.ndarray:
    """
    Perform Filtered Back Projection (FBP) reconstruction using the iradon transform.
    
    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram.
    theta : np.ndarray
        Array of projection angles (in degrees).
    filter_name : str, optional
        Filter to use in reconstruction (default "ramp").
    circle : bool, optional
        Whether to assume a circular reconstruction.
    
    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    reconstruction = iradon(sinogram, theta=theta, filter_name=filter_name, circle=circle)
    return reconstruction

def os_sart_reconstruction(sinogram: np.ndarray, theta: np.ndarray, gamma: float, max_iter: int, num_subsets: int) -> np.ndarray:
    """
    Perform OS-SART (Ordered Subset Simultaneous Algebraic Reconstruction Technique) reconstruction.
    
    The sinogram is divided into equal subsets and updated iteratively.
    
    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram.
    theta : np.ndarray
        Array of projection angles (in degrees).
    gamma : float
        Relaxation parameter.
    max_iter : int
        Maximum number of iterations.
    num_subsets : int
        Number of subsets to divide the sinogram.
    
    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    N = sinogram.shape[0]
    reconstruction = np.zeros((N, N), dtype=np.float64)
    subset_size = len(theta) // num_subsets
    subsets = [theta[i * subset_size: (i + 1) * subset_size] for i in range(num_subsets)]
    
    for it in range(max_iter):
        current = reconstruction.copy()
        for subset in np.random.permutation(subsets):
            start_idx = np.where(theta == subset[0])[0][0]
            end_idx = start_idx + len(subset)
            subset_sino = sinogram[:, start_idx:end_idx]
            reconstruction = iradon_sart(subset_sino, theta=subset, image=reconstruction, relaxation=gamma)
        diff = np.linalg.norm(reconstruction - current)
        logger.info(f"OS-SART iteration {it+1}/{max_iter}: diff = {diff:.6f}")
    return reconstruction

def sirt_reconstruction(sinogram: np.ndarray, theta: np.ndarray, gamma: float, max_iter: int) -> np.ndarray:
    """
    Perform SIRT (Simultaneous Iterative Reconstruction Technique) reconstruction.
    
    A simple gradient descent update is applied.
    
    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram.
    theta : np.ndarray
        Array of projection angles (in degrees).
    gamma : float
        Relaxation parameter.
    max_iter : int
        Maximum number of iterations.
    
    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    N = sinogram.shape[0]
    reconstruction = np.zeros((N, N), dtype=np.float64)
    for it in range(max_iter):
        current = reconstruction.copy()
        proj = radon(reconstruction, theta=theta, circle=True)
        residual = sinogram - proj
        backproj = iradon(residual, theta=theta, filter_name=None, circle=True)
        reconstruction += gamma * backproj
        diff = np.linalg.norm(reconstruction - current)
        logger.info(f"SIRT iteration {it+1}/{max_iter}: diff = {diff:.6f}")
    return reconstruction

def osem_reconstruction(sinogram: np.ndarray, theta: np.ndarray, gamma: float, max_iter: int, num_subsets: int) -> np.ndarray:
    """
    Perform OSEM (Ordered Subset Expectation Maximization) reconstruction for PET.
    
    Parameters
    ----------
    sinogram : np.ndarray
        Attenuation-corrected PET sinogram.
    theta : np.ndarray
        Array of projection angles (in degrees).
    gamma : float
        Relaxation parameter.
    max_iter : int
        Maximum number of iterations.
    num_subsets : int
        Number of subsets to divide the sinogram.
    
    Returns
    -------
    np.ndarray
        Reconstructed PET image.
    """
    N = sinogram.shape[0]
    reconstruction = np.ones((N, N), dtype=np.float64)
    subset_size = len(theta) // num_subsets
    subsets = [np.arange(i * subset_size, (i + 1) * subset_size) for i in range(num_subsets)]
    
    for it in range(max_iter):
        current = reconstruction.copy()
        for subset_idx in np.random.permutation(num_subsets):
            angles_sub = theta[subsets[subset_idx]]
            sino_sub = sinogram[:, subsets[subset_idx]]
            proj = radon(reconstruction, theta=angles_sub, circle=True)
            ratio = np.divide(sino_sub, proj + 1e-8)
            bp = iradon(ratio, theta=angles_sub, filter_name=None, circle=True)
            reconstruction *= (1 + gamma * (bp - 1))
        diff = np.linalg.norm(reconstruction - current)
        logger.info(f"OSEM iteration {it+1}/{max_iter}: diff = {diff:.6f}")
    return reconstruction

def mlem_reconstruction(sinogram: np.ndarray, theta: np.ndarray, gamma: float, max_iter: int) -> np.ndarray:
    """
    Perform MLEM (Maximum Likelihood Expectation Maximization) reconstruction for PET.
    
    Parameters
    ----------
    sinogram : np.ndarray
        Attenuation-corrected PET sinogram.
    theta : np.ndarray
        Array of projection angles (in degrees).
    gamma : float
        Relaxation parameter to control the update step.
    max_iter : int
        Maximum number of iterations.
    
    Returns
    -------
    np.ndarray
        Reconstructed PET image.
    """
    N = sinogram.shape[0]
    reconstruction = np.ones((N, N), dtype=np.float64)
    ones_proj = np.ones_like(sinogram)
    norm_factor = iradon(ones_proj, theta=theta, filter_name=None, circle=True)
    
    for it in range(max_iter):
        current = reconstruction.copy()
        Ax = radon(reconstruction, theta=theta, circle=True)
        ratio = np.divide(sinogram, Ax + 1e-8)
        bp = iradon(ratio, theta=theta, filter_name=None, circle=True)
        update = gamma * (bp / (norm_factor + 1e-8) - 1) + 1
        reconstruction *= update
        diff = np.linalg.norm(reconstruction - current)
        logger.info(f"MLEM iteration {it+1}/{max_iter}: diff = {diff:.6f}")
    return reconstruction
