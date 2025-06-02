# mri_denoising/denoise.py
import numpy as np
from numpy import fft
from skimage.restoration import denoise_wavelet

def denoise_mri(kspace_data: np.ndarray, coil_dim: int = 0) -> np.ndarray:
    """
    Denoise MRI images from k-space data and combine multiple coil images 
    using the root-sum-of-squares method.
    
    This function performs the following steps:
      1. Transform the k-space data to image space using inverse FFT.
      2. For each coil, extract magnitude and phase; normalize the magnitude,
         apply wavelet denoising, then restore the original intensity range and
         recombine with the phase to obtain a denoised complex image.
      3. Combine the denoised images from all coils by computing the square 
         root of the sum of the squared magnitudes.
    
    Parameters
    ----------
    kspace_data : np.ndarray
        A numpy array containing the complex k-space data. The expected shape 
        is such that one dimension corresponds to the coils (e.g., (n_coils, H, W)).
    coil_dim : int, optional
        The dimension index corresponding to the coils. Must be 0, 1, or 2.
    
    Returns
    -------
    combined_image : np.ndarray
        The final combined denoised MRI image (magnitude only) as a 2D numpy array.
    """
    # Validate coil_dim parameter before using it
    if coil_dim not in (0, 1, 2):
        raise ValueError("coil_dim must be 0, 1, or 2.")
    
    # Get the number of coils from the specified dimension
    n_coils = kspace_data.shape[coil_dim]
    # Create an empty array to store the image space data for each coil
    image_data = np.empty_like(kspace_data, dtype=np.complex128)

    # Process each coil individually
    for i in range(n_coils):
        # Slice the data for the current coil based on coil_dim
        if coil_dim == 0:
            coil_kspace = kspace_data[i, :, :]
        elif coil_dim == 1:
            coil_kspace = kspace_data[:, i, :]
        else:  # coil_dim == 2
            coil_kspace = kspace_data[:, :, i]

        # Inverse FFT to convert k-space data to image space
        coil_image = fft.ifft2(coil_kspace)
        # Extract magnitude and phase from the complex image
        magnitude = np.abs(coil_image)
        phase = np.angle(coil_image)
        # Normalize magnitude to [0,1]
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        norm = (magnitude - mag_min) / (mag_max - mag_min) if mag_max > mag_min else magnitude
        # Apply wavelet denoising to the normalized magnitude
        denoised_norm = denoise_wavelet(norm, wavelet='db4', mode='soft', channel_axis=None)
        # Restore the original magnitude scale
        denoised_magnitude = denoised_norm * (mag_max - mag_min) + mag_min
        # Recombine with the original phase to form a denoised complex image
        denoised_image = denoised_magnitude * np.exp(1j * phase)

        # Store the denoised image back to the image_data array
        if coil_dim == 0:
            image_data[i, :, :] = denoised_image
        elif coil_dim == 1:
            image_data[:, i, :] = denoised_image
        else:
            image_data[:, :, i] = denoised_image

    # Combine the denoised images from all coils using the root-sum-of-squares method
    if coil_dim == 0:
        combined = np.sqrt(np.sum(np.abs(image_data)**2, axis=0))
    elif coil_dim == 1:
        combined = np.sqrt(np.sum(np.abs(image_data)**2, axis=1))
    else:
        combined = np.sqrt(np.sum(np.abs(image_data)**2, axis=2))

    return combined
