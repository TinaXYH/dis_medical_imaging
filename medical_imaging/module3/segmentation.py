# medical_imaging/module3/segmentation.py
import os
import logging
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute segmentation evaluation metrics including Dice, IoU, sensitivity,
    specificity, precision, and F1 score.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask.
    gt : np.ndarray
        Ground truth binary mask.
    
    Returns
    -------
    dict
        Dictionary of metrics.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    TP = np.sum(pred & gt)
    FP = np.sum(pred & ~gt)
    FN = np.sum(~pred & gt)
    TN = np.sum(~pred & ~gt)
    
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity + 1e-8)
    
    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1
    }

def process_case(case_id: str, data_folder: str) -> dict:
    """
    Process a single case by loading the CT scan and mask, extracting a subvolume
    (ROI) based on the mask, applying a threshold-based segmentation, and computing metrics.
    
    Parameters
    ----------
    case_id : str
        Case identifier (e.g., "case_0").
    data_folder : str
        Directory containing the NIfTI files.
    
    Returns
    -------
    dict
        Dictionary with subvolume, bounding box info, segmentation result, and metrics.
    """
    scan_file = os.path.join(data_folder, f"{case_id}.nii")
    mask_file = os.path.join(data_folder, f"{case_id}_mask.nii")
    
    try:
        scan_img = nib.load(scan_file)
        scan_data = scan_img.get_fdata()
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
    except Exception as e:
        logger.error(f"Failed to load case {case_id}: {e}")
        return None
    
    indices = np.where(mask_data > 0)
    if indices[0].size == 0:
        logger.warning(f"{case_id}: No segmented voxels found!")
        return None
    min_x, max_x = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    min_z, max_z = np.min(indices[2]), np.max(indices[2])
    
    expansion_xy = 30
    expansion_z = 5
    lower_x = max(0, min_x - expansion_xy)
    upper_x = min(scan_data.shape[0], max_x + expansion_xy + 1)
    lower_y = max(0, min_y - expansion_xy)
    upper_y = min(scan_data.shape[1], max_y + expansion_xy + 1)
    lower_z = max(0, min_z - expansion_z)
    upper_z = min(scan_data.shape[2], max_z + expansion_z + 1)
    
    bbox = {
        "original": {"x": (min_x, max_x), "y": (min_y, max_y), "z": (min_z, max_z)},
        "expanded": {"x": (lower_x, upper_x), "y": (lower_y, upper_y), "z": (lower_z, upper_z)}
    }
    
    scan_sub = scan_data[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z]
    mask_sub = mask_data[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z]
    
    segmented_intensities = scan_sub[mask_sub > 0]
    if segmented_intensities.size == 0:
        logger.warning(f"{case_id}: No segmented intensities in subvolume!")
        return None
    threshold_min = np.min(segmented_intensities)
    threshold_max = np.max(segmented_intensities)
    
    segmentation_result = (scan_sub >= threshold_min) & (scan_sub <= threshold_max)
    
    metrics = compute_metrics(segmentation_result, mask_sub > 0)
    logger.info(f"{case_id}: Metrics = {metrics}")
    
    return {
        "case_id": case_id,
        "scan_sub": scan_sub,
        "mask_sub": mask_sub,
        "segmentation_result": segmentation_result,
        "bbox": bbox,
        "metrics": metrics
    }

def draw_slice_bounding_box(ax, slice_mask: np.ndarray, color: str = 'r'):
    """
    Draw a bounding box on the given axis based on the minimal rectangle that encloses the nonzero pixels.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw the bounding box on.
    slice_mask : np.ndarray
        2D binary mask of a slice.
    color : str, optional
        Color for the bounding box (default 'r').
    """
    rows, cols = np.where(slice_mask > 0)
    if rows.size == 0 or cols.size == 0:
        return
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    rect = patches.Rectangle((min_c, min_r), max_c - min_c, max_r - min_r,
                             linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

def visualize_subvolume(result: dict, n_slices: int = 3):
    """
    Visualize several slices of the extracted subvolume along with the ground truth and threshold segmentation.
    
    Parameters
    ----------
    result : dict
        Dictionary output from process_case.
    n_slices : int, optional
        Number of slices to visualize (default 3).
    """
    if result is None:
        print("No result to visualize.")
        return
    case_id = result["case_id"]
    scan_sub = result["scan_sub"]
    mask_sub = result["mask_sub"]
    seg_result = result["segmentation_result"]
    dice = result["metrics"]["dice"]
    
    zsize = scan_sub.shape[2]
    slice_indices = [zsize // 4, zsize // 2, (3 * zsize) // 4]
    
    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(12, 4 * len(slice_indices)))
    for i, z in enumerate(slice_indices):
        axes[i, 0].imshow(scan_sub[:, :, z], cmap="gray")
        axes[i, 0].set_title(f"{case_id} - CT (z={z})")
        axes[i, 0].axis("off")
        draw_slice_bounding_box(axes[i, 0], mask_sub[:, :, z], color='r')
        
        axes[i, 1].imshow(mask_sub[:, :, z], cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(seg_result[:, :, z], cmap="gray")
        axes[i, 2].set_title(f"Threshold Segmentation\nDice={dice:.4f}")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.show()

def visualize_full_and_subvolume(result: dict, full_scan: np.ndarray, full_mask: np.ndarray, n_slices: int = 3):
    """
    Compare full scan slices with the extracted subvolume.
    
    Parameters
    ----------
    result : dict
        Result dictionary from process_case.
    full_scan : np.ndarray
        The full CT scan data.
    full_mask : np.ndarray
        The full ground truth mask.
    n_slices : int, optional
        Number of slices to visualize.
    """
    if result is None:
        print("No result to visualize.")
        return
    
    case_id = result["case_id"]
    scan_sub = result["scan_sub"]
    mask_sub = result["mask_sub"]
    seg_result = result["segmentation_result"]
    dice = result["metrics"]["dice"]
    
    bbox = result["bbox"]["expanded"]
    lower_z = bbox["z"][0]
    
    zsize_sub = scan_sub.shape[2]
    slice_indices_sub = [zsize_sub // 4, zsize_sub // 2, (3 * zsize_sub) // 4]
    
    fig, axes = plt.subplots(len(slice_indices_sub), 5, figsize=(18, 4 * len(slice_indices_sub)))
    for i, z_sub in enumerate(slice_indices_sub):
        z_full = z_sub + lower_z
        
        axes[i, 0].imshow(full_scan[:, :, z_full], cmap="gray")
        axes[i, 0].set_title(f"Full CT (z={z_full})")
        axes[i, 0].axis("off")
        draw_slice_bounding_box(axes[i, 0], full_mask[:, :, z_full], color='r')
        
        axes[i, 1].imshow(full_mask[:, :, z_full], cmap="gray")
        axes[i, 1].set_title(f"Full GT (z={z_full})")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(scan_sub[:, :, z_sub], cmap="gray")
        axes[i, 2].set_title(f"Sub CT (z={z_sub})")
        axes[i, 2].axis("off")
        draw_slice_bounding_box(axes[i, 2], mask_sub[:, :, z_sub], color='r')
        
        axes[i, 3].imshow(mask_sub[:, :, z_sub], cmap="gray")
        axes[i, 3].set_title("Sub GT")
        axes[i, 3].axis("off")
        
        axes[i, 4].imshow(seg_result[:, :, z_sub], cmap="gray")
        axes[i, 4].set_title(f"Threshold\nDice={dice:.4f}")
        axes[i, 4].axis("off")
    
    plt.tight_layout()
    plt.show()
