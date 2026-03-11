"""
Detailed CT Volume Analysis and Visualization Script.

This script performs an in-depth exploratory data analysis (EDA) on a 
single patient's full 3D CT scan. It demonstrates core medical imaging 
preprocessing techniques, including loading DICOM series, anatomical Z-axis 
sorting, mathematical conversion to Hounsfield Units (HU), and diagnostic 
radiological windowing. It generates and saves several visual diagnostic plots.
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
PATH_MAPPING = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
DIR_FIGURES = PROJECT_ROOT / "results" / "figures"
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

def load_scan(path):
    """
    Loads all DICOM files from a specified folder and sorts them spatially.

    The function reads all '.dcm' files in the directory and sorts the list 
    based on the 'ImagePositionPatient[2]' DICOM header attribute. This ensures 
    the 2D slices are stacked in the correct anatomical order along the Z-axis.

    Args:
        path (Path or str): The directory path containing the DICOM files.

    Returns:
        list of pydicom.dataset.FileDataset: A spatially sorted list of DICOM objects.
    """
    # 1. List all files
    slices = [pydicom.dcmread(path / f) for f in os.listdir(path) if f.endswith('.dcm')]
    
    # 2. Sort by ImagePositionPatient (Z-coordinate)
    # This is important, otherwise the images are mixed up!
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    return slices

def get_pixels_hu(slices):
    """
    Converts raw DICOM pixel values to a 3D numpy array of Hounsfield Units (HU).

    This function applies the linear transformation defined in the DICOM header 
    (RescaleIntercept and RescaleSlope) to convert arbitrary scanner pixel 
    values into standardized radiodensity metrics (Hounsfield Units). It also 
    normalizes out-of-bounds scanner padding (often -2000) to 0 (Air).

    Args:
        slices (list of pydicom.dataset.FileDataset): Spatially sorted DICOM objects.

    Returns:
        numpy.ndarray: A 3D array of shape (Slices, Rows, Columns) containing 
        the calculated HU values as 16-bit integers.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Outside the scan area is often -2000, we set this to 0 (air)
    image[image == -2000] = 0
    
    # Conversion: HU = Pixel * Slope + Intercept
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def apply_window(image, center, width):
    """
    Applies radiological CT windowing to restrict the visible HU range.

    Windowing acts as a contrast/brightness adjustment, clipping the HU values 
    to a specific range defined by a Window Center (Level) and Window Width. 
    This highlights specific tissues, such as lung parenchyma or mediastinal soft tissue.

    Args:
        image (numpy.ndarray): The input 2D or 3D image array in Hounsfield Units.
        center (int): The center value (Level) of the window in HU.
        width (int): The width of the window in HU.

    Returns:
        numpy.ndarray: The windowed (clipped) image array.
    """
    img_min = center - width // 2
    img_max = center + width // 2
    
    windowed = np.clip(image, img_min, img_max)
    return windowed

def main():
    """
    Executes the detailed CT volume analysis and visualization pipeline.

    The function performs the following steps on a sample patient:
    1. Determines the physical DICOM directory path from the mapping file.
    2. Loads the full 3D scan and converts it to Hounsfield Units (HU).
    3. Generates a histogram illustrating the global HU density distribution.
    4. Extracts the middle slice and visualizes it under three conditions: 
       Raw HU, Lung Window, and Mediastinal Window.
    5. Creates a 4x4 overview montage (grid) of evenly spaced slices across 
       the entire Z-axis using the Lung Window.

    Returns:
        None. All visualizations are saved to the 'results/figures' directory.
    """
    print("### STEP 3: DETAILED CT ANALYSIS (WINDOWING & 3D) ###")
    
    # 1. Get patient path
    df_map = pd.read_csv(PATH_MAPPING)
    match = df_map[df_map['File Location'].notna()].iloc[0]
    subject_id = match['Subject ID']
    
    raw_path = match['File Location']
    clean_path = raw_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    full_path = DIR_DICOM / clean_path
    
    print(f"Loading 3D volume for {subject_id}...")
    
    # 2. Load volume & convert to HU
    slices = load_scan(full_path)
    patient_pixels = get_pixels_hu(slices)
    
    print(f" -> Volume shape: {patient_pixels.shape} (Slices, X, Y)")
    print(f" -> Min HU: {np.min(patient_pixels)}, Max HU: {np.max(patient_pixels)}")

    # 3. VISUALIZATION 1: HISTOGRAM (Density distribution)
    plt.figure(figsize=(10, 5))
    plt.hist(patient_pixels.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.title(f"HU Distribution for {subject_id}")
    plt.xlim(-1000, 2000) # We look at the relevant range
    # Mark important areas
    plt.axvline(-1000, color='r', linestyle='dashed', label='Air (-1000)')
    plt.axvline(0, color='b', linestyle='dashed', label='Water (0)')
    plt.axvline(400, color='g', linestyle='dashed', label='Bone (>400)')
    plt.legend()
    plt.savefig(DIR_FIGURES / "analysis_hu_histogram.png", dpi=300)
    plt.show()

    # 4. VISUALIZATION 2: WINDOWING COMPARISON
    # We take a slice from the middle
    mid_idx = len(patient_pixels) // 2
    slice_img = patient_pixels[mid_idx]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # A) Without windowing (Raw)
    ax[0].imshow(slice_img, cmap='gray')
    ax[0].set_title("No Windowing (Raw HU)")
    
    # B) Lung Window (W:1500, L:-600) -> Here you can see the tumor
    lung_window = apply_window(slice_img, center=-600, width=1500)
    ax[1].imshow(lung_window, cmap='gray')
    ax[1].set_title("Lung Window (L:-600, W:1500)")
    
    # C) Mediastinal Window (W:350, L:50) -> Here you can see soft tissues/heart
    soft_window = apply_window(slice_img, center=50, width=350)
    ax[2].imshow(soft_window, cmap='gray')
    ax[2].set_title("Mediastinal Window (L:50, W:350)")
    
    plt.savefig(DIR_FIGURES / "analysis_windowing_comparison.png", dpi=300)
    plt.show()

    # 5. VISUALIZATION 3: MONTAGE (The Grid)
    # Solution for "What if the lung is not in the middle?"
    print("Creating overview grid (Montage)...")
    
    # We take e.g., 16 slices at equal intervals
    num_slices = 16
    indices = np.linspace(0, len(patient_pixels)-1, num_slices).astype(int)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # We show the lung window, as we can recognize the most there
        img = apply_window(patient_pixels[idx], center=-600, width=1500)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis('off')
        
    plt.suptitle(f"Volume Overview: {subject_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(DIR_FIGURES / "analysis_volume_montage.png", dpi=300)
    plt.show()
    
    print("Analysis complete. Images saved in results/figures/.")

if __name__ == "__main__":
    main()