"""
CT Windowing Sweep Experiment Script.

This script acts as a visual 'walkthrough' of radiological windowing. 
It loads a single middle slice from a sample patient and systematically 
sweeps the Window Center (Level) from air-density values to bone-density 
values, keeping the Window Width constant. The result is a large grid (poster) 
showing how the visible anatomical structures change across the Hounsfield scale.
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import math

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
PATH_MAPPING = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
DIR_FIGURES = PROJECT_ROOT / "results" / "figures"
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

def load_middle_slice(path):
    """
    Loads only the middle slice from a DICOM folder for testing purposes.

    The function reads all '.dcm' files in the directory, sorts them anatomically 
    along the Z-axis using 'ImagePositionPatient', and returns the median slice.

    Args:
        path (Path or str): The directory path containing the DICOM files.

    Returns:
        pydicom.dataset.FileDataset or None: The middle DICOM slice object, 
        or None if the directory is empty.
    """
    files = [f for f in os.listdir(path) if f.endswith('.dcm')]
    if not files:
        return None
    
    # We load all to sort them (important!)
    slices = [pydicom.dcmread(path / f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    return slices[len(slices) // 2] # The middle

def get_pixels_hu(slice_item):
    """
    Converts a single DICOM slice into Hounsfield Units (HU).

    This function applies the linear transformation defined in the DICOM header 
    (RescaleIntercept and RescaleSlope) to convert scanner pixel values into 
    standardized radiodensity metrics (Hounsfield Units). 

    Args:
        slice_item (pydicom.dataset.FileDataset): The single DICOM object.

    Returns:
        numpy.ndarray: A 2D array containing the calculated HU values.
    """
    image = slice_item.pixel_array.astype(np.int16)
    image[image == -2000] = -1000 # Correction for Air
    
    intercept = slice_item.RescaleIntercept
    slope = slice_item.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    return image

def apply_window(image, center, width):
    """
    Applies radiological CT windowing to restrict the visible HU range.

    Args:
        image (numpy.ndarray): The input 2D image array in Hounsfield Units.
        center (int): The center value (Level) of the window in HU.
        width (int): The width of the window in HU.

    Returns:
        numpy.ndarray: The windowed (clipped) image array.
    """
    img_min = center - width // 2
    img_max = center + width // 2
    return np.clip(image, img_min, img_max)

def main():
    """
    Executes the windowing sweep experiment and generates a visualization grid.

    The function performs the following steps:
    1. Loads the exact DICOM path for a sample patient (AMC-003).
    2. Extracts the middle slice and converts it to Hounsfield Units.
    3. Iterates through a defined range of Window Centers (from -900 to 600) 
       using a fixed Window Width.
    4. Plots each resulting windowed image into a large matplotlib grid.
    5. Saves the final grid as a 'poster' image for visual inspection.

    Returns:
        None. The resulting grid is saved to the 'results/figures' directory.
    """
    print("### STEP 4: WINDOWING EXPERIMENT (The 'Walkthrough') ###")
    
    # 1. Load patient
    df_map = pd.read_csv(PATH_MAPPING)
    match = df_map[df_map['Subject ID'] == 'AMC-003'].iloc[0] # We stick with our friend AMC-003
    
    raw_path = match['File Location']
    clean_path = raw_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    full_path = DIR_DICOM / clean_path
    
    print(f"Loading middle slice of {match['Subject ID']}...")
    mid_slice = load_middle_slice(full_path)
    img_hu = get_pixels_hu(mid_slice)
    
    # 2. Configure the experiment
    # We sweep the "Center" (Level) from -900 (Air) to +500 (Bone)
    # We keep the "Width" (Contrast) constant at 1000 (good average)
    
    start_center = -900
    end_center = 600
    step = 60 # Steps of 60 HU (yields approx. 25 images)
    fixed_width = 1500 
    
    centers = list(range(start_center, end_center, step))
    num_plots = len(centers)
    
    # Calculate grid (e.g., 5 columns)
    cols = 5
    rows = math.ceil(num_plots / cols)
    
    print(f"Creating {num_plots} images (Center from {start_center} to {end_center})...")
    
    plt.figure(figsize=(20, 4 * rows)) # Large image
    
    for i, center in enumerate(centers):
        ax = plt.subplot(rows, cols, i + 1)
        
        # Apply windowing
        img_windowed = apply_window(img_hu, center=center, width=fixed_width)
        
        ax.imshow(img_windowed, cmap='gray')
        ax.set_title(f"Level (Center): {center} HU\n(Width: {fixed_width})", fontsize=10)
        ax.axis('off')
            
    plt.tight_layout()
    save_path = DIR_FIGURES / "window_walkthrough.png"
    plt.savefig(save_path, dpi=150) # Not too high, otherwise the file becomes huge
    print(f">> Poster saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()