"""
2D Coordinate Mapping and Visual QC Script.

This script updates the central manifest by translating raw clinical XML 
float coordinates (x_raw, y_raw) into exact, integer-based pixel array indices 
(x_pixel, y_pixel). It validates that the coordinates fall within the image 
bounds (e.g., 512x512). To ensure the mapping algorithm is flawless before 
extracting ML tensors, it randomly selects 20 patients and generates visual 
Quality Control (QC) overlay images, plotting a red cross directly onto the 
target DICOM slice.
"""

import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random  # NEW: For random selection

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
DIR_QC = PROJECT_ROOT / "data" / "qc_overlays"
DIR_QC.mkdir(parents=True, exist_ok=True)

def clean_uid(uid):
    """
    Cleans a DICOM UID string by removing null bytes and whitespace.

    Args:
        uid (str): The raw UID string.

    Returns:
        str or None: The cleaned UID string, or None if the input is empty.
    """
    if pd.isna(uid) or uid is None: return None
    return str(uid).strip().replace('\x00', '')

def transform_to_hu(dicom_ds):
    """
    Converts raw pixel values to Hounsfield Units (Standard for CTs).

    This function applies the RescaleIntercept and RescaleSlope from the 
    DICOM header to the raw pixel array, transforming it into radiologically 
    standardized Hounsfield Units.

    Args:
        dicom_ds (pydicom.dataset.FileDataset): The loaded DICOM file object.

    Returns:
        numpy.ndarray: The converted image array in HU (as float64).
    """
    image = dicom_ds.pixel_array.astype(np.float64)
    intercept = dicom_ds.RescaleIntercept if 'RescaleIntercept' in dicom_ds else -1024.0
    slope = dicom_ds.RescaleSlope if 'RescaleSlope' in dicom_ds else 1.0
    image = (image * slope) + intercept
    return image

def main():
    """
    Executes the coordinate mapping and random visual QC pipeline.

    The function performs the following operations:
    1. Loads the manifest and filters for valid patients with XML data.
    2. Randomly selects a subset of up to 20 patients for visual QC.
    3. Iterates over the valid patients, locating their specific target DICOM 
       file using the `SOPInstanceUID`.
    4. Rounds the raw XML float coordinates to nearest integer pixel indices.
    5. Performs an in-bounds check against the image dimensions (Rows/Columns).
    6. For the selected QC patients, renders the CT slice (converted to HU) 
       and plots a red cross at the calculated pixel coordinates, saving it to disk.
    7. Updates the central manifest with the calculated `x_pixel`, `y_pixel`, 
       and a boolean `coordinate_mapped_successfully` flag.
    """
    print("Starting 2D Coordinate Mapping and QC...")
    
    if not FILE_MANIFEST.exists():
        print("ERROR: Manifest not found.")
        return
        
    df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    
    # We only process those that have an XML and where we found images
    mask = (df['xml_present'] == True) & (df['qc_pass'] == True)
    patients_to_process = df[mask].copy()
    
    # NEW: Select 20 RANDOM patients for the QC images
    all_valid_pids = patients_to_process['subject_id'].unique().tolist()
    num_qc_images = min(20, len(all_valid_pids)) # Take 20, or fewer if not enough exist
    qc_target_pids = set(random.sample(all_valid_pids, num_qc_images))
    
    print(f"Processing {len(patients_to_process)} patients with XML annotations...")
    print(f"Creating QC images for {num_qc_images} randomly selected patients.")

    # Iterate over all relevant patients (for the manifest update)
    for idx, row in tqdm(patients_to_process.iterrows(), total=len(patients_to_process), desc="Mapping Slices"):
        pid = row['subject_id']
        target_series = clean_uid(row['chosen_series_uid'])
        target_sop = clean_uid(row['sop_instance_uid'])
        
        # If the SOP UID is missing
        if not target_sop:
            df.at[idx, 'coordinate_mapped_successfully'] = False
            continue
            
        patient_dir = DIR_DICOM / pid
        if not patient_dir.exists():
            patient_dir = DIR_DICOM / "NSCLC Radiogenomics" / pid
            
        slice_found = False
        
        # STEP 1: Find the exact slice (SOPInstanceUID)
        for root_dir, dirs, files in os.walk(patient_dir):
            if slice_found: break
            dicom_files = [f for f in files if f.endswith('.dcm')]
            
            for dcm_file in dicom_files:
                dcm_path = Path(root_dir) / dcm_file
                try:
                    # We first read only the metadata to save RAM
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    current_series = clean_uid(ds.SeriesInstanceUID)
                    current_sop = clean_uid(ds.SOPInstanceUID)
                    
                    if current_series == target_series and current_sop == target_sop:
                        slice_found = True
                        
                        # STEP 2: Validate coordinates
                        columns = ds.Columns
                        rows = ds.Rows
                        
                        x_raw = float(row['x_raw'])
                        y_raw = float(row['y_raw'])
                        
                        # Round to whole pixels
                        x_pixel = int(round(x_raw))
                        y_pixel = int(round(y_raw))
                        
                        # In-Bounds Check
                        if (0 <= x_pixel < columns) and (0 <= y_pixel < rows):
                            # Add to manifest
                            df.at[idx, 'x_pixel'] = x_pixel
                            df.at[idx, 'y_pixel'] = y_pixel
                            df.at[idx, 'coordinate_mapped_successfully'] = True
                            
                            # STEP 3: Generate Overlay QC (Only for the random 20.)
                            if pid in qc_target_pids:
                                ds_full = pydicom.dcmread(dcm_path)
                                image_hu = transform_to_hu(ds_full)
                                
                                plt.figure(figsize=(8, 8))
                                plt.imshow(image_hu, cmap='gray', vmin=-1000, vmax=400) 
                                plt.plot(x_pixel, y_pixel, 'rx', markersize=15, markeredgewidth=3)
                                plt.title(f"{pid} - Tumor Marking\nSOP: {current_sop[-10:]}\nX: {x_pixel}, Y: {y_pixel}")
                                plt.axis('off')
                                
                                qc_path = DIR_QC / f"{pid}_QC_Overlay.png"
                                plt.savefig(qc_path, bbox_inches='tight')
                                plt.close()
                                
                                # So we don't accidentally make 2 images for the same patient
                                qc_target_pids.remove(pid) 
                                
                        else:
                            # Out of bounds
                            df.at[idx, 'coordinate_mapped_successfully'] = False
                            print(f"\nWARNING: {pid}: Coordinates Out-Of-Bounds. (x:{x_pixel}, y:{y_pixel} on {columns}x{rows})")
                            
                        break # Slice found, we can jump to the next file
                except Exception as e:
                    continue
        
        if not slice_found:
            df.at[idx, 'coordinate_mapped_successfully'] = False
            
    # Update Manifest
    df.to_csv(FILE_MANIFEST, index=False, sep=';', decimal=',')
    
    print("\n" + "="*50)
    print("MAPPING AND QC COMPLETE")
    print("="*50)
    print(f"Manifest updated: {FILE_MANIFEST}")
    print(f"QC images saved in: {DIR_QC}")
    print("\nMapping Statistics:")
    print(df[df['xml_present'] == True]['coordinate_mapped_successfully'].value_counts())

if __name__ == "__main__":
    main()