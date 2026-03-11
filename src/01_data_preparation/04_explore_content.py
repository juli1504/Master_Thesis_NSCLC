"""
Initial Data Exploration and Visualization Script.

This script performs basic exploratory data analysis (EDA) on the curated 
dataset. It generates a distribution plot of the clinical diagnoses (histology), 
extracts semantic features from a sample patient's XML annotation, and 
visualizes a middle slice of their 3D CT volume using Matplotlib.
"""

import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
import os

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
PATH_MAPPING = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
PATH_METADATA = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
PATH_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

# New folder for saved images
DIR_FIGURES = PROJECT_ROOT / "results" / "figures"
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

def print_xml_features(xml_path):
    """
    Extracts and prints semantic observation characteristics from an AIM v4 XML file.

    This function parses the XML tree to locate 'ImagingObservationCharacteristic' 
    nodes, extracting their nested label and value codes. This provides human-readable 
    features (e.g., 'Margin', 'Spiculation') annotated by the radiologist.

    Args:
        xml_path (Path): The file path to the patient's XML annotation file.

    Returns:
        None. The extracted features are printed to standard output.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        print("\n--- XML CONTENT (Excerpt) ---")
        features_found = 0
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if tag == 'ImagingObservationCharacteristic':
                label = None
                value = None
                for child in elem:
                    c_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if c_tag == 'label':
                        label = child.attrib.get('value')
                    elif c_tag == 'typeCode':
                        value = child.attrib.get('code') or child.attrib.get('codeSystem')
                
                if label:
                    print(f"  • {label}: {value}")
                    features_found += 1
                    
        if features_found == 0:
            print("  (No structured features found)")
            
    except Exception as e:
        print(f"Error reading XML: {e}")

def main():
    """
    Executes the initial data exploration pipeline.

    The function performs three main steps:
    1. Reads the clinical labels CSV, calculates the distribution of diagnoses 
       (histology), and saves a bar chart visualization.
    2. Selects the first valid patient from the image mapping file and prints 
       their structured XML features using `print_xml_features()`.
    3. Locates the patient's physical DICOM directory, loads the middle slice 
       of the CT scan, and saves a 2D plot of the slice.

    Returns:
        None. Generated plots are saved to the 'results/figures' directory.
    """
    print("### STEP 2: INITIAL DATA EXPLORATION ###\n")
    
    # 1. LABEL DISTRIBUTION (CSV)
    print("(1) LABEL DISTRIBUTION (CSV)")
    df_clinical = pd.read_csv(PATH_CLINICAL)
    df_clinical.columns = df_clinical.columns.str.strip()
    
    counts = df_clinical['Histology'].value_counts()
    print(counts)
    
    # Create plot
    plt.figure(figsize=(10, 6)) # Wider image to fit text
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title('Distribution of diagnoses (histology)')
    plt.ylabel('Number of Patients')
    plt.xlabel('Diagnosis')
    
    # Fix for horizontal text:
    plt.xticks(rotation=0) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Prevents text from being cut off
    
    # Save first, then display
    save_path_chart = DIR_FIGURES / "distribution_histology.png"
    plt.savefig(save_path_chart, dpi=300)
    print(f"\n>> Chart saved to: {save_path_chart}")
    
    plt.show() # Pop-up window
    
    # 2. ANALYZE EXAMPLE PATIENT
    print("\n(2) ANALYZE EXAMPLE PATIENT")
    df_map = pd.read_csv(PATH_MAPPING)
    match = df_map[df_map['File Location'].notna()].iloc[0]
    subject_id = match['Subject ID']
    
    print(f"Patient: {subject_id}")
    print_xml_features(DIR_XML / match['XML_File'])
    
    # 3. DICOM IMAGE
    print(f"\n(3) DICOM IMAGE (CT) FOR {subject_id}")
    
    raw_path = match['File Location']
    clean_path = raw_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    dicom_path = DIR_DICOM / clean_path
    
    target_file = None
    if dicom_path.is_dir():
        files = sorted(list(dicom_path.glob("*.dcm")))
        if files:
            mid_slice = len(files) // 2
            target_file = files[mid_slice]
            print(f"  • Loading slice {mid_slice+1} of {len(files)}: {target_file.name}")
    elif dicom_path.exists():
        target_file = dicom_path

    if target_file:
        ds = pydicom.dcmread(target_file)
        
        plt.figure(figsize=(8,8))
        plt.imshow(ds.pixel_array, cmap='gray')
        plt.title(f"Patient: {subject_id} | {ds.Modality} | Slice Location: {ds.get('SliceLocation', 'N/A')}")
        plt.axis('off')
        
        # Save & Display
        save_path_img = DIR_FIGURES / f"example_ct_{subject_id}.png"
        plt.savefig(save_path_img, dpi=300)
        print(f">> CT image saved to: {save_path_img}")
        
        plt.show() # Pop-up window
        print("\nData pipeline is ready.")

if __name__ == "__main__":
    main()