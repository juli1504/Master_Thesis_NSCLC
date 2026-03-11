"""
Pure XML to DICOM Validation and Mapping Script.

This script establishes the Ground Truth mapping by bypassing the provided
metadata CSV entirely. Instead, it parses the AIM XML annotations directly
to extract the target `SeriesInstanceUID`. It then physically scans the 
patient's local DICOM directories to find the exact matching folder, 
extracting spatial metadata like slice thickness and slice count in the process.
"""

import pydicom
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"  # <--- THIS IS THE FIX
DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

# XML Namespace
NS = {'aim': 'gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM'}

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

def parse_xml(xml_path):
    """
    Reads the patient ID and the required image UID from the XML.

    This function navigates the AIM XML tree using predefined namespaces to 
    extract the patient identifier and the exact SeriesInstanceUID annotated 
    by the radiologist.

    Args:
        xml_path (Path): The file path to the AIM XML annotation.

    Returns:
        tuple: A tuple containing (patient_id, target_uid) as strings. 
        Returns (None, None) if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        person = root.find('.//aim:person/aim:id', NS)
        patient_id = person.attrib['value'] if person is not None else None
        
        series_uid_node = root.find('.//aim:imageSeries/aim:instanceUid', NS)
        target_uid = series_uid_node.attrib['root'] if series_uid_node is not None else None
        
        return patient_id, clean_uid(target_uid)
    except Exception:
        return None, None

def find_dicom_folder_for_uid(patient_dir, target_uid):
    """
    Searches ONLY the folder of this specific patient for the correct UID.

    This function iterates through all subdirectories of a given patient, 
    reading the header of the first DICOM file in each directory to check 
    if the `SeriesInstanceUID` matches the target from the XML.

    Args:
        patient_dir (Path): The base directory containing the patient's DICOMs.
        target_uid (str): The target SeriesInstanceUID to find.

    Returns:
        tuple: (matched_folder_path, status_message, slice_count, slice_thickness).
        If not found, returns None for the path, count, and thickness.
    """
    if not patient_dir.exists():
        return None, "Patient folder missing", None, None

    for root, dirs, files in os.walk(patient_dir):
        dicom_files = [f for f in files if f.endswith('.dcm')]
        
        if not dicom_files:
            continue
            
        test_dcm_path = Path(root) / dicom_files[0]
        try:
            ds = pydicom.dcmread(test_dcm_path, stop_before_pixels=True)
            current_uid = clean_uid(ds.SeriesInstanceUID)
            
            if current_uid == target_uid:
                slice_count = len(dicom_files)
                thickness = float(ds.SliceThickness) if 'SliceThickness' in ds else None
                return Path(root), "Successful", slice_count, thickness
                
        except Exception:
            continue
            
    return None, "UID not found in patient folder", None, None

def main():
    """
    Executes the pure XML-to-DICOM validation and mapping pipeline.

    The function performs the following steps:
    1. Locates all XML files in the raw data directory.
    2. Parses each XML for the target Series UID.
    3. Physically searches the respective patient's DICOM folders for the match.
    4. Aggregates the results (including slice counts and thicknesses) into a DataFrame.
    5. Saves the mapping to a CSV file and prints a statistical summary.
    """
    print("\n" + "="*60)
    print("PURE XML TO DICOM VALIDATION (No CSV used.)")
    print("="*60)
    
    # 1. Find all XMLs in the correct folder
    print(f"Searching for XML files in: {DIR_XML}")
    if not DIR_XML.exists():
        print(f"STOP: The folder {DIR_XML} does not exist.")
        return

    xml_files = list(DIR_XML.rglob("*.xml"))
    print(f"Found: {len(xml_files)} XML files (Ground Truth).")
    
    if len(xml_files) == 0:
        print("STOP: No XML files found. Script is aborting.")
        return
    
    results = []
    
    # 2. Find the matching folder for each XML
    for xml_path in tqdm(xml_files, desc="Processing patients"):
        patient_id, target_uid = parse_xml(xml_path)
        
        if not patient_id or not target_uid:
            continue
            
        # Where we need to look
        patient_dir = DIR_DICOM / patient_id
        if not patient_dir.exists():
            patient_dir = DIR_DICOM / "NSCLC Radiogenomics" / patient_id
            
        # 3. Search the folder and read DICOM data
        matched_folder, status, slice_count, thickness = find_dicom_folder_for_uid(patient_dir, target_uid)
        
        results.append({
            'PatientID': patient_id,
            'XML_Target_UID': target_uid,
            'Status': status,
            'SliceCount': slice_count,
            'SliceThickness': thickness,
            'Found_Folder': str(matched_folder.relative_to(PROJECT_ROOT)) if matched_folder else None
        })
        
    if len(results) == 0:
        print("STOP: Error parsing all XMLs.")
        return

    # 4. Save & analyze results
    df = pd.DataFrame(results)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['PatientID', 'XML_Target_UID'])
    
    out_path = DIR_PROCESSED / "xml_dicom_patient_matching.csv"
    df.to_csv(out_path, index=False, sep=';', decimal=',')
    
    print("\n" + "="*60)
    print("RESULT ANALYSIS")
    print("="*60)
    
    df_success = df[df['Status'] == 'Successful']
    print(f"Successfully matched patients: {len(df_success)} of {len(df)}")
    
    if len(df_success) > 0:
        print("\nDistribution of slice thicknesses:")
        print(df_success['SliceThickness'].value_counts(dropna=False).to_string())
        
        print("\nSlices per patient (3D volume size):")
        print(f"Minimum Slices: {df_success['SliceCount'].min():.0f}")
        print(f"Maximum Slices: {df_success['SliceCount'].max():.0f}")
        print(f"Average:   {df_success['SliceCount'].mean():.0f}")
        
    df_errors = df[df['Status'] != 'Successful']
    if len(df_errors) > 0:
        print(f"\n{len(df_errors)} errors found (e.g., DICOMs deleted or UID not found).")

    print(f"\nList saved to: {out_path}")

if __name__ == "__main__":
    main()