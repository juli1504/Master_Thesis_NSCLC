"""
Central Data Manifest Creation Script.

This script acts as the core of the data curation pipeline. It aggregates 
the clinical ground truth (histology), the spatial XML annotations (X, Y 
coordinates and SOPInstanceUID), and the physical DICOM metadata (spacing, 
thickness, slice count) into a single, centralized 'manifest.csv'. 

For patients without XML annotations, it implements an automated fallback 
search to locate the most suitable CT scan (e.g., thinnest slices, highest 
slice count, excluding scouts/topograms) for potential unsupervised learning.
"""

import os
import pandas as pd
import pydicom
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"
# The new, correct path to your file!
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

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

def parse_all_xmls():
    """
    Parses all XML files and builds a dictionary mapping Patient IDs to XML data.

    This function iterates over all AIM v4 XML files in the raw data folder, 
    extracting the target SeriesInstanceUID, the exact SOPInstanceUID of the 
    slice containing the tumor, and the raw (X, Y) pixel coordinates.

    Returns:
        dict: A dictionary where keys are Patient IDs and values are nested 
        dictionaries containing 'series_uid', 'sop_uid', 'x_raw', and 'y_raw'.
    """
    xml_dict = {}
    xml_files = list(DIR_XML.rglob("*.xml"))
    
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            person = root.find('.//aim:person/aim:id', NS)
            if person is None: continue
            pid = person.attrib.get('value')
            
            series_node = root.find('.//aim:imageSeries/aim:instanceUid', NS)
            t_uid = clean_uid(series_node.attrib.get('root')) if series_node is not None else None
            
            sop_node = root.find('.//aim:imageReferenceEntityCollection//aim:imageCollection/aim:Image/aim:sopInstanceUid', NS)
            sop_uid = clean_uid(sop_node.attrib.get('root')) if sop_node is not None else None
                
            coord_node = root.find('.//aim:markupEntityCollection//aim:TwoDimensionSpatialCoordinate', NS)
            x_raw, y_raw = None, None
            if coord_node is not None:
                x_node = coord_node.find('aim:x', NS)
                y_node = coord_node.find('aim:y', NS)
                if x_node is not None and y_node is not None:
                    x_raw = float(x_node.attrib.get('value'))
                    y_raw = float(y_node.attrib.get('value'))
            
            # We save the data under the PatientID
            xml_dict[pid] = {
                'series_uid': t_uid,
                'sop_uid': sop_uid,
                'x_raw': x_raw,
                'y_raw': y_raw
            }
        except Exception:
            pass
            
    return xml_dict

def get_best_fallback_series(patient_dir):
    """
    Automatically searches for the best CT series if no XML exists.

    This acts as a fallback heuristic. It scans all DICOM series for a patient, 
    ignores scouts/topograms, and scores the remaining CT series based on 
    slice thickness (thinner is better) and slice count (more is better).

    Args:
        patient_dir (Path): The base directory containing the patient's DICOMs.

    Returns:
        tuple: (best_uid, best_meta_dict). If no valid CT is found, returns (None, None).
    """
    best_meta = None
    best_score = (999.0, 0) # (Thickness ascending, Slices descending)
    best_uid = None
    
    for root_dir, dirs, files in os.walk(patient_dir):
        dicom_files = [f for f in files if f.endswith('.dcm')]
        if len(dicom_files) < 20: # Ignore folders with almost no images
            continue
            
        try:
            ds = pydicom.dcmread(Path(root_dir) / dicom_files[0], stop_before_pixels=True)
            if 'Modality' in ds and ds.Modality != 'CT': continue
            
            desc = str(ds.SeriesDescription).lower() if 'SeriesDescription' in ds else ""
            if any(x in desc for x in ['scout', 'topogram', 'loc']): continue
            
            thickness = float(ds.SliceThickness) if 'SliceThickness' in ds else 999.0
            slice_count = len(dicom_files)
            
            score = (thickness, -slice_count)
            if score < best_score:
                best_score = score
                best_uid = clean_uid(ds.SeriesInstanceUID)
                
                spacing_y, spacing_x = None, None
                if 'PixelSpacing' in ds:
                    spacing_y = float(ds.PixelSpacing[0])
                    spacing_x = float(ds.PixelSpacing[1])
                    
                best_meta = {
                    'slice_thickness': thickness,
                    'pixel_spacing_x': spacing_x,
                    'pixel_spacing_y': spacing_y,
                    'reconstruction_kernel': str(ds.ConvolutionKernel) if 'ConvolutionKernel' in ds else None,
                    'slice_count': slice_count
                }
        except Exception:
            continue
            
    return best_uid, best_meta

def get_exact_dicom_metadata(patient_dir, target_uid):
    """
    Finds the exact folder for a specific UID (XML Match) and extracts metadata.

    Args:
        patient_dir (Path): The base directory containing the patient's DICOMs.
        target_uid (str): The target SeriesInstanceUID to find.

    Returns:
        dict or None: A dictionary containing DICOM metadata (spacing, thickness, 
        kernel, slice count), or None if the UID is not found.
    """
    for root_dir, dirs, files in os.walk(patient_dir):
        dicom_files = [f for f in files if f.endswith('.dcm')]
        if not dicom_files: continue
            
        try:
            ds = pydicom.dcmread(Path(root_dir) / dicom_files[0], stop_before_pixels=True)
            if clean_uid(ds.SeriesInstanceUID) == target_uid:
                spacing_y, spacing_x = None, None
                if 'PixelSpacing' in ds:
                    spacing_y = float(ds.PixelSpacing[0])
                    spacing_x = float(ds.PixelSpacing[1])
                    
                return {
                    'slice_thickness': float(ds.SliceThickness) if 'SliceThickness' in ds else None,
                    'pixel_spacing_x': spacing_x,
                    'pixel_spacing_y': spacing_y,
                    'reconstruction_kernel': str(ds.ConvolutionKernel) if 'ConvolutionKernel' in ds else None,
                    'slice_count': len(dicom_files)
                }
        except Exception:
            continue
    return None

def main():
    """
    Executes the central manifest creation pipeline.

    The function performs the following steps:
    1. Loads the clinical master list to establish the patient cohort and histology.
    2. Parses all XML files to pre-load ground-truth coordinates and target UIDs.
    3. Iterates over all patients, checking for the presence of XML data.
       - If XML exists: Extracts exact metadata from the corresponding DICOM folder.
       - If no XML exists: Runs a fallback heuristic to pick the best CT scan.
    4. Compiles all data into a standardized CSV manifest.
    5. Prints a statistical summary of the dataset.

    Returns:
        None. The generated dataset is exported to 'manifest.csv'.
    """
    print("Starting manifest creation (Full Cohort)...")
    
    # 1. Load clinical data (The master list)
    if not FILE_CLINICAL.exists():
        print(f"ERROR: Clinical CSV not found at {FILE_CLINICAL}")
        return
        
    df_clinical = pd.read_csv(FILE_CLINICAL)
    # Clean column names (removes the invisible space in 'Histology ')
    df_clinical.columns = [c.strip() for c in df_clinical.columns]
    
    # All patient IDs from the clinical file
    all_patients = df_clinical['Case ID'].dropna().unique().tolist()
    
    # Optional: Find DICOM folders of patients who are NOT in the clinical CSV at all
    # (Just to be safe, in case there are images without data)
    for p in DIR_DICOM.iterdir():
        if p.is_dir() and p.name not in all_patients and p.name != "NSCLC Radiogenomics":
            all_patients.append(p.name)

    print(f"{len(all_patients)} patients found in the master cohort.")
    
    # 2. Parse XML data in advance
    print("Parsing all XML files...")
    xml_dict = parse_all_xmls()
    
    manifest_rows = []
    
    # 3. Iterate through each patient
    for pid in tqdm(all_patients, desc="Creating patient entries"):
        
        # Extract histology
        histology = "Unknown"
        clinical_match = df_clinical[df_clinical['Case ID'] == pid]
        if len(clinical_match) > 0:
            histology = clinical_match.iloc[0]['Histology']
        
        # Base dictionary for this patient
        row = {
            'subject_id': pid,
            'histology': histology,
            'xml_present': False,
            'chosen_series_uid': None,
            'selection_reason': None,
            'slice_thickness': None,
            'pixel_spacing_x': None,
            'pixel_spacing_y': None,
            'reconstruction_kernel': None,
            'slice_count': 0,
            'qc_pass': False,
            'sop_instance_uid': None,
            'x_raw': None,
            'y_raw': None,
            'x_pixel': None,
            'y_pixel': None,
            'coordinate_mapped_successfully': False
        }
        
        # Where are the images located?
        patient_dir = DIR_DICOM / pid
        if not patient_dir.exists():
            patient_dir = DIR_DICOM / "NSCLC Radiogenomics" / pid
            
        if not patient_dir.exists():
            row['selection_reason'] = 'No_DICOM_Folder_Found'
            manifest_rows.append(row)
            continue
            
        # DOES THE PATIENT HAVE AN XML?
        if pid in xml_dict:
            row['xml_present'] = True
            row['selection_reason'] = 'XML_Ground_Truth'
            row['chosen_series_uid'] = xml_dict[pid]['series_uid']
            row['sop_instance_uid'] = xml_dict[pid]['sop_uid']
            row['x_raw'] = xml_dict[pid]['x_raw']
            row['y_raw'] = xml_dict[pid]['y_raw']
            
            # Get metadata for exactly this UID
            meta = get_exact_dicom_metadata(patient_dir, row['chosen_series_uid'])
            if meta:
                row.update(meta)
                row['qc_pass'] = True
                if row['x_raw'] is not None:
                    row['coordinate_mapped_successfully'] = True
            else:
                row['qc_pass'] = False
                row['selection_reason'] = 'XML_UID_Not_Found_On_Disk'
                
        # NO XML: FALLBACK SEARCH
        else:
            row['xml_present'] = False
            row['selection_reason'] = 'Fallback_Auto_Best_CT'
            
            fallback_uid, meta = get_best_fallback_series(patient_dir)
            if fallback_uid and meta:
                row['chosen_series_uid'] = fallback_uid
                row.update(meta)
                row['qc_pass'] = True  # Images are present, even if no label exists
        
        manifest_rows.append(row)
        
    df_manifest = pd.DataFrame(manifest_rows)
    
    # Save as European CSV format (semicolon delimited, comma for decimals)
    out_path = DIR_PROCESSED / "manifest.csv"
    df_manifest.to_csv(out_path, index=False, sep=';', decimal=',')
    
    print("\n" + "="*50)
    print("MANIFEST SUCCESSFULLY CREATED")
    print("="*50)
    print(f"Saved to: {out_path}")
    print(f"\nTotal Patients: {len(df_manifest)}")
    
    print("\nStatistics by XML presence:")
    print(df_manifest['xml_present'].value_counts())
    
    print("\nHistology overview:")
    print(df_manifest['histology'].value_counts(dropna=False))
    
    print("\nQC Pass (Are usable images present?):")
    print(df_manifest['qc_pass'].value_counts())

if __name__ == "__main__":
    main()