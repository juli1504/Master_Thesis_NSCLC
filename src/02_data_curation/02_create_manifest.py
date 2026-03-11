"""
Central Data Manifest Creation Script.

This script acts as the core of the data curation pipeline. It aggregates 
the clinical ground truth (histology), the spatial XML annotations (X, Y 
coordinates and SOPInstanceUID), and the physical DICOM metadata (spacing, 
thickness, slice count, and kernels) into a single, centralized 'manifest.csv'. 

For patients without XML annotations, it implements an automated fallback 
search to locate the most suitable CT scan for potential unsupervised learning.
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
        if len(dicom_files) < 20: 
            continue
            
        try:
            ds = pydicom.dcmread(Path(root_dir) / dicom_files[0], stop_before_pixels=True)
            if 'Modality' in ds and ds.Modality != 'CT': continue
            
            desc_lower = str(ds.SeriesDescription).lower() if 'SeriesDescription' in ds else ""
            if any(x in desc_lower for x in ['scout', 'topogram', 'loc']): continue
            
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
                    
                spacing_between = float(ds.SpacingBetweenSlices) if 'SpacingBetweenSlices' in ds else None
                series_desc = str(ds.SeriesDescription) if 'SeriesDescription' in ds else None
                    
                best_meta = {
                    'slice_thickness': thickness,
                    'spacing_between_slices': spacing_between,
                    'pixel_spacing_x': spacing_x,
                    'pixel_spacing_y': spacing_y,
                    'reconstruction_kernel': str(ds.ConvolutionKernel) if 'ConvolutionKernel' in ds else None,
                    'series_description': series_desc,
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
        dict or None: A dictionary containing comprehensive DICOM metadata.
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
                
                spacing_between = float(ds.SpacingBetweenSlices) if 'SpacingBetweenSlices' in ds else None
                series_desc = str(ds.SeriesDescription) if 'SeriesDescription' in ds else None
                    
                return {
                    'slice_thickness': float(ds.SliceThickness) if 'SliceThickness' in ds else None,
                    'spacing_between_slices': spacing_between,
                    'pixel_spacing_x': spacing_x,
                    'pixel_spacing_y': spacing_y,
                    'reconstruction_kernel': str(ds.ConvolutionKernel) if 'ConvolutionKernel' in ds else None,
                    'series_description': series_desc,
                    'slice_count': len(dicom_files)
                }
        except Exception:
            continue
    return None

def main():
    """
    Executes the central manifest creation pipeline.
    """
    print("Starting manifest creation (Full Cohort)...")
    
    if not FILE_CLINICAL.exists():
        print(f"ERROR: Clinical CSV not found at {FILE_CLINICAL}")
        return
        
    df_clinical = pd.read_csv(FILE_CLINICAL)
    df_clinical.columns = [c.strip() for c in df_clinical.columns]
    all_patients = df_clinical['Case ID'].dropna().unique().tolist()
    
    for p in DIR_DICOM.iterdir():
        if p.is_dir() and p.name not in all_patients and p.name != "NSCLC Radiogenomics":
            all_patients.append(p.name)

    print(f"{len(all_patients)} patients found in the master cohort.")
    print("Parsing all XML files...")
    xml_dict = parse_all_xmls()
    
    manifest_rows = []
    
    for pid in tqdm(all_patients, desc="Creating patient entries"):
        histology = "Unknown"
        clinical_match = df_clinical[df_clinical['Case ID'] == pid]
        if len(clinical_match) > 0:
            histology = clinical_match.iloc[0]['Histology']
        
        row = {
            'subject_id': pid,
            'histology': histology,
            'xml_present': False,
            'chosen_series_uid': None,
            'selection_reason': None,
            'slice_thickness': None,
            'spacing_between_slices': None,
            'pixel_spacing_x': None,
            'pixel_spacing_y': None,
            'reconstruction_kernel': None,
            'series_description': None,
            'slice_count': 0,
            'qc_pass': False,
            'sop_instance_uid': None,
            'x_raw': None,
            'y_raw': None,
            'x_pixel': None,
            'y_pixel': None,
            'coordinate_mapped_successfully': False
        }
        
        patient_dir = DIR_DICOM / pid
        if not patient_dir.exists():
            patient_dir = DIR_DICOM / "NSCLC Radiogenomics" / pid
            
        if not patient_dir.exists():
            row['selection_reason'] = 'No_DICOM_Folder_Found'
            manifest_rows.append(row)
            continue
            
        if pid in xml_dict:
            row['xml_present'] = True
            row['selection_reason'] = 'XML_Ground_Truth'
            row['chosen_series_uid'] = xml_dict[pid]['series_uid']
            row['sop_instance_uid'] = xml_dict[pid]['sop_uid']
            row['x_raw'] = xml_dict[pid]['x_raw']
            row['y_raw'] = xml_dict[pid]['y_raw']
            
            meta = get_exact_dicom_metadata(patient_dir, row['chosen_series_uid'])
            if meta:
                row.update(meta)
                row['qc_pass'] = True
                if row['x_raw'] is not None:
                    row['coordinate_mapped_successfully'] = True
            else:
                row['qc_pass'] = False
                row['selection_reason'] = 'XML_UID_Not_Found_On_Disk'
                
        else:
            row['xml_present'] = False
            row['selection_reason'] = 'Fallback_Auto_Best_CT'
            
            fallback_uid, meta = get_best_fallback_series(patient_dir)
            if fallback_uid and meta:
                row['chosen_series_uid'] = fallback_uid
                row.update(meta)
                row['qc_pass'] = True 
        
        manifest_rows.append(row)
        
    df_manifest = pd.DataFrame(manifest_rows)
    
    out_path = DIR_PROCESSED / "manifest.csv"
    df_manifest.to_csv(out_path, index=False, sep=';', decimal=',')
    
    print("\n" + "="*50)
    print("MANIFEST SUCCESSFULLY CREATED")
    print("="*50)
    print(f"Saved to: {out_path}")
    print(f"\nTotal Patients: {len(df_manifest)}")

if __name__ == "__main__":
    main()