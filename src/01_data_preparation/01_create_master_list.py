"""
Data Mapping and Master List Creation Script.

This script aggregates clinical data, XML annotation files, and DICOM metadata
to create a consolidated master list (`patient_overview.csv`). It performs 
robust checks to ensure DICOM paths exist on the local disk and applies fixes 
for known formatting issues in the TCIA metadata files.
"""

import pandas as pd
import os
from pathlib import Path

# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent 

DIR_RAW = PROJECT_ROOT / "data" / "raw"
PATH_CLINICAL = DIR_RAW / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
PATH_METADATA = DIR_RAW / "metadata.csv"
DIR_XML = DIR_RAW / "xml"
DIR_DICOM_ROOT = DIR_RAW / "dicom" 

PATH_OUTPUT = PROJECT_ROOT / "data" / "processed" / "patient_overview.csv"

def main():
    """
    Executes the data mapping pipeline to create a master patient overview.

    The function performs the following steps:
    1. Loads clinical data to retrieve base 'Subject ID' and 'Histology'.
    2. Scans the local XML directory to identify which patients have annotations.
    3. Loads DICOM metadata, corrects potential column shifts (where UIDs are 
       listed instead of Subject IDs), and verifies the physical existence of 
       the DICOM directories on the hard drive.
    4. Merges clinical, XML, and DICOM data into a single master DataFrame.
    5. Exports the compiled dataset to 'patient_overview.csv'.

    Returns:
        None. The result is written to PATH_OUTPUT.
    """
    print(f"--- Start Data Mapping (Robust Mode) ---")
    
    # 1. CLINICAL DATA
    df_clinical = pd.read_csv(PATH_CLINICAL)
    df_clinical.columns = df_clinical.columns.str.strip()
    df_labels = df_clinical[['Case ID', 'Histology']].copy().rename(columns={'Case ID': 'Subject ID'})
    print(f"-> {len(df_labels)} patients in Clinical CSV.")

    # 2. XML DATA
    xml_ids = [f.stem for f in DIR_XML.glob("*.xml")]
    df_xml = pd.DataFrame({'Subject ID': xml_ids, 'Has_XML': True})
    print(f"-> {len(df_xml)} XML files found.")

    # 3. DICOM DATA (Fix for shifted Subject ID)
    df_meta = pd.read_csv(PATH_METADATA)
    df_meta.columns = df_meta.columns.str.strip()
    
    # TCIA Metadata Fix: Sometimes 'Subject ID' is not the column containing 'AMC-001'
    id_col = 'Subject ID'
    if df_meta[id_col].iloc[0].startswith('1.3.6.'):
        print("DEBUG: Subject ID column contains UIDs, attempting fix...")
        df_meta['Real_ID'] = df_meta['Subject ID'] 
    else:
        df_meta['Real_ID'] = df_meta['Subject ID']

    # Filter for CT
    df_ct = df_meta[df_meta['Modality'].str.contains('CT', na=False)].copy()
    
    def check_path_exists(path_str):
        """
        Validates if a DICOM directory exists on the local file system.

        Args:
            path_str (str): The relative path string from the metadata.

        Returns:
            bool: True if the directory exists locally, False otherwise.
        """
        if pd.isna(path_str): return False
        clean_path_str = path_str.lstrip('./').lstrip('.\\').replace('\\', os.sep).replace('/', os.sep)
        full_path = DIR_DICOM_ROOT / clean_path_str
        return full_path.exists()

    df_ct['Path_Exists'] = df_ct['File Location'].apply(check_path_exists)
    
    available_ct_ids = df_ct[df_ct['Path_Exists'] == True]['Subject ID'].unique()
    
    # Fix: If the IDs in the metadata are UIDs, we extract the ID from the path
    if len(available_ct_ids) > 0 and str(available_ct_ids[0]).startswith('1.3.6.'):
        print("Extracting patient IDs from folder paths...")
        
        def extract_id(path_str):
            """
            Extracts the standard patient ID (e.g., 'AMC-001') from a folder path.

            Args:
                path_str (str): The raw directory path string.

            Returns:
                str: The extracted patient ID, or the original path string if no 
                matching pattern ('AMC-' or 'R01-') is found.
            """
            parts = path_str.replace('\\', '/').split('/')
            for p in parts:
                if p.startswith('AMC-') or p.startswith('R01-'):
                    return p
            return path_str
            
        available_ct_ids = df_ct[df_ct['Path_Exists'] == True]['File Location'].apply(extract_id).unique()

    df_dicom = pd.DataFrame({'Subject ID': available_ct_ids, 'Has_DICOM': True})
    print(f"-> {len(df_dicom)} patients with valid CT paths found.")

    # 4. MERGE
    all_ids = sorted(list(set(df_labels['Subject ID']) | set(xml_ids) | set(available_ct_ids)))
    df_master = pd.DataFrame({'Subject ID': all_ids})
    df_master = df_master.merge(df_labels, on='Subject ID', how='left')
    df_master = df_master.merge(df_xml, on='Subject ID', how='left')
    df_master = df_master.merge(df_dicom, on='Subject ID', how='left')
    
    df_master['Has_XML'] = df_master['Has_XML'].fillna(False)
    df_master['Has_DICOM'] = df_master['Has_DICOM'].fillna(False)

    # 5. RESULT
    df_ready = df_master[(df_master['Histology'].notna()) & (df_master['Has_XML']) & (df_master['Has_DICOM'])]
    print("-" * 30)
    print(f"COMPLETE DATASETS (Ready for ML): {len(df_ready)}")
    print("-" * 30)

    PATH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(PATH_OUTPUT, index=False)

if __name__ == "__main__":
    main()