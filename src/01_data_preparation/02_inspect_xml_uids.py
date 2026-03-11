"""
XML UID Extraction and DICOM Mapping Script.

This script parses AIM v4 XML annotation files to extract the exact 
'SeriesInstanceUID' used by the radiologist. It then matches these UIDs 
against the dataset's 'metadata.csv' to locate the corresponding physical 
DICOM directories on the local file system. It includes a specific fix 
for parsing errors caused by shifted columns in the TCIA metadata file.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_XML = PROJECT_ROOT / "data" / "raw" / "xml"
PATH_METADATA = PROJECT_ROOT / "data" / "raw" / "metadata.csv"

def extract_series_uid_robust(xml_path):
    """
    Parses an XML file to robustly extract the SeriesInstanceUID.

    This function navigates the XML tree, handling potential XML namespace 
    inconsistencies by stripping them out, to find the 'imageSeries' node 
    and its underlying 'instanceUid'.

    Args:
        xml_path (Path): The file path to the AIM v4 XML annotation file.

    Returns:
        str or None: The extracted Series UID string if successfully found, 
        otherwise None.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            tag_clean = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_clean == 'imageSeries':
                for child in elem:
                    child_tag_clean = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag_clean == 'instanceUid':
                        return child.attrib.get('root')
    except Exception as e:
        print(f"Error processing {xml_path.name}: {e}")
        return None
    return None

def main():
    """
    Executes the XML parsing and DICOM metadata matching pipeline.

    The function performs the following operations:
    1. Iterates through all XML files in the raw data directory.
    2. Extracts the target Series UID from each XML file.
    3. Loads the 'metadata.csv' and applies a fix for a known issue where 
       Pandas incorrectly sets the UID column as the DataFrame index.
    4. Merges the extracted XML UIDs with the fixed metadata to find the 
       correct local folder paths ('File Location').
    5. Saves the successfully mapped records to 'exact_image_mapping.csv'.

    Returns:
        None. The mapping result is exported to disk.
    """
    print("Starting robust XML deep scan (AIM v4)...")
    
    xml_files = list(DIR_XML.glob("*.xml"))
    results = []
    for xml_file in xml_files:
        patient_id = xml_file.stem 
        series_uid = extract_series_uid_robust(xml_file)
        results.append({
            'Subject ID': patient_id,
            'XML_File': xml_file.name,
            'Linked_Series_UID': series_uid
        })
    
    df_xml_mapping = pd.DataFrame(results)
    df_xml_mapping['Linked_Series_UID'] = df_xml_mapping['Linked_Series_UID'].astype(str).str.strip()
    
    print(f"XML scan complete. Unique UIDs found: {df_xml_mapping['Linked_Series_UID'].nunique()}")
    
    print("Comparing with metadata.csv...")
    if not PATH_METADATA.exists():
        print("ERROR: Metadata CSV not found.")
        return

    # --- THE FIX ---
    # Read the CSV normally (Pandas incorrectly sets the UID as the index)
    df_meta = pd.read_csv(PATH_METADATA)
    
    # Retrieve the UID from the index back into a real column
    df_meta['Series UID Fix'] = df_meta.index.astype(str).str.strip()
    
    df_meta['File Location'] = df_meta['File Location'].astype(str).str.strip()

    print(f"Metadata loaded. First UID (Fix): {df_meta['Series UID Fix'].iloc[0]}")

    # Merge on the new, fixed column
    df_final_mapping = df_xml_mapping.merge(
        df_meta[['Series UID Fix', 'File Location']], 
        left_on='Linked_Series_UID', 
        right_on='Series UID Fix', 
        how='left'
    )
    
    matched = df_final_mapping['File Location'].notna().sum()
    
    print("-" * 30)
    print("MATCHING RESULT:")
    print(f"Total XMLs: {len(df_xml_mapping)}")
    print(f"Successfully linked with DICOM: {matched}")
    print("-" * 30)
    
    if matched > 0:
        out_path = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_final_mapping.to_csv(out_path, index=False)
        print(f"Mapping saved to: {out_path}.")
    else:
        print("Still 0 matches. Check the UIDs in the debug print above.")

if __name__ == "__main__":
    main()