# Multimodal Subtype Classification of Non-Small Cell Lung Cancer (NSCLC)

## Integrating CT Imaging and Structured Radiological Data Using Machine Learning

**Status:** In Progress  
**Institution:** University of Mannheim  
**Faculty:** School of Business Informatics and Mathematics

### Overview
This repository contains the source code and documentation for the Master's Thesis titled **"Multimodal Subtype Classification of Non-Small Cell Lung Cancer (NSCLC): Integrating CT Imaging and Structured Radiological Data Using Machine Learning"**.

The primary objective of this research is to develop a deep learning framework that performs non-invasive histological subtype classification (Adenocarcinoma vs. Squamous Cell Carcinoma). The proposed architecture utilizes a **Late Fusion** strategy to integrate high-dimensional visual features extracted from volumetric CT scans with semantic features derived from structured radiologist annotations.

### Dataset and Access
The project utilizes the **NSCLC Radiogenomics** dataset, which is publicly available via The Cancer Imaging Archive (TCIA).

**Data Source:** [NSCLC Radiogenomics on TCIA](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiogenomics)  
**Cohort Size:** 211 subjects

#### Instructions for Reproduction
This repository does not contain the raw medical imaging data due to repository size limitations. To reproduce the results, please follow these steps:

1.  **Download Data:**
    * Visit the TCIA link above.
    * Download the "Images (DICOM)" and the "Semantic Annotations (XML)" using the NBIA Data Retriever.
    * Download the "Clinical Data" (CSV).

2.  **Organize Directory:**
    Place the downloaded files into the local `data/` directory following this structure:
    ```text
    data/
    ├── raw/
    │   ├── dicom/          # "NSCLC Radiogenomics" folder
    │   ├── xml/            # XML files
    │   └── clinical/       # CSV file
    ```

3.  **Run Mapping Script:**
    Execute `notebooks/02_mapping/create_master_list.ipynb` to link the modalities.

### Methodology
The implementation is divided into three phases:

1.  **Data Mapping and Preprocessing:**
    * Linkage of patients across DICOM, XML, and Clinical files.
    * Preprocessing of CT volumes (windowing, resampling, cropping).
    * Parsing and encoding of XML annotation features.

2.  **Unimodal Baselines:**
    * **Semantic Model:** Machine Learning classifiers (XGBoost/Random Forest) trained on structured XML features.
    * **Visual Model:** 3D Convolutional Neural Networks trained on voxel data.

3.  **Multimodal Fusion:**
    * Implementation of a Late Fusion architecture concatenating feature vectors from both unimodal streams.

### Repository Structure

```text
.
├── data/                    # Local directory for dataset (excluded from version control)
├── notebooks/
│   ├── 01_exploration/      # Initial data analysis scripts
│   ├── 02_mapping/          # Scripts for linking data modalities
│   └── 03_preprocessing/    # Pipelines for DICOM and XML processing
├── src/                     # Source code for model architectures
├── results/                 # Evaluation metrics
├── requirements.txt         # Python dependencies
└── README.md
