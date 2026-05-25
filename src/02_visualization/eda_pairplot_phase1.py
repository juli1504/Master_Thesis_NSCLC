import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def generate_pairplot_1b():
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
    
    df = pd.read_csv(FILE_CLINICAL)
    
    # Phase 1b Features (Enriched)
    features = ['Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Quit Smoking Year']
    target_col = next((c for c in df.columns if c.strip().lower() == 'histology'), None)
    
    # Clean all numeric features
    for col in features:
        df[col] = pd.to_numeric(df[col].replace(['Not Collected', 'Unknown', ' '], np.nan), errors='coerce')
    
    df = df.dropna(subset=[target_col] + features)
    
    # Plot configuration
    sns.set_theme(style="ticks", font_scale=1.0)
    g = sns.pairplot(df, vars=features, hue=target_col, palette='magma', 
                     diag_kind='kde', height=3, aspect=1)
    
    plt.suptitle("Phase 1b: Enriched Clinical Interactions", y=1.02, fontsize=18)
    g.fig.set_size_inches(15, 15)
    
    save_path = PROJECT_ROOT / "results" / "figures" / "pairplot_1b.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Phase 1b pairplot saved to {save_path}")

if __name__ == "__main__":
    generate_pairplot_1b()