import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


def get_control_data(adata):
    is_control = adata.obs['target_gene'] == "non-targeting"
    control_adata = adata[is_control].copy()
    return control_adata

def clean_and_preprocess_data(adata: sc.AnnData, 
                                         mad_cutoff=3.0) -> sc.AnnData:
    """
    Performs an ADAPTIVE, data-driven pre-processing workflow with the corrected
    sequential filtering calls.
    """
    print("--- Starting ADAPTIVE Data Cleanup and Pre-processing ---")
    print(f"Initial shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # --- The rest of the function is the same ---
    adata.layers['counts'] = adata.X.copy()
    print("Step 5: Saved original raw counts to adata.layers['counts']")
    
    print("\n--- Normalizing and Transforming Data ---")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Steps 6 & 7: Normalization and Log1p transformation complete.")
    
    print("\n--- Pre-processing Complete. ---")
    return adata