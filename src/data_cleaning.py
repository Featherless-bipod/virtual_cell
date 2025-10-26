import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


def get_control_data(adata):
    is_control = adata.obs['target_gene'] == "non-targeting"
    control_adata = adata[is_control].copy()
    return control_adata

def adaptive_clean_and_preprocess_data(adata: sc.AnnData, 
                                         mad_cutoff=3.0) -> sc.AnnData:
    """
    Performs an ADAPTIVE, data-driven pre-processing workflow with the corrected
    sequential filtering calls.
    """
    print("--- Starting ADAPTIVE Data Cleanup and Pre-processing ---")
    print(f"Initial shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # --- Step 1 & 2: Calculate and Visualize QC ---
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    print("Step 1 & 2: Calculated and visualizing QC metrics...")
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], 
                  jitter=0.4, multi_panel=True, show=True)
    
    # --- Step 3: Calculate Adaptive Thresholds ---
    print("\nStep 3: Calculating adaptive filtering thresholds...")
    min_genes_per_cell = 200 
    median_genes = adata.obs.n_genes_by_counts.median()
    mad_genes = (adata.obs.n_genes_by_counts - median_genes).abs().median()
    max_genes_per_cell = median_genes + mad_cutoff * mad_genes
    min_cells_per_gene = 3
    print(f"  - Using min_genes_per_cell: {min_genes_per_cell}")
    print(f"  - Calculated max_genes_per_cell: {max_genes_per_cell:.0f}")

    # --- Step 4: Apply Filters Sequentially (THE FIX) ---
    print("\nStep 4: Applying filters sequentially...")
    
    # First, filter for minimum genes
    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    
    # Second, filter for maximum genes
    sc.pp.filter_cells(adata, max_genes=max_genes_per_cell)
    
    # Third, filter for minimum cells per gene
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    
    print(f"   Filtering complete. Shape is now: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # --- The rest of the function is the same ---
    adata.layers['counts'] = adata.X.copy()
    print("Step 5: Saved original raw counts to adata.layers['counts']")
    
    print("\n--- Normalizing and Transforming Data ---")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Steps 6 & 7: Normalization and Log1p transformation complete.")
    
    print("\n--- Pre-processing Complete. ---")
    return adata