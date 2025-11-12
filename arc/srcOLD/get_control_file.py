import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


def get_control_data(adata):
    is_control = adata.obs['target_gene'] == "non-targeting"
    control_adata = adata[is_control].copy()
    return control_adata