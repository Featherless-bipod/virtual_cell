import scanpy as sc

print("hello")

adata = sc.read_h5ad('../vcc_data/adata_Training.h5ad')

print(adata.obs['perturbation_idx'])