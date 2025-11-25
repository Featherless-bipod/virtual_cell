from IPython.display import clear_output, display
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
print("torch imported")
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
print("scanpy imported")
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pyensembl
import time
print("pyensembl imported")
import src.model as m
import src.data_cleaning as dc
import src.position_encoding as pos
import src.pathway_encoding as path
import src.training as train
print("------importing pyensembl-----")

ensembl_data = pyensembl.EnsemblRelease(109)
ensembl_data.download()
ensembl_data.index()

adata = sc.read_h5ad('vcc_data/adata_Training.h5ad')
#adata.obs['perturbation_idx'] = np.random.randint(0, CONFIG["n_perturbations"], size=adata.n_obs)
gene_names = pd.read_csv('vcc_data/gene_names.csv', header = None)
gene_names = gene_names.iloc[:, 0].tolist()


CONFIG = {
    "n_genes": len(gene_names), #total number of expression genes needed to predict
    "n_perturbations": len(adata.obs['target_gene'].unique().tolist()), #number of unique perturbations
    "n_chromosomes": 25, #chromosome number 23+X+Y

    "perturbation_dim": 256,      # Condition embedding
    "chrom_embedding_dim": 16,     # Learned in-model for chromosome identity
    "locus_fourier_features": 8,   # Number of Fourier frequency pairs (2*F)
    "pathway_dim": 50,             # From pre-trained Autoencoder(based on hallmark MSigDB)
    "gene_identity_dim": 189,       # Main learnable gene embedding

    # Backbone dims
    "d_model": 512,                # Mamba hidden size
    "mamba_layers": 4,
    "n_heads": 8,
    "n_layers": 4, 

    # Head
    "prediction_head": "probabilistic", # "linear" | "probabilistic"

    # Training
    "batch_size": 64,
    "learning_rate": 5e-5, # Lowered LR for AdamW stability
    "epochs": 100,
}

# Derived dimensions for clarity
# Positional encoding dim = chromosome embedding + MLP output from Fourier features
POS_DIM = CONFIG["chrom_embedding_dim"] + CONFIG["chrom_embedding_dim"]
GENE_FEAT_DIM = CONFIG["gene_identity_dim"] + CONFIG["pathway_dim"] + POS_DIM

unique_perturbations = adata.obs['target_gene'].unique().tolist()
perturbation_to_idx_map = {name: i for i, name in enumerate(unique_perturbations)}
adata.obs['perturbation_idx'] = adata.obs['target_gene'].map(perturbation_to_idx_map)
control_adata = dc.get_control_data(adata)
chr_idx, locus_norm, locus_fourier = pos.precompute_positional_indices(ensembl_data, gene_names, CONFIG)

pathway_feats = path.precompute_pathway_features(control_adata, CONFIG)

device = torch.device("cuda")
print(f"\nUsing device: {device}")
model = m.TranscriptomePredictor(CONFIG, GENE_FEAT_DIM, pathway_feats, chr_idx, locus_fourier)
model.to(device)

dataset = m.PerturbationDataset(adata, CONFIG)
train_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

loss_history = train.train_model(model, train_loader, CONFIG,device)
