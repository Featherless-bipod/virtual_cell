import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scanpy as sc
import pathway_encoding as path
import position_encoding as pos


'''class Mamba(nn.Module):
    """Placeholder for the Mamba implementation. Replace with `mamba-ssm`."""
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.placeholder = nn.LSTM(d_model, d_model, n_layers, batch_first=True, bidirectional=True)
    def forward(self, x):
        y, _ = self.placeholder(x)
        d = y.shape[-1] // 2
        return y[..., :d] + y[..., d:]'''

class TranscriptomePredictor(nn.Module):
    def __init__(self, config, GENE_FEAT_DIM, pathway_features, chr_idx, locus_fourier):
        super().__init__()
        self.cfg = config

        # --- Register buffers for fixed, non-trainable data ---
        self.register_buffer('chr_idx', chr_idx)
        self.register_buffer('locus_fourier', locus_fourier)
        self.register_buffer('pathway_features', torch.tensor(pathway_features, dtype=torch.float32))
        self.register_buffer('all_gene_idx', torch.arange(config['n_genes'], dtype=torch.long))
        print(self.all_gene_idx.shape)
        # --- Learnable Modules for Feature Generation ---
        self.gene_id_emb = nn.Embedding(config['n_genes'], config["gene_identity_dim"])
        self.chr_emb = nn.Embedding(config["n_chromosomes"], config["chrom_embedding_dim"])
        self.locus_mlp = nn.Sequential(
            nn.Linear(2 * config["locus_fourier_features"], config["chrom_embedding_dim"]),
            nn.GELU(),
        )
        self.pert_emb = nn.Embedding(config["n_perturbations"], config["perturbation_dim"])
        
        # --- Projection Layers to Interface with Backbone ---
        self.cond_proj = nn.Linear(config["perturbation_dim"], GENE_FEAT_DIM)
        self.input_proj = nn.Linear(GENE_FEAT_DIM, config["d_model"])

        # --- Core Model Backbone ---
        #self.backbone = Mamba(d_model=config["d_model"], n_layers=config["mamba_layers"])
        self.backbone = nn.LSTM(config['d_model'],config['d_model'], num_layers=config['mamba_layers'],batch_first =True, bidirectional=True  )

        # --- Prediction Heads ---
        self.head = nn.Linear(config["d_model"], 1)

        
        '''
        elif config["prediction_head"] == "probabilistic":
            self.head = nn.Linear(config["d_model"], 2) # Outputs log(mean) and log(dispersion)
            self.alpha_lib = nn.Parameter(torch.zeros(1)) # Learnable library size coefficient
        else:
            raise ValueError("Unknown prediction_head in config")
        '''
        # --- Caching for efficiency ---
        self.gene_feature_cache = None

    def build_gene_features(self, B, device):
        """Helper to construct the full (B, G, D_feat) gene feature matrix."""
        #gpuuuuuu
        if self.gene_feature_cache is None:
            e_id = self.gene_id_emb(self.all_gene_idx)
            print(f"e_id shape: {e_id.shape}")
            print(f"all_gene_idx shape: {self.all_gene_idx.shape}")
            
            e_chr = self.chr_emb(self.chr_idx)
            e_locus = self.locus_mlp(self.locus_fourier)
            e_pos = torch.cat([e_chr, e_locus], dim=1)
            print(e_pos.shape)
            print(f"e_pos shape: {e_pos.shape}")
            print(f"chr_idx shape: {self.chr_idx.shape}")
            e_path = self.pathway_features
            print(e_path.shape)
            print(f"epath shape: {e_path.shape}")
            print(f"pathway_features shape: {self.pathway_features.shape}")
            
            self.gene_feature_cache = torch.cat([e_id, e_path, e_pos], dim=1).to(device)
        
        # Expand for batching
        return self.gene_feature_cache.unsqueeze(0).expand(B, -1, -1)

    def forward(self, perturbation_idx):
        B, device = perturbation_idx.shape[0], perturbation_idx.device

        # 1. Condition Token: (Perturbation + Cell Covariates) -> Projected
        cond_vec = self.pert_emb(perturbation_idx)
        cond_token = self.cond_proj(cond_vec).unsqueeze(1) # (B, 1, GENE_FEAT_DIM)

        # 2. Gene Matrix: (Identity + Pathway + Position)
        gene_matrix = self.build_gene_features(B, device) # (B, G, GENE_FEAT_DIM)

        # 3. Create Full Sequence and Project to d_model
        seq = torch.cat([cond_token, gene_matrix], dim=1) # (B, G+1, GENE_FEAT_DIM)
        seq = self.input_proj(seq)                        # (B, G+1, d_model)

        # 4. Process with Mamba Backbone
        H = self.backbone(seq)
        H_genes = H[:, 1:, :] # Discard condition token output -> (B, G, d_model)

        # 5. Prediction Head
        out = self.head(H_genes)

        #if self.cfg["prediction_head"] == "probabilistic":
            #mean_log, disp_log = out.tensor_split(2, dim=-1)
            #og_umi = cell_covariates[:, 0].view(B, 1, 1) # (B, 1, 1) library size
            # Adjust mean by library size
            #mean_log = mean_log + self.alpha_lib * log_umi
            #return mean_log, disp_log
        #else:
        return out




class PerturbationDataset(Dataset):
    def __init__(self, adata):
        self.perturbations = adata.obs['perturbation_idx']
        self.covariates = np.log1p(adata.obs[['total_counts','n_genes_by_counts']].values)
        X = adata.X
        self.expression = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    def __len__(self): return len(self.perturbations)
    def __getitem__(self, idx):
        return {
            "perturbation_idx": torch.tensor(self.perturbations[idx], dtype=torch.long),
            "target_expression": torch.tensor(self.expression[idx], dtype=torch.float32).unsqueeze(-1)
        }

def nb_nll_loss(y_true, mean_log, disp_log):
    """Numerically stable Negative Binomial Negative Log-Likelihood loss."""
    mean = torch.exp(mean_log)
    # Use softplus for safe, positive dispersion
    disp = torch.nn.functional.softplus(disp_log) + 1e-6
    logits = torch.log(mean + 1e-8) - torch.log(mean + disp + 1e-8)
    dist = torch.distributions.NegativeBinomial(total_count=disp.squeeze(-1), logits=logits.squeeze(-1))
    return -dist.log_prob(y_true.squeeze(-1)).mean()

def train_model(model, dataloader, config):
    print("\n--- Starting Model Training ---")
    model.train()
    opt = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
    
    for ep in range(config["epochs"]):
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            opt.zero_grad(set_to_none=True)
            pert_idx = batch["perturbation_idx"].to(next(model.parameters()).device)
            targets = batch["target_expression"].to(next(model.parameters()).device)

            #if config["prediction_head"] == "probabilistic":
                #mean_log, disp_log = model(pert_idx, covariates)
                #loss = nb_nll_loss(targets, mean_log, disp_log)
            #else:
            predictions = model(pert_idx)
            loss = nn.functional.mse_loss(predictions, targets)

            loss.backward()
            opt.step()
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {ep+1}/{config['epochs']}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        print(f"--- Epoch {ep+1} Average Loss: {total_loss / len(dataloader):.4f} ---")
