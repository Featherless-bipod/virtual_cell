import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from mamba_ssm.modules.mamba2 import Mamba2
from tqdm import tqdm
import math

# --- Your BiMamba class (no changes) ---
class BiMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba_fwd = Mamba2(d_model=d_model)
        self.mamba_bwd = Mamba2(d_model=d_model)
    def forward(self, x):
        h_fwd = self.mamba_fwd(x)
        x_rev = torch.flip(x, dims=[1])
        h_bwd_rev = self.mamba_bwd(x_rev)
        h_bwd = torch.flip(h_bwd_rev, dims=[1])
        h_combined = h_fwd + h_bwd
        return h_combined

# -----------------------------------------------------------------
# --- MODIFIED TranscriptomePredictor class ---
# -----------------------------------------------------------------
class TranscriptomePredictor(nn.Module):
    def __init__(self, config, GENE_FEAT_DIM, pathway_features, chr_idx, locus_fourier):
        super().__init__()
        self.cfg = config
        # --- NEW ---
        # Define the padding multiple for Mamba kernel stability
        self.padding_multiple = 16 

        # --- Register buffers (all same as before) ---
        self.register_buffer('chr_idx', chr_idx)
        self.register_buffer('locus_fourier', locus_fourier)
        self.register_buffer('pathway_features', torch.tensor(pathway_features, dtype=torch.float32))
        self.register_buffer('all_gene_idx', torch.arange(config['n_genes'], dtype=torch.long))
        
        # --- Learnable Modules (all same as before) ---
        self.gene_id_emb = nn.Embedding(config['n_genes'], config["gene_identity_dim"])
        self.chr_emb = nn.Embedding(config["n_chromosomes"], config["chrom_embedding_dim"])
        self.locus_mlp = nn.Sequential(
            nn.Linear(2 * config["locus_fourier_features"], config["chrom_embedding_dim"]),
            nn.GELU(),
        )
        self.pert_emb = nn.Embedding(config["n_perturbations"], config["perturbation_dim"])
        
        # --- Projection Layers (all same as before) ---
        self.cond_proj = nn.Linear(config["perturbation_dim"], GENE_FEAT_DIM)
        self.input_proj = nn.Linear(GENE_FEAT_DIM, config["d_model"])
        self.input_norm = nn.LayerNorm(config["d_model"])


        # --- Core Model Backbone (all same as before) ---
        self.backbone = BiMamba(d_model=config["d_model"])

        # --- Prediction Heads (all same as before) ---
        if config["prediction_head"] == "linear":
            self.head = nn.Linear(config["d_model"], 1)
        elif config["prediction_head"] == "probabilistic":
            self.head = nn.Linear(config["d_model"], 2)
        else:
            raise ValueError("Unknown prediction_head in config")

    def build_gene_features(self, B, device):
        # This function is identical
        e_id = self.gene_id_emb(self.all_gene_idx)
        e_chr = self.chr_emb(self.chr_idx)
        e_locus = self.locus_mlp(self.locus_fourier)
        e_pos = torch.cat([e_chr, e_locus], dim=1)
        e_path = self.pathway_features
        self.gene_feature_cache = torch.cat([e_id, e_path, e_pos], dim=1).to(device)
        return self.gene_feature_cache.unsqueeze(0).expand(B, -1, -1)

    def forward(self, perturbation_idx):
        B, device = perturbation_idx.shape[0], perturbation_idx.device
        
        # --- 1-3. Build sequence (same as before) ---
        cond_vec = self.pert_emb(perturbation_idx)
        cond_token = self.cond_proj(cond_vec).unsqueeze(1)
        gene_matrix = self.build_gene_features(B, device)
        seq = torch.cat([cond_token, gene_matrix], dim=1)
        seq = self.input_proj(seq) # Shape is (B, 18081, 512)
        seq = self.input_norm(seq)


        # ------------------------------------
        # --- 4. NEW PADDING FIX ---
        # ------------------------------------
        # Remember the original length (e.g., 18081)
        L_original = seq.shape[1]  
        
        # Calculate how much padding we need to add to be a multiple of 128
        if L_original % self.padding_multiple != 0:
            padding_len = self.padding_multiple - (L_original % self.padding_multiple)
            
            # Create a padding tensor of zeros
            padding = torch.zeros(
                B, padding_len, self.cfg["d_model"], 
                device=device, dtype=seq.dtype
            )
            
            # Add the padding to the end of the sequence
            seq = torch.cat([seq, padding], dim=1)
            # New shape is now a "safe" multiple, e.g., (B, 18176, 512)
        
        # ------------------------------------
        # --- 5. Process with Mamba (safer now) ---
        # ------------------------------------
        output = self.backbone(seq)

        # ------------------------------------
        # --- 6. NEW UN-PADDING FIX ---
        # ------------------------------------
        # Slice the output back to the original length to remove padding tokens
        output = output[:, :L_original, :] # Shape (B, 18081, 512)
        
        # ------------------------------------
        # --- 7. Rest of the model (same as before) ---
        # ------------------------------------
        if isinstance(output, tuple):
            H = output[0] 
        else:
            H = output
        
        # Discard condition token (at position 0)
        H_genes = H[:, 1:, :] 
        out = self.head(H_genes)

        if self.cfg["prediction_head"] == "probabilistic":
            mean_log, disp_log = out.tensor_split(2, dim=-1)
            return mean_log, disp_log
        else:
            return out

class PerturbationDataset(Dataset):
    def __init__(self, adata,CONFIG):
        # --- THE FIX IS HERE ---
        # By adding .values, we convert the pandas Series to a simple NumPy array.
        # This makes indexing with `idx` safe and unambiguous.
        self.perturbations = adata.obs['perturbation_idx'].values
        
        # We should do the same for the covariates for consistency.
        
        # Use raw counts if probabilistic, otherwise use the log-normalized data in .X
        print("Dataset using log-normalized data from adata.X for linear head.")
        X = adata.X
        self.expression = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    def __len__(self): 
        return len(self.perturbations)

    def __getitem__(self, idx):
        # This code now works correctly because self.perturbations is a NumPy array,
        # so `idx` will always refer to the position.
        return {
            "perturbation_idx": torch.tensor(self.perturbations[idx], dtype=torch.long),
            "target_expression": torch.tensor(self.expression[idx], dtype=torch.float32).unsqueeze(-1)
        }