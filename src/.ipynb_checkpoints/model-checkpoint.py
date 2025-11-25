import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from mamba_ssm.modules.mamba2 import Mamba2
from tqdm import tqdm

class BiMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba_fwd = Mamba2(d_model=d_model)
        self.mamba_bwd = Mamba2(d_model=d_model)
        self.linear = nn.Linear(d_model,d_model) 
    def forward(self, x):
        print("0")
        h_fwd = self.mamba_fwd(x)
        
        print("1")
        # 2. Backward pass
        x_rev = torch.flip(x, dims=[1])
        print("2")
        h_bwd_rev = self.mamba_bwd(x_rev)
        print("3")
        h_bwd = torch.flip(h_bwd_rev, dims=[1])
        print("4")

        # 3. Combine the outputs
        h_combined = h_fwd + h_bwd
        
        return h_combined

class TransformerBackbone(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
    def forward(self, x):
        return self.transformer_encoder(x)

class Linear(nn.Module):
    def __init(self,d_model):
        super().__init()
        self.linear = nn.Linear(d_model,d_model)
    def foward(self,x):
        out = self.linear(x)
        return out

    
class TranscriptomePredictor(nn.Module):
    def __init__(self, config, GENE_FEAT_DIM, pathway_features, chr_idx, locus_fourier):
        super().__init__()
        self.cfg = config

        # --- Register buffers for fixed, non-trainable data ---
        self.register_buffer('chr_idx', chr_idx)
        self.register_buffer('locus_fourier', locus_fourier)
        self.register_buffer('pathway_features', torch.tensor(pathway_features, dtype=torch.float32))
        self.register_buffer('all_gene_idx', torch.arange(config['n_genes'], dtype=torch.long)) #all the different genes that have their expressions measuresd
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
        self.backbone = BiMamba(d_model=config["d_model"])
        #self.backbone = TransformerBackbone(
            #d_model=config["d_model"],
            #nhead=config["n_heads"],
            #num_layers=config["n_layers"]
        #)
        #self.backbone = nn.LSTM(config['d_model'],config['d_model'], num_layers=config['mamba_layers'],batch_first =True, bidirectional=True  )

        # --- Prediction Heads ---
        if config["prediction_head"] == "linear":
            self.head = nn.Linear(config["d_model"], 1)
        elif config["prediction_head"] == "probabilistic":
            self.head = nn.Linear(config["d_model"], 2) # Outputs log(mean) and log(dispersion)
        else:
            raise ValueError("Unknown prediction_head in config")

    def build_gene_features(self, B, device):
        """Helper to construct the full (B, G, D_feat) gene feature matrix."""
        #gpuuuuuu
        e_id = self.gene_id_emb(self.all_gene_idx) #(n_gene,189)
        e_chr = self.chr_emb(self.chr_idx) #(n_gene,16)
        e_locus = self.locus_mlp(self.locus_fourier) #(n_gene,16)
        e_pos = torch.cat([e_chr, e_locus], dim=1) #(n_gene, 32)
        e_path = self.pathway_features
            
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

        print(f"-------shape of seq{seq.shape}-------")

        # 4. Process with Mamba Backbone
        output = self.backbone(seq)
        
        H = output

        H_genes = H[:, 1:, :] # Discard condition token output -> (B, G, d_model)

        # 5. Prediction Head
        out = self.head(H_genes)

        if self.cfg["prediction_head"] == "probabilistic":
            mean_log, disp_log = out.tensor_split(2, dim=-1)
            return mean_log, disp_log # Return the parameters directly
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