import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PathwayAutoencoder(nn.Module):
    def __init__(self, n_cells, pathway_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_cells, 512), nn.ReLU(), nn.Linear(512, pathway_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(pathway_dim, 512), nn.ReLU(), nn.Linear(512, n_cells)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def precompute_pathway_features(control_adata, config,epochs=100):
    print("--- Precomputing pathway features on control data ---")
    X = control_adata.X.toarray() if hasattr(control_adata.X, "toarray") else np.asarray(control_adata.X)
    X_gc = torch.tensor(X.T, dtype=torch.float32) # Transpose to (Genes, Cells)

    ae = PathwayAutoencoder(n_cells=X_gc.shape[1], pathway_dim=config["pathway_dim"])
    opt = optim.Adam(ae.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        ae.train()
        opt.zero_grad()
        z, recon = ae(X_gc)
        loss = loss_fn(recon, X_gc)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"AE epoch {ep+1}/{epochs} | recon MSE: {loss.item():.4f}")


    ae.eval()
    with torch.no_grad():
        pathway_features, _ = ae(X_gc)
    print(f"Generated pathway_features shape: {pathway_features.shape}")
    return pathway_features.cpu().numpy()