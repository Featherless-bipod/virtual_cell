from mamba_ssm import Mamba
import torch


class BidirectionalMamba(nn.Module):
    """
    A wrapper for a bidirectional Mamba model.
    
    This module contains two independent Mamba blocks: one for the forward pass
    and one for the backward pass.
    """
    def __init__(self, d_model, n_layers):
        super().__init__()

        self.mamba_fwd = Mamba(d_model=d_model)
        self.mamba_bwd = Mamba(d_model=d_model)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, L, D)
        """
        # 1. Forward pass
        h_fwd = self.mamba_fwd(x) # (B, L, D)

        # 2. Backward pass
        x_rev = torch.flip(x, dims=[1])
        h_bwd_rev = self.mamba_bwd(x_rev)
        h_bwd = torch.flip(h_bwd_rev, dims=[1]) # (B, L, D)

        # 3. Combine the outputs
        h_combined = h_fwd + h_bwd
        
        return h_combined