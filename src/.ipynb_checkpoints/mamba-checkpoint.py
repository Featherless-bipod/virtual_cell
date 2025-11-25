#from mamba_ssm import Mamba
import torch
from torch import nn


class BidirectionalMamba(nn.Module):
    """
    A wrapper for a bidirectional Mamba model.
    
    This module contains two independent Mamba blocks: one for the forward pass
    and one for the backward pass.
    """
    def __init__(self, d_model, n_layers):
        super().__init__()

        #self.mamba_fwd = Mamba(d_model=d_model)
        #self.mamba_bwd = Mamba(d_model=d_model)

        self.placeholder = nn.LSTM(
            d_model, 
            d_model, 
            num_layers=n_layers, # Using n_layers here
            batch_first=True, 
            bidirectional=True
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, L, D)
        """
        # A simple and effective way is to add them
        y, _ = self.placeholder(x)
        d_out = y.shape[-1] // 2
        return y[..., :d_out] + y[..., d_out:]
        """ # 1. Forward pass
        h_fwd = self.mamba_fwd(x) # (B, L, D)

        # 2. Backward pass
        x_rev = torch.flip(x, dims=[1])
        h_bwd_rev = self.mamba_bwd(x_rev)
        h_bwd = torch.flip(h_bwd_rev, dims=[1]) # (B, L, D)

        # 3. Combine the outputs
        h_combined = h_fwd + h_bwd
        
        return h_combined
        """
        
        