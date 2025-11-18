import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.std import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.amp import autocast, GradScaler


def nb_nll_loss(y_true, mean_log, disp_log):
    """Numerically stable Negative Binomial Negative Log-Likelihood loss."""
    mean = torch.exp(mean_log)
    # Use softplus for safe, positive dispersion
    disp = torch.nn.functional.softplus(disp_log) + 1e-6
    
    # --- THIS IS THE FIX ---
    # Calculate the probability 'p' directly
    # p = r / (r + mu)
    # Add epsilon for numerical stability
    probs = (disp + 1e-8) / (disp + mean + 1e-8)

    # Use 'probs' instead of 'logits'
    dist = torch.distributions.NegativeBinomial(total_count=disp.squeeze(-1), probs=probs.squeeze(-1))
    
    return -dist.log_prob(y_true.squeeze(-1)).mean()

# -----------------------------------------------------------------
# 1. PLOTTING FUNCTION
# (Fixed to accept 'current_epoch')
# -----------------------------------------------------------------

def plot_training_loss(epoch_loss_history, avg_epoch_loss, current_epoch):
    """
    Plots the training loss over epochs.
    
    Args:
        epoch_loss_history (list): A list of the average loss for each epoch.
        avg_epoch_loss (float): The average loss from the most recent epoch.
        current_epoch (int): The current epoch number (e.g., 0, 1, 2...).
    """
    # Clear the output of the current cell to prepare for the new plot
    clear_output(wait=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(epoch_loss_history) + 1), epoch_loss_history, marker='o', linestyle='-')
    ax.set_title("Live Training Loss vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.grid(True)
    ax.set_xticks(range(1, len(epoch_loss_history) + 1))
        
    # Use the 'current_epoch' argument (we add 1 for display)
    status_text = f"Latest Loss (Epoch {current_epoch + 1}): {avg_epoch_loss:.6f}"
    fig.text(0.5, 0.01, status_text, ha='center', va='bottom', fontsize=12)
        
    # Adjust layout to prevent the text from being cut off
    plt.tight_layout(pad=3.0)
        
    # Display the updated plot
    plt.show()

# -----------------------------------------------------------------
# 2. TRAINING FUNCTION
# (Fixed to prevent NaN loss with AMP)
# -----------------------------------------------------------------

def train_model(model, dataloader, config, device):

    print("\n--- Starting Model Training ---")
    model.train()
    opt = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)

    epoch_loss_history = []

    # --- FIX 1: Update GradScaler syntax ---
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"])
    for ep in range(config["epochs"]):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{config['epochs']}", leave=False)

        for i, batch in enumerate(pbar):
            opt.zero_grad(set_to_none=True)
            pert_idx = batch["perturbation_idx"].to(device)
            targets = batch["target_expression"].to(device)

            # --- FIX 2: Only 'autocast' the model's forward pass ---
            with autocast(device_type = device.type):
                if config["prediction_head"] == "probabilistic":
                    mean_log, disp_log = model(pert_idx)
                else:
                    predictions = model(pert_idx)

            # --- FIX 3: Run the loss function OUTSIDE autocast ---
            if config["prediction_head"] == "probabilistic":
                loss = nb_nll_loss(targets, mean_log.float(), disp_log.float())
            else:
                loss = nn.functional.mse_loss(predictions.float(), targets)

            # --- NaN Check ---
            if torch.isnan(loss):
                print(f"!!! NaN loss detected at epoch {ep+1}, batch {i}. Stopping. !!!")
                return epoch_loss_history # Stop training

            # --- Backpropagation ---
            scaler.scale(loss).backward()
            # --- FIX 4: Add Gradient Clipping ---
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
            

        scheduler.step()
        torch.save(model.state_dict(), "drive/MyDrive/Projects/MambaPerturb/virtual_cell/model_weights2.pth")
        # --- Plotting ---
        avg_epoch_loss = total_loss / len(dataloader)
        epoch_loss_history.append(avg_epoch_loss)

        # Call the plotting function and pass 'ep'
        plot_training_loss(epoch_loss_history, avg_epoch_loss, ep)

    print("\n--- Training Complete ---")
    print("Saving trained model weights...")

    torch.save(model.state_dict(), "model_weights.pth")

    print("Model saved to model_weights.pth")
    return epoch_loss_history