
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.std import tqdm 
import matplotlib.pyplot as plt
from IPython.display import clear_output


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
# 1. SIMPLE PLOTTING FUNCTION
# -----------------------------------------------------------------
def plot_training_loss(epoch_loss_history, avg_epoch_loss, current_epoch):
    """
    Plots the live training loss over epochs.
    """
    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(1, len(epoch_loss_history) + 1), epoch_loss_history, marker='o', linestyle='-')
    ax.set_title("Live Training Loss vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.grid(True)
    ax.set_xticks(range(1, len(epoch_loss_history) + 1))
        
    status_text = f"Latest Loss (Epoch {current_epoch + 1}): {avg_epoch_loss:.6f}"
    fig.text(0.5, 0.01, status_text, ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout(pad=3.0)
    plt.show()

# -----------------------------------------------------------------
# 2. SIMPLE TRAINING FUNCTION (Full Float32 Precision)
# -----------------------------------------------------------------
def train_model(model, dataloader, config, device):
    print("\n--- Starting Model Training (in full float32) ---")
    model.train()
    opt = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
    
    epoch_loss_history = []
    
    for ep in range(config["epochs"]):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{config['epochs']}", leave=False)
        
        for i, batch in enumerate(pbar):
            opt.zero_grad(set_to_none=True)
            pert_idx = batch["perturbation_idx"].to(device)
            targets = batch["target_expression"].to(device)

            # --- Forward Pass (in full float32) ---
            if config["prediction_head"] == "probabilistic":
                mean_log, disp_log = model(pert_idx)
                loss = nb_nll_loss(targets, mean_log, disp_log)
            else:
                # This is the path that gave you the MSE=16
                predictions = model(pert_idx)
                loss = nn.functional.mse_loss(predictions, targets)

            # --- NaN Check (still good to have) ---
            if torch.isnan(loss):
                print(f"!!! NaN loss detected at epoch {ep+1}, batch {i}. Stopping. !!!")
                return epoch_loss_history

            # --- Standard Backpropagation ---
            loss.backward()
            
            # --- Gradient clipping (still a good idea for stability) ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            
            opt.step()

            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        # --- Plotting ---
        avg_epoch_loss = total_loss / len(dataloader)
        epoch_loss_history.append(avg_epoch_loss)

        # Call the simple plotting function
        plot_training_loss(epoch_loss_history, avg_epoch_loss, ep)

    print("\n--- Training Complete ---")
    
    print("Saving trained model weights...")
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model saved to model_weights.pth")

    return epoch_loss_history