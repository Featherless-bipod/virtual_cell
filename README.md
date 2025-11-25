# Mamba-based Transcriptome Perturbation Predictor

## 1. Project Overview

This project implements a state-of-the-art deep learning model to predict genome-wide gene expression changes in response to chemical or genetic perturbations. The architecture is a conditional sequence model that treats the transcriptome as a single, long sequence of genes, enabling it to learn complex, long-range gene-gene dependencies.

The core of the model is a `BiMamba` (bidirectional Mamba) backbone, which is specifically chosen for its ability to efficiently process ultra-long sequences (18,000+ genes) with linear-time complexity ($O(L)$), avoiding the quadratic ($O(L^2)$) bottleneck of standard Transformers.

This model is designed for tasks like the "Virtual Cell Challenge," where the goal is to predict the full transcriptomic state resulting from a novel perturbation.

## 2. Model Architecture

The model architecture is "BERT-like," processing an input sequence of length `G+1`, where `G` is the number of genes and `+1` is a special **Condition Token**.

1.  **Condition Token (`[COND]`):** A learnable embedding vector (`perturbation_dim`) that represents the specific perturbation (e.g., a SMILES string or gene knockdown).
2.  **Gene Tokens (`[GENE_1]...[GENE_G]`):** A sequence of all genes in the transcriptome.

The model's objective is to learn how the `[COND]` token's information modulates the state of every gene in the sequence.

### 2.1. Feature Engineering

The representation for each gene token is a rich, concatenated vector composed of:

* **Learnable Gene Identity:** A trainable `nn.Embedding` (`gene_identity_dim`) that allows the model to learn a unique "fingerprint" for each gene.
* **Genomic Position Encoding:** A fixed positional encoding derived from:
    * A learnable embedding for the chromosome (`chr_emb`).
    * A Fourier feature-based encoding of the gene's locus (`locus_fourier`), which is passed through an MLP.
* **Precomputed Pathway Features:** A feature vector (`pathway_dim`) from a `PathwayAutoencoder` pre-trained *only* on control cell data. This provides a stable, data-driven baseline of each gene's typical co-expression and pathway relationships.

### 2.2. Prediction Head

The model supports two output heads, selectable in the `CONFIG`:

1.  **Linear Head (`prediction_head: "linear"`)**
    * **Output:** A single value per gene (predicting log-normalized expression).
    * **Loss:** `nn.MSELoss()`.
    * **Data:** Trained on `adata.X` (log-normalized data).

2.  **Probabilistic Head (`prediction_head: "probabilistic"`)**
    * **Output:** Two values per gene: `mean_log` and `disp_log`.
    * **Loss:** `nb_nll_loss` (Negative Binomial Negative Log-Likelihood).
    * **Data:** Trained on raw, integer counts (e.g., from `adata.layers['counts']` or `adata.raw.X`).

## 3. Setup & Installation

The environment is managed with `conda`.

1.  **Create a new conda environment:** This command installs all necessary packages from the correct channels to avoid dependency conflicts.
    ```bash
    # Create the new environment (e.g., 'cell_env')
    conda create -n cell_env -c pytorch -c conda-forge python=3.10

    # Activate the environment
    conda activate cell_env

    # Install core packages
    # (Specify your CUDA version, e.g., pytorch-cuda=11.8 or 12.1)
    conda install -c pytorch pytorch torchvision torchaudio pytorch-cuda=11.8
    
    # Install scientific stack
    conda install -c conda-forge scanpy anndata numpy pandas matplotlib tqdm ipywidgets scikit-learn
    
    # Install Mamba (requires pip)
    pip install mamba-ssm causal-conv1d
    ```

2.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

## 4. Usage / Workflow

The pipeline is split into preprocessing, training, and evaluation.

### Step 1: Preprocessing

1.  **Load Data:** Load your `AnnData` object (e.g., `adata_Training.h5ad`).
2.  **Create Perturbation Index:** Create the `adata.obs['perturbation_idx']` column with integer IDs for each unique perturbation.
3.  **Ensure Data Types (CRITICAL):**
    * If using the **Linear Head**, ensure `adata.X` contains log-normalized data.
    * If using the **Probabilistic Head**, ensure the raw integer counts are available in `adata.layers['counts']` or `adata.raw.X`. The `PerturbationDataset` class will automatically find this data.
4.  **Split Data:** Create training and validation sets:
    ```python
    from sklearn.model_selection import train_test_split
    
    indices = adata.obs.index
    train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_adata = adata[train_idx, :].copy()
    valid_adata = adata[valid_idx, :].copy()
    ```
5.  **Precompute Control Features:**
    ```python
    import src.position_encoding as pos
    import src.pathway_encoding as path
    
    control_adata = adata[adata.obs['cell_type'] == 'control']
    
    # Run 1: Get positional features
    chr_idx, locus_norm, locus_fourier = pos.precompute_positional_indices(gene_names, CONFIG)
    
    # Run 2: Pre-train autoencoder on control data
    pathway_feats = path.precompute_pathway_features(control_adata, CONFIG)
    ```

### Step 2: Training

The `main.py` script ties all components together.

1.  **Configure Model:** Set all hyperparameters in the `CONFIG` dictionary in `main.py`.
2.  **Run Training:** Execute the main script. This will:
    * Instantiate the `TranscriptomePredictor` model.
    * Create the `train_loader` and `valid_loader`.
    * Call the `train_model` function from `training.py`.
    * Start the training loop, plotting live loss and $R^2$ metrics.
    * Save the final weights to `model_weights.pth`.

    ```bash
    python main.py
    ```
    *Note: For long training runs on an HPC, it is recommended to submit this command via a batch script (e.g., `sbatch run_training.sh`).*

### Step 3: Evaluation

The training script automatically evaluates the model at the end of each epoch. The primary metric for performance is the **Validation $R^2$ Score**, which is plotted live. A score > 0.0 indicates the model is outperforming the baseline (mean) guess.

## 5. File Structure

- `main.ipynb`: notebook used to run model
- `src` (main code repository)
    - `data_cleaning`: contains functions for cleaning data and extracting control data
    - `pathway_encoding`: functions to create pathway embedder
    - `position_encoding`: functions to create position embedder
    - `modelnew.py`: contains backbone and data structure
    - `trainingnew.py`: contains training, plotting, and loss function
- `arc` (archived old algorithms)

