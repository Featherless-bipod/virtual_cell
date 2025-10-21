# Genomic Perturbation Response Predictor

This project implements a state-of-the-art deep learning model to predict the full transcriptome-wide effects of a single gene perturbation. Given a specific gene knockout from a Perturb-seq experiment, the model predicts the resulting expression levels for all ~18,000 genes in the cell.

The architecture is based on modern sequence modeling principles, treating the genome as a sequence and using a powerful backbone like Mamba to learn complex, long-range biological interactions.

## Model Architecture

The model is designed as a causal predictor, learning the function `f(Cause, Baseline_State) -> Effect`. It fuses information about the specific experimental **condition** with a rich, universal representation of all **genes**.



### 1. Input Sequence Construction

The model processes a single, long sequence of `1 + n_genes` tokens.

* **The "1" (Condition Token):** A single vector representing the experimental condition. It is a combination of:
    * A learnable embedding for the specific perturbed gene (e.g., "TP53").
    * (Optional but recommended) Technical covariates like total UMI counts and the number of detected genes.
* **The "`n_genes`" (Gene Matrix):** A large matrix representing the universal properties of every gene in the genome. Each gene's feature vector is a concatenation of:
    1.  **Learnable Identity (`e_id`):** The primary learnable embedding for each gene, capturing its unique functional identity.
    2.  **Pathway Features (`e_path`):** A pre-computed vector describing the gene's baseline network connections. This can be generated in two ways:
        * **Data-Driven (Autoencoder):** Training an autoencoder on a control dataset of healthy, unperturbed cells (e.g., the 10k PBMC dataset).
        * **Knowledge-Driven (Initialization):** Using a known pathway database like the **MSigDB Hallmark Collection (50 pathways)** to guide or initialize the autoencoder.
    3.  **Positional Features (`e_pos`):** A hybrid encoding of the gene's physical location:
        * A learnable embedding for the chromosome identity (1-22, X, Y).
        * A **Fourier Feature** representation of the normalized locus, passed through a small MLP to create a rich, continuous positional vector.

### 2. Model Backbone

The `(1 + n_genes)` sequence is processed by a powerful sequential model to capture context and dependencies.

* **Primary Choice (Mamba):** A **Bidirectional Mamba-2** architecture. This is the state-of-the-art choice due to its linear scalability ($O(N)$), which is essential for handling the long sequence of the genome. It excels at modeling the long-range interactions characteristic of genomic regulation.
* **Fallback (LSTM):** A standard `nn.LSTM(bidirectional=True)` is used as a robust placeholder for development and for environments where Mamba's custom CUDA kernels cannot be compiled.

### 3. Prediction Head

The model supports two output modes, configurable via the `CONFIG` dictionary:

* **`linear`:** A simple linear layer that outputs a single value per gene, trained with Mean Squared Error (MSE) loss. This is a good baseline.
* **`probabilistic` (Recommended):** A linear layer that outputs the **two parameters (mean, dispersion)** of a **Negative Binomial (NB)** distribution. This is the statistically correct approach for modeling overdispersed UMI count data and is trained with the Negative Binomial Negative Log-Likelihood (`nb_nll_loss`).

## Data Pre-processing

A rigorous, multi-stage QC and pre-processing pipeline is essential.

1.  **Transcriptome QC:** An adaptive filtering process is used on the raw `AnnData` object to remove low-quality cells. Thresholds for `min_genes_per_cell` and `max_genes_per_cell` are determined by visualizing the data distributions, rather than using fixed numbers.
2.  **Perturb-seq QC:** A second filtering step is applied to ensure the validity of the perturbation label for each cell. This removes:
    * **Unlabeled cells:** Cells where no guide RNA (gRNA) was detected.
    * **Multiplets:** Cells where more than one unique gRNA was detected.
3.  **Data Representation:** The pre-processing pipeline correctly prepares two versions of the expression data:
    * **`adata.X`:** Normalized (CP10k) and log-transformed (`log1p`) data, used for training the `PathwayAutoencoder`.
    * **`adata.layers['counts']`:** The original, raw integer counts, used as the target for the main model when using the `probabilistic` head.

## Setup and Usage

This project is designed to be run in a GPU-enabled environment like Google Colab.

### 1. Installation

The Mamba library requires compilation. The following installation order is recommended in a fresh environment.

```bash
# It is strongly recommended to use a version of numpy < 2.0 for broad compatibility
!pip install "numpy<2"

# Install Mamba's compiled dependencies and the main package
!pip install causal-conv1d --no-build-isolation
!pip install "mamba-ssm @ git+[https://github.com/state-spaces/mamba.git#egg=mamba-ssm](https://github.com/state-spaces/mamba.git#egg=mamba-ssm)"

# Install the remaining data science packages
!pip install scanpy pandas pyensembl