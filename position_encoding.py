import torch
from tqdm import tqdm

def create_fourier_features(tensor: torch.Tensor, n_features: int) -> torch.Tensor:
    """
    Helper function to create Fourier features for a continuous variable in [0, 1].
    This is now a standalone function, callable from anywhere.

    Args:
        tensor (torch.Tensor): A tensor of shape (N, 1) with values between 0 and 1.
        n_features (int): The number of Fourier frequency pairs (F).

    Returns:
        torch.Tensor: A tensor of shape (N, 2*F).
    """
    freqs = torch.pi * (2**torch.arange(n_features, dtype=tensor.dtype))
    angles = tensor * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def precompute_positional_indices(data, gene_names, config):
    """
    Uses pyensembl to fetch the chromosome and locus for every gene in the dataset,
    returning the two tensors required by the model: chromosome indices and normalized locus.

    Args:
        data (list): A list of all of the different genes from the pyensmbl database
        gene_names (list): A list of all gene names from your dataset (e.g., adata.var_names).
        config (dict): The main configuration dictionary.

    Returns:
        torch.Tensor: chr_idx, a tensor of shape (n_genes,) with integer chromosome indices.
        torch.Tensor: locus_norm, a tensor of shape (n_genes, 1) with normalized start positions.
    """
    print("\n--- Preparing positional indices using pyensembl ---")
    
    # --- Chromosome Mapping and Lengths ---
    chr_names = [str(i) for i in range(1, 23)] + ['X', 'Y']
    chromosome_map = {name: i for i, name in enumerate(chr_names)}
    
    chromosome_lengths = {
        '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555, 
        '5': 181538259, '6': 170805979, '7': 159345973, '8': 145138636, 
        '9': 138394717, '10': 133797422, '11': 135086622, '12': 133275309, 
        '13': 114364328, '14': 107043718, '15': 101991189, '16': 90338345, 
        '17': 83257441, '18': 80373285, '19': 58617616, '20': 64444167, 
        '21': 46709983, '22': 50818468, 'X': 156040895, 'Y': 57227415
    }

    # --- Data Fetching Loop ---
    chr_indices = []
    locus_positions = []
    
    # Default to a "not found" encoding (e.g., an extra chromosome index)
    unknown_chr_index = len(chromosome_map) # 24
    if config["n_chromosomes"] <= unknown_chr_index:
        config["n_chromosomes"] = unknown_chr_index + 1
        print(f"Increased n_chromosomes to {config['n_chromosomes']} to handle unknown genes.")


    for gene_name in tqdm(gene_names, desc="Fetching gene positions"):
        try:
            # Fetch gene data from Ensembl
            gene = data.genes_by_name(gene_name)[0]
            contig = gene.contig
            
            if contig in chromosome_map and contig in chromosome_lengths:
                chr_index = chromosome_map[contig]
                # Normalize locus by chromosome length
                normalized_locus = gene.start / chromosome_lengths[contig]
                
                chr_indices.append(chr_index)
                locus_positions.append(normalized_locus)
            else:
                # Handle cases where contig is not a standard chromosome (e.g., mitochondrial DNA)
                chr_indices.append(unknown_chr_index)
                locus_positions.append(0.5) # A neutral position
    
        except (IndexError, ValueError):
            # Gene not found in Ensembl database
            chr_indices.append(unknown_chr_index)
            locus_positions.append(0.5)

    # --- Convert to Tensors ---
    chr_idx = torch.tensor(chr_indices, dtype=torch.long)
    locus_norm = torch.tensor(locus_positions, dtype=torch.float32).unsqueeze(1)

    locus_fourier = create_fourier_features(
        locus_norm, 
        config["locus_fourier_features"]
    )

    print(f"\nSUCCESS: Generated positional tensors with shapes:")
    print(f"Chromosome Indices (chr_idx): {chr_idx.shape}")
    print(f"Normalized Locus (locus_norm): {locus_norm.shape}")
    print(f"Locus Fourier Features (locus_fourier): {locus_fourier.shape}")
    
    # Return all three tensors
    return chr_idx, locus_norm, locus_fourier

