import numpy as np
import torch
from scipy.stats import qmc
import os


def generate_and_save_initial_points(n_contexts, input_dim=3, n_points=20, save_dir='data'):
    """
    Generate and save different initial points for each context in separate .pth files.

    Args:
        n_contexts (int): Number of contexts
        input_dim (int): Dimension of input space
        n_points (int): Number of points per context
        save_dir (str): Directory to save the point files
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set base random seed
    base_seed = 42

    # Generate different points for each context
    for context_idx in range(n_contexts):
        # Use different seed for each context
        seed = base_seed + context_idx

        # Generate points using LHS
        sampler = qmc.LatinHypercube(d=input_dim, seed=seed)
        points = sampler.random(n=n_points)

        # Convert to tensor [n_points, input_dim]
        points_tensor = torch.tensor(points, dtype=torch.float32)

        # Save points for this context
        save_path = os.path.join(save_dir, f'init_points_context_{context_idx}_{input_dim}_{n_points}.pth')
        torch.save(points_tensor, save_path)
        print(f"Saved context {context_idx} points of shape {points_tensor.shape} to {save_path}")


def load_initial_points(context_idx, save_dir='data', input_dim=5, input_points=10):
    """
    Load initial points for a specific context.

    Args:
        context_idx (int): Index of the context
        save_dir (str): Directory where point files are saved

    Returns:
        torch.Tensor: Initial points for the specified context of shape [n_points, input_dim]
    """
    load_path = os.path.join(save_dir, f'init_points_context_{context_idx}_{input_dim}_{input_points}.pth')
    return torch.load(load_path)


def generate_and_save_contexts(n_contexts, context_dim, save_dir='data'):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set base random seed
    base_seed = 42

    # Use LHS to generate contexts
    sampler = qmc.LatinHypercube(d=context_dim)
    contexts = sampler.random(n=n_contexts)

    # Convert to torch tensor
    contexts_tensor = torch.tensor(contexts, dtype=torch.float32)
    new_tensor = torch.rand([n_contexts, 1])
    contexts_tensor = torch.cat([new_tensor, contexts_tensor], dim=1)
    print(contexts_tensor)

    save_path = os.path.join(save_dir, f'context_{n_contexts}_{context_dim + 1}.pth')
    torch.save(contexts_tensor, save_path)

    return contexts_tensor


# Example usage
if __name__ == "__main__":
    # generate_and_save_contexts(n_contexts=8, context_dim=2)
    # Configuration
    N_CONTEXTS = 8  # Number of contexts
    INPUT_DIM = 3  # 3D input space
    N_POINTS = 5  # 20 points per context

    # Generate and save points
    generate_and_save_initial_points(
        n_contexts=N_CONTEXTS,
        input_dim=INPUT_DIM,
        n_points=N_POINTS
    )

    # Load and verify points for each context
    for i in range(N_CONTEXTS):
        points = load_initial_points(context_idx=i, input_dim=INPUT_DIM, input_points=N_POINTS)
        print(f"\nLoaded points for context {i}:")
        print("Shape:", points.shape)
        print("First 3 points:")
        print(points[:3])