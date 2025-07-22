import os
import torch
from PMTO import ContextualMultiObjectiveFunction, ParetoSetModel, VAE, ConditionalDDIM
from scipy.stats import qmc
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def generate_and_save_contexts(n_contexts, context_dim, file_path):
    # Use LHS to generate contexts
    sampler = qmc.LatinHypercube(d=context_dim)
    contexts = sampler.random(n=n_contexts)

    # Convert to torch tensor
    contexts_tensor = torch.tensor(contexts, dtype=torch.float32)

    # Save contexts to file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(contexts_tensor, file_path)

    return contexts_tensor

def generate_uniform_weights(num_uniform_weights: int, output_dim: int):
    """
    Generate uniformly distributed weight vectors using Dirichlet distribution.
    Each weight vector sums to 1 and provides good coverage of the preference space.

    Args:
        num_uniform_weights: Number of weight vectors to generate
        output_dim: Dimension of each weight vector (number of objectives)

    Returns:
        torch.Tensor: Shape (num_uniform_weights, output_dim) with each row summing to 1
    """
    import scipy.stats as stats

    if output_dim == 1:
        return torch.ones(num_uniform_weights, 1)

    # Use symmetric Dirichlet distribution with alpha=1 for uniform distribution on simplex
    # Dirichlet(1, 1, ..., 1) gives uniform distribution over the probability simplex
    alpha = [1.0] * output_dim

    # Generate samples using scipy's Dirichlet
    dirichlet_samples = stats.dirichlet.rvs(alpha, size=num_uniform_weights)

    # Convert to torch tensor
    weights = torch.tensor(dirichlet_samples, dtype=torch.float32)

    return weights

def reconstruct_inverse_model(complete_data, method_name):
    """
    Reconstruct the trained inverse model from saved data
    """
    if method_name == 'PSL-MOBO':
        return reconstruct_psl_models(complete_data)
    elif method_name == 'VAE':
        return reconstruct_vae_model(complete_data)
    elif method_name == 'DDIM':
        return reconstruct_ddim_model(complete_data)
    else:
        print(f"Unknown method: {method_name}")
        return None

def reconstruct_psl_models(complete_data):
    """
    Reconstruct PSL-MOBO models (one per training context)
    """
    models = {}

    if 'trained_models' not in complete_data:
        print("Warning: No trained PSL models found")
        return None

    for context_key, model_state in complete_data['trained_models'].items():
        config = complete_data['model_configs'][context_key]

        # Recreate ParetoSetModel
        model = ParetoSetModel(config['input_dim'], config['output_dim'])
        model.load_state_dict(model_state)
        model.eval()

        models[context_key] = model

    print(f"Reconstructed {len(models)} PSL models")
    return models

def reconstruct_vae_model(complete_data):
    """
    Reconstruct VAE model
    """
    if complete_data['trained_vae_model'] is None:
        print("Warning: No trained VAE model found")
        return None

    config = complete_data['vae_config']

    # ✅ CORRECTED: Match the original ParetoVAETrainer architecture
    encoder_sizes = [config['input_dim'], max(config['input_dim'], 2 * config['latent_dim'], config['input_dim'])]
    decoder_sizes = [max(config['input_dim'], 2 * config['latent_dim'], config['input_dim']), config['input_dim']]

    model = VAE(
        encoder_layer_sizes=encoder_sizes,
        latent_size=config['latent_dim'],
        decoder_layer_sizes=decoder_sizes,
        conditional=config['conditional'],
        context_size=config['context_dim'],
        true_conditional=config['true_conditional']
    )

    model.load_state_dict(complete_data['trained_vae_model'])
    model.eval()

    print("Reconstructed VAE model")
    return model

def reconstruct_ddim_model(complete_data):
    """
    Reconstruct DDIM model
    """
    if complete_data['trained_ddim_model'] is None:
        print("Warning: No trained DDIM model found")
        return None

    config = complete_data['ddim_config']

    # Recreate DDIM architecture
    model = ConditionalDDIM(
        input_dim=config['input_dim'],
        condition_dim=config['condition_dim'],
        timesteps=config['timesteps'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )

    model.load_state_dict(complete_data['trained_ddim_model'])
    model.eval()

    print("Reconstructed DDIM model")
    return model

def generate_solution(model, method_name, combined_input):
    """
    Generate solution using the trained inverse model
    """
    with torch.no_grad():
        if method_name == 'VAE':
            # For VAE: sample from latent space and decode
            if model.true_conditional:
                # Use conditional prior
                context_batch = combined_input.unsqueeze(0)
                z = model.sample_from_conditional_prior(context_batch, num_samples=1)
            else:
                # Use standard prior
                z = torch.randn(1, model.latent_size)

            context_batch = combined_input.unsqueeze(0)
            x_pred = model.inference(z, context_batch)
            return x_pred.squeeze(0)

        elif method_name == 'DDIM':
            # For DDIM: sample using DDIM process
            context_batch = combined_input.unsqueeze(0)
            shape = (1, model.input_dim)

            x_pred = model.ddim_sample(
                shape=shape,
                c=context_batch,
                num_steps=20,
                eta=0.5
            )
            return x_pred.squeeze(0)

    return None


def generate_solution_batch(model, method_name, combined_inputs):
    """
    Generate solutions in batch for VAE/DDIM
    """
    with torch.no_grad():
        if method_name == 'VAE':
            batch_size = combined_inputs.shape[0]

            if model.true_conditional:
                # Use conditional prior for batch
                z = model.sample_from_conditional_prior(combined_inputs, num_samples=1)
            else:
                # Use standard prior for batch
                z = torch.randn(batch_size, model.latent_size)

            x_pred_batch = model.inference(z, combined_inputs)
            return x_pred_batch

        elif method_name == 'DDIM':
            batch_size = combined_inputs.shape[0]
            shape = (batch_size, model.input_dim)

            x_pred_batch = model.ddim_sample(
                shape=shape,
                c=combined_inputs,
                num_steps=20,
                eta=0.0
            )

            return x_pred_batch

    return None

def evaluate_inverse_models(base_path, problem_config, methods_config,
                            reference_point, n_test_contexts=100):
    """
    Main evaluation function for inverse models

    Parameters:
    -----------
    base_path : str
        Base path to the results directory
    problem_config : dict
        Problem configuration with keys: 'name', 'dim', 'obj'
    methods_config : list of dict
        Method configurations (without dim/obj which come from problem_config)
    reference_point : array-like
        Reference point for hypervolume calculations
    n_test_contexts : int
        Number of test contexts to evaluate
    """
    problem_name = problem_config['name']
    n_variables = problem_config['dim']
    n_objectives = problem_config['obj']

    print(f"Evaluating inverse models for {problem_name}")

    # Initialize objective function
    obj_func = ContextualMultiObjectiveFunction(
        func_name=problem_name,
        n_objectives=n_objectives,
        n_variables=n_variables
    )

    # Load test contexts
    test_contexts_file = f'data/context_{n_test_contexts}_{obj_func.context_dim}.pth'
    if os.path.exists(test_contexts_file):
        test_contexts = torch.load(test_contexts_file)
        print(f"Loaded {len(test_contexts)} test contexts")
    else:
        test_contexts = generate_and_save_contexts(n_test_contexts, obj_func.context_dim, test_contexts_file)

    # Generate preference vectors - PLACEHOLDER
    if n_objectives == 2:
        preferences = generate_uniform_weights(100, n_objectives)
    if n_objectives == 3:
        preferences = generate_uniform_weights(1000, n_objectives)

    # Results storage
    inverse_model_results = {}

    for config in methods_config:
        method_name = config['name']
        print(f"\n=== Evaluating {method_name} ===")

        # Add problem dimensions to config for loading
        config_with_problem = {**config}
        config_with_problem['timestamp_params'].update({
            'problem': problem_name,
            'dim': n_variables,
            'obj': n_objectives
        })

        # Load complete data
        complete_data = load_complete_data(base_path, config_with_problem)
        if complete_data is None:
            continue

        # Reconstruct trained model
        trained_model = reconstruct_inverse_model(complete_data, method_name)
        if trained_model is None:
            continue

        # Evaluate on test contexts
        method_hvs = []

        for i, test_context in enumerate(test_contexts):
            if i % 20 == 0:
                print(f"  Processing test context {i + 1}/{len(test_contexts)}")

            if method_name == 'PSL-MOBO':
                # PSL-MOBO: batch process all preferences at once
                first_model_key = next(iter(trained_model.keys()))
                x_pred_batch = trained_model[first_model_key](preferences)  # Shape: [n_preferences, n_dim]

            else:
                # VAE/DDIM: batch process combined inputs
                n_preferences = len(preferences)
                test_context_clean = test_context[1:]  # Remove dummy dimension

                # Create batch of combined inputs: [context, preference] for each preference
                context_batch = test_context_clean.unsqueeze(0).repeat(n_preferences, 1)  # [n_pref, context_dim-1]
                combined_inputs = torch.cat([context_batch, preferences], dim=1)  # [n_pref, context_dim-1+n_obj]

                # Generate solutions in batch
                x_pred_batch = generate_solution_batch(trained_model, method_name, combined_inputs)
                # print("minimum {} and maximum {}".format(torch.min(x_pred_batch), torch.max(x_pred_batch)))

            # Batch evaluate objectives
            # Create batch of full inputs: [decision_vars, context] for each solution
            n_solutions = x_pred_batch.shape[0]
            context_batch = test_context.unsqueeze(0).repeat(n_solutions, 1)  # [n_solutions, context_dim]
            x_with_context_batch = torch.cat([x_pred_batch, context_batch],
                                             dim=1)  # [n_solutions, n_dim+context_dim]

            # Batch objective evaluation
            y_pred_batch = obj_func.evaluate(x_with_context_batch)  # [n_solutions, n_obj]
            # print(y_pred_batch)

            # Convert to list of individual solutions
            predicted_solutions = y_pred_batch
            # print(y_pred_batch)

            if len(predicted_solutions) > 0:
                # Compute hypervolume for this test context - PLACEHOLDER
                predicted_front = predicted_solutions
                hv = compute_hypervolume_placeholder(predicted_front.detach().numpy(), reference_point)
                # print(hv)
                method_hvs.append(hv)
            else:
                print(f"    Warning: No valid solutions for test context {i}")

        inverse_model_results[method_name] = method_hvs
        print(f"  Completed evaluation: {len(method_hvs)} valid hypervolumes")

    return inverse_model_results

def load_complete_data(base_path, config):
    """
    Load complete data for a method (removed n_runs parameter since not used)
    """
    method_name = config['name']
    method_dir = config['method_dir']

    # Format timestamp
    formatted_params = {}
    for key, value in config['timestamp_params'].items():
        if isinstance(value, float):
            formatted_params[key] = f"{value:.2f}"
        else:
            formatted_params[key] = value

    timestamp = config['timestamp_template'].format(**formatted_params)

    # Try to load data from run 0 (can be extended to try multiple runs)
    run = 1
    if method_name == 'PSL-MOBO':
        file_pattern = f"A_PSLMOBO3_complete_{timestamp}_run_{run}.pth"
    elif method_name == 'VAE':
        file_pattern = f"A_betaCVAE-CMOBO-nosigmoid_aug_2_0.1_complete_{timestamp}_run_{run}.pth"
    elif method_name == 'DDIM':
        file_pattern = f"A_DDIM-CMOBO_20steps_100_16_0.5_3_0.1_complete_{timestamp}_run_{run}.pth"
    else:
        print(f"Unknown method: {method_name}")
        return None

    file_path = Path(base_path) / method_dir / file_pattern
    if file_path.exists():
        complete_data = torch.load(file_path)
        print(f"Loaded complete data for {method_name} from run {run}")
        return complete_data
    else:
        print(f"Warning: No complete data found for {method_name} at {file_path}")
        return None

def compute_hypervolume_placeholder(pareto_front, reference_point):
    """
    PLACEHOLDER: Compute hypervolume
    Replace with your existing hypervolume computation function
    """
    nds = NonDominatedSorting()
    idx_nds = nds.do(pareto_front)
    Y_nds = pareto_front[idx_nds[0]]

    hv_computer = Hypervolume(ref_point=reference_point)
    hv_value = hv_computer.do(Y_nds)

    return hv_value

def plot_inverse_model_comparison(results, problem_name, reference_point, save_path):
    """
    Create boxplot comparison of inverse model performance

    Parameters:
    -----------
    results : dict
        Dictionary with method names as keys and lists of hypervolume values as values
    problem_name : str
        Name of the problem (e.g., 'dtlz1')
    reference_point : array-like
        Reference point used for hypervolume calculations
    save_path : str
        Path where to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    # ✅ UPDATED: Specific colors for each method
    method_colors = {
        'PSL-MOBO': 'pink',
        'VAE': 'green',
        'DDIM': 'red'
    }

    colors = []
    for method_name, hvs in results.items():
        if len(hvs) > 0:  # Only include methods with valid results
            data_to_plot.append(hvs)
            labels.append(method_name)
            colors.append(method_colors.get(method_name, 'lightblue'))  # Default to lightblue if method not found

    if len(data_to_plot) == 0:
        print("Warning: No valid results to plot")
        return

    # Create boxplot
    box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)

    # Color the boxes with specified colors
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title(f'Inverse Model Performance Comparison - {problem_name.upper()}', fontsize=14)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.grid(True, alpha=0.3)

    # ✅ REMOVED: No statistics text on the graph
    # No digits displayed on the plot

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved inverse model comparison plot to {save_path}")

# Example usage with new structure
def example_inverse_model_evaluation():
    # Problem configuration (extracted from methods)
    problem_config = {
        'name': 'bicopter',
        'dim': 12,
        'obj': 2,
        'ref_point': np.array([1.0, 1.0])
        # 'ref_point': np.array([1.0, 1.0, 1.0])
    }

    base_path = f"result/{problem_config['name']}"

    # Adjust reference point for your problem
    reference_point = problem_config['ref_point']

    # Methods configuration (no longer contains dim/obj)
    methods_config = [
        {
            'name': 'PSL-MOBO',
            'method_dir': '',  # Files are in base directory
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_pslmobo_test_hv",
            'timestamp_params': {
                'epsilon': 1.00
            }
        },
        {
            'name': 'VAE',
            'method_dir': '',  # Files are in base directory
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_CustomGP_hv_constrain",
            'timestamp_params': {
                'epsilon': 1.00
            }
        },
        {
            'name': 'DDIM',
            'method_dir': '',  # Files are in base directory
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_ExactGP_hv_constrain",
            'timestamp_params': {
                'epsilon': 1.00
            }
        }
    ]

    # Run evaluation
    results = evaluate_inverse_models(
        base_path, problem_config, methods_config,
        reference_point, n_test_contexts=100
    )

    # Create comparison plot
    save_path = f"{base_path}/inverse_model_comparison_{problem_config['name']}.png"
    plot_inverse_model_comparison(results, problem_config['name'], reference_point, save_path)

    # Print statistics
    print(f"\n=== INVERSE MODEL PERFORMANCE ON {problem_config['name'].upper()} ===")
    for method_name, hvs in results.items():
        if len(hvs) > 0:
            print(f"{method_name:15}: {np.mean(hvs):.4f} ± {np.std(hvs):.4f} (n={len(hvs)})")
        else:
            print(f"{method_name:15}: No valid results")

    return results


if __name__ == "__main__":
    results = example_inverse_model_evaluation()