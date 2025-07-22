import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import seaborn as sns

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import matplotlib.colors as mcolors
from pymoo.indicators.hv import HV
from copy import deepcopy
from matplotlib.gridspec import GridSpec


def load_run_data(base_path, method, timestamp, n_runs):
    """Load data from all runs for a given method."""
    all_runs_data = []
    for run in range(n_runs):
        filename = f'{method}_optimization_history_{timestamp}_run_{run}.pth'
        filepath = Path(base_path) / filename
        if filepath.exists():
            run_data = torch.load(filepath)  # Load using torch.load since data contains tensors
            all_runs_data.append(run_data)
        else:
            print(f"Warning: Could not find file {filepath}")
    return all_runs_data


def compute_hypervolume(pareto_front, reference_point):
    """
    Compute hypervolume of a Pareto front with respect to a reference point.

    Parameters:
    -----------
    pareto_front : numpy.ndarray
        Array of objective vectors in the Pareto front
    reference_point : numpy.ndarray
        Reference point for hypervolume computation

    Returns:
    --------
    float
        Hypervolume value
    """
    # Initialize the hypervolume indicator
    hv = HV(ref_point=reference_point)

    return hv.do(pareto_front.numpy())


def extract_pareto_fronts_and_compute_hv(all_runs_data, reference_point, use_final_pareto=False):
    """
    Extract Pareto fronts from run data and compute hypervolumes.

    Parameters:
    -----------
    all_runs_data : list
        List of run data dictionaries
    reference_point : numpy.ndarray
        Reference point for hypervolume computation
    use_final_pareto : bool, optional
        If True, compute HV based only on the final Pareto front

    Returns:
    --------
    dict
        Dictionary of hypervolume histories for each context
    """
    # First, get all context keys from the first run
    context_keys = list(all_runs_data[0].keys())

    # Initialize dictionary to store HV data for each context
    context_hv_data = {context: [] for context in context_keys}

    # Process each run
    for run_data in all_runs_data:
        for context in context_keys:
            if context in run_data:
                # Directly use the pareto_front_history
                if 'pareto_front_history' in run_data[context]:
                    pareto_fronts = run_data[context]['pareto_front_history']
                    if torch.is_tensor(pareto_fronts):
                        pareto_fronts = pareto_fronts.cpu().numpy()

                    # Compute hypervolume history
                    n_iterations = len(pareto_fronts)
                    hv_history = np.zeros(n_iterations)

                    # Compute HV at each iteration
                    if use_final_pareto and n_iterations > 0:
                        # Only compute final Pareto front and use that for all iterations
                        final_pareto = pareto_fronts[-1]
                        if len(final_pareto) > 0:
                            final_hv = compute_hypervolume(final_pareto, reference_point)
                            hv_history[-1] = final_hv
                    else:
                        # Compute HV for each Pareto front in the history
                        for i in range(n_iterations):
                            current_pareto = pareto_fronts[i]
                            if len(current_pareto) > 0:
                                hv_history[i] = compute_hypervolume(current_pareto, reference_point)

                    context_hv_data[context].append(hv_history)
                else:
                    print(f"Warning: No pareto_front_history found for context {context}")

    # Convert lists to numpy arrays and ensure all sequences have the same length
    for context in context_keys:
        if context_hv_data[context]:  # Check if list is not empty
            # Pad shorter sequences with their last value
            max_len = max(len(hv) for hv in context_hv_data[context])
            padded_hvs = []
            for hv_seq in context_hv_data[context]:
                if len(hv_seq) < max_len:
                    padding = [hv_seq[-1]] * (max_len - len(hv_seq))
                    padded_hvs.append(np.concatenate([hv_seq, padding]))
                else:
                    padded_hvs.append(hv_seq)
            context_hv_data[context] = np.array(padded_hvs)
        else:
            # Remove context if no data
            del context_hv_data[context]

    return context_hv_data


def analyze_and_plot_with_custom_hv(base_path, problem_name, n_runs, methods_config, reference_point, cut_size=50,
                                    use_final_pareto=False, dim=0):
    """
    Analyze and plot results using custom hypervolume calculations.

    Parameters:
    -----------
    base_path : str
        Base path to the results directory
    problem_name : str
        Name of the problem being solved
    n_runs : int
        Number of runs to analyze
    methods_config : list of dict
        List of method configurations, each with keys:
        - 'name': display name for the method
        - 'method_dir': directory containing the method's results
        - 'timestamp_template': template for timestamp formatting
        - 'timestamp_params': parameters to insert into template
        - 'color': (optional) color for plotting this method
    reference_point : array-like
        Reference point for hypervolume calculations
    cut_size : int, optional
        Number of iterations to include in plots (default: 50)
    use_final_pareto : bool, optional
        If True, compute HV based only on the final Pareto front

    Returns:
    --------
    dict
        Statistics for each method
    """
    # Set up colors if not provided
    color_cycle = list(mcolors.TABLEAU_COLORS)
    for i, config in enumerate(methods_config):
        if 'color' not in config:
            config['color'] = color_cycle[i % len(color_cycle)]

    # Load data for all methods
    methods_data = {}
    methods_hv_data = {}

    for config in methods_config:
        # Format the timestamp with proper floating-point formatting
        formatted_params = {}
        for key, value in config['timestamp_params'].items():
            if isinstance(value, float):
                # Format floats with fixed precision (2 decimal places)
                formatted_params[key] = f"{value:.2f}"
            else:
                formatted_params[key] = value

        timestamp = config['timestamp_template'].format(**formatted_params)

        # Load the data
        data = load_run_data(base_path, config['method_dir'], timestamp, n_runs)

        if not data:
            print(f"Error: No data found for method {config['name']}")
            continue

        methods_data[config['name']] = data
        # Compute hypervolumes with the provided reference point
        methods_hv_data[config['name']] = extract_pareto_fronts_and_compute_hv(
            data, reference_point, use_final_pareto
        )

    if not methods_hv_data:
        print("Error: No data found for any method")
        return None

    # Get the list of contexts (assuming all methods have the same contexts)
    first_method = list(methods_hv_data.keys())[0]
    contexts = list(methods_hv_data[first_method].keys())
    n_contexts = len(contexts)

    # Create subplots for each context
    n_cols = min(5, n_contexts)
    n_rows = (n_contexts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle(f'Comparison of {problem_name} Optimization Methods (Custom HV)', fontsize=16)

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each context
    for idx, context in enumerate(contexts):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Plot each method
        for method_name, method_data in methods_hv_data.items():
            if context not in method_data:
                continue

            config = next(m for m in methods_config if m['name'] == method_name)

            # Calculate statistics
            mean_values = np.mean(method_data[context], axis=0)
            std_values = np.std(method_data[context], axis=0) / 5

            # Ensure we don't exceed the available data length
            plot_size = min(cut_size, len(mean_values))
            x = np.arange(plot_size)

            # Plot mean and standard deviation
            ax.plot(x, mean_values[:plot_size], label=method_name, color=config['color'])
            ax.fill_between(
                x,
                mean_values[:plot_size] - std_values[:plot_size],
                mean_values[:plot_size] + std_values[:plot_size],
                alpha=0.2,
                color=config['color']
            )

        ax.set_title(f'Context {idx}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypervolume')
        ax.legend()
        ax.grid(True)

    # Remove empty subplots if any
    for idx in range(n_contexts, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create a filename incorporating all method names and reference point
    method_names_str = "_".join([m['name'].replace(" ", "-") for m in methods_config])
    ref_point_str = "_".join([f"{x:.1f}" for x in reference_point])
    comparison_filename = f'custom_hv_comparison_plot_finerr_{dim}_{problem_name}_{method_names_str}_ref_{ref_point_str}.png'
    save_path = Path(base_path) / comparison_filename
    plt.savefig(save_path)
    plt.close()

    # Create a summary plot of average performance across all contexts
    plt.figure(figsize=(10, 6))

    # Calculate and plot overall statistics for each method
    stats = {}

    for method_name, method_data in methods_hv_data.items():
        config = next(m for m in methods_config if m['name'] == method_name)

        # Calculate average across all contexts
        overall_mean = np.mean([np.mean(data, axis=0) for data in method_data.values()], axis=0)
        overall_std = np.mean([np.std(data, axis=0) for data in method_data.values()], axis=0)

        # Ensure we don't exceed the available data length
        plot_size = min(cut_size, len(overall_mean))
        x = np.arange(plot_size)

        # Plot overall statistics
        plt.plot(x, overall_mean[:plot_size], label=method_name, color=config['color'])
        plt.fill_between(
            x,
            overall_mean[:plot_size] - overall_std[:plot_size] / 2,
            overall_mean[:plot_size] + overall_std[:plot_size] / 2,
            alpha=0.2,
            color=config['color']
        )

        # Store final statistics
        stats[method_name] = {
            'final_mean_hv': float(overall_mean[plot_size - 1]),
            'final_std_hv': float(overall_std[plot_size - 1])
        }

    plt.title(f'Average Custom Hypervolume Progress Across All Tasks - {problem_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Hypervolume')
    plt.legend()
    plt.grid(True)

    # Save the summary plot
    summary_filename = f'custom_hv_summary_comparison_finerr_{dim}_{problem_name}_{method_names_str}_ref_{ref_point_str}.png'
    summary_save_path = Path(base_path) / summary_filename
    plt.savefig(summary_save_path)
    plt.close()

    return stats


# Example usage
def example_custom_hv_usage():
    # problem_name = "dtlz2"
    problem_name = "magnetic_sifter"
    base_path = f"result/{problem_name}"
    n_runs = 5
    example_dim = 3
    example_obj = 3

    # Define reference point for hypervolume calculation
    # Adjust this reference point according to your specific problem
    # reference_point = np.array([10.0, 10.0]) * 0.1  # For 2-objective problems
    # reference_point = np.array([100.0, 100.0])*1.0  # For 2-objective problems
    reference_point = torch.ones(3).numpy() * 1.0

    # Define the methods to compare (same as in the original code)
    methods_config = [
        {
            'name': 'MOBO',
            'method_dir': 'MOBO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_hv",
            'timestamp_params': {
                'problem': problem_name,
                'dim': example_dim,
                'obj': example_obj,
                'epsilon': 1.00
            },
        },
        {
            'name': 'P-MOBO',
            'method_dir': 'CMOBO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_CustomGP_hv_constrain",
            'timestamp_params': {
                'problem': problem_name,
                'dim': example_dim,
                'obj': example_obj,
                'epsilon': 1.00
            },
            # 'color': 'red'
        },
        # {
        #     'name': 'P-MOBO-VAE-U',
        #     'method_dir': 'betaVAE-CMOBO-uniform-nosigmoid_aug_2_0.1',
        #     'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_CustomGP_hv_constrain",
        #     'timestamp_params': {
        #         'problem': problem_name,
        #         'dim': 5,
        #         'obj': 2,
        #         'epsilon': 1.00
        #     },
        #     'color': 'brown'
        # },
        {
            'name': 'P-MOBO-VAE',
            'method_dir': 'betaVAE-CMOBO-nosigmoid_aug_2_0.1',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_CustomGP_hv_constrain",
            'timestamp_params': {
                'problem': problem_name,
                'dim': example_dim,
                'obj': example_obj,
                'epsilon': 1.00
            },
            # 'color': 'green'
        },
        # {
        #     'name': 'P-MOBO-VAE-AGG',
        #     'method_dir': 'betaVAE-CMOBO-agg-nosigmoid_aug_2_0.1',
        #     'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_CustomGP_hv_constrain",
        #     'timestamp_params': {
        #         'problem': problem_name,
        #         'dim': example_dim,
        #         'obj': example_obj,
        #         'epsilon': 1.00
        #     },
        #     'color': 'red'
        # },
        {
            'name': 'P-MOBO-DDIM3',
            'method_dir': 'DDIM-CMOBO_20steps_100_16_0.5_3_0.1',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_CustomGP_hv_constrain",
            'timestamp_params': {
                'problem': problem_name,
                'dim': example_dim,
                'obj': example_obj,
                'epsilon': 1.00
            },
            # 'color': 'brown'
        },
        {
            'name': 'PAREGO',
            'method_dir': 'ParEGO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_parego_test_hv",
            'timestamp_params': {
                'problem': problem_name,
                'dim': example_dim,
                'obj': example_obj,
                'epsilon': 1.00
            },
            # 'color': 'purple'
        },
        {
            'name': 'EHVI',
            'method_dir': 'EHVI3',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_ehvi_test_hv",
            'timestamp_params': {
                'problem': problem_name,
                'dim': example_dim,
                'obj': example_obj,
                'epsilon': 1.00
            },
            # 'color': 'yellow'
        },
        {
            'name': 'PSLMOBO',
            'method_dir': 'PSLMOBO3',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_pslmobo_test_hv",
            'timestamp_params': {
                'problem': problem_name,
                'dim': example_dim,
                'obj': example_obj,
                'epsilon': 1.00
            },
            # 'color': 'orange'
        },
    ]

    # Run the analysis with custom hypervolume computation
    stats = analyze_and_plot_with_custom_hv(
        base_path,
        problem_name,
        n_runs,
        methods_config,
        reference_point,
        cut_size=50,
        use_final_pareto=False,  # Set to True to use only the final Pareto front
        dim=example_dim
    )

    # Print the final statistics
    if stats:
        print("\nFinal Custom Hypervolume Statistics:")
        for method, method_stats in stats.items():
            print(f"{method}: {method_stats['final_mean_hv']:.4f} ± {method_stats['final_std_hv']:.4f}")


def analyze_and_plot_results(base_path, problem_name, n_runs, methods_config, cut_size=50):
    """
    Analyze and plot results for multiple MOBO method variants.

    Parameters:
    -----------
    base_path : str
        Base path to the results directory
    problem_name : str
        Name of the problem being solved
    n_runs : int
        Number of runs to analyze
    methods_config : list of dict
        List of method configurations, each with keys:
        - 'name': display name for the method
        - 'method_dir': directory containing the method's results
        - 'timestamp_template': template for timestamp formatting
        - 'timestamp_params': parameters to insert into template
        - 'color': (optional) color for plotting this method
    cut_size : int, optional
        Number of iterations to include in plots (default: 50)

    Returns:
    --------
    dict
        Statistics for each method
    """
    # Set up colors if not provided
    color_cycle = list(mcolors.TABLEAU_COLORS)
    for i, config in enumerate(methods_config):
        if 'color' not in config:
            config['color'] = color_cycle[i % len(color_cycle)]

    # Load data for all methods
    methods_data = {}
    methods_hv_data = {}

    for config in methods_config:
        # Format the timestamp with proper floating-point formatting
        formatted_params = {}
        for key, value in config['timestamp_params'].items():
            if isinstance(value, float):
                # Format floats with fixed precision (2 decimal places)
                formatted_params[key] = f"{value:.2f}"
            else:
                formatted_params[key] = value

        timestamp = config['timestamp_template'].format(**formatted_params)

        # Load the data
        data = load_run_data(base_path, config['method_dir'], timestamp, n_runs)

        if not data:
            print(f"Error: No data found for method {config['name']}")
            continue

        methods_data[config['name']] = data
        methods_hv_data[config['name']] = process_hypervolumes(data)

    if not methods_hv_data:
        print("Error: No data found for any method")
        return None

    # Get the list of contexts (assuming all methods have the same contexts)
    first_method = list(methods_hv_data.keys())[0]
    contexts = list(methods_hv_data[first_method].keys())
    n_contexts = len(contexts)

    # Create subplots for each context
    n_cols = min(5, n_contexts)
    n_rows = (n_contexts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle(f'Comparison of {problem_name} Optimization Methods', fontsize=16)

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each context
    for idx, context in enumerate(contexts):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Plot each method
        for method_name, method_data in methods_hv_data.items():
            if context not in method_data:
                continue

            config = next(m for m in methods_config if m['name'] == method_name)

            # Calculate statistics
            mean_values = np.mean(method_data[context], axis=0)
            std_values = np.std(method_data[context], axis=0) / 5

            # Ensure we don't exceed the available data length
            plot_size = min(cut_size, len(mean_values))
            x = np.arange(plot_size)

            # Plot mean and standard deviation
            ax.plot(x, mean_values[:plot_size], label=method_name, color=config['color'])
            ax.fill_between(
                x,
                mean_values[:plot_size] - std_values[:plot_size],
                mean_values[:plot_size] + std_values[:plot_size],
                alpha=0.2,
                color=config['color']
            )

        ax.set_title(f'Context {idx}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypervolume')
        ax.legend()
        ax.grid(True)

    # Remove empty subplots if any
    for idx in range(n_contexts, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create a filename incorporating all method names
    method_names_str = "_".join([m['name'].replace(" ", "-") for m in methods_config])
    comparison_filename = f'comparison_plot_{problem_name}_{method_names_str}.png'
    save_path = Path(base_path) / comparison_filename
    plt.savefig(save_path)
    plt.close()

    # Create a summary plot of average performance across all contexts
    plt.figure(figsize=(10, 6))

    # Calculate and plot overall statistics for each method
    stats = {}

    for method_name, method_data in methods_hv_data.items():
        config = next(m for m in methods_config if m['name'] == method_name)

        # Calculate average across all contexts
        overall_mean = np.mean([np.mean(data, axis=0) for data in method_data.values()], axis=0)
        overall_std = np.mean([np.std(data, axis=0) for data in method_data.values()], axis=0)

        # Ensure we don't exceed the available data length
        plot_size = min(cut_size, len(overall_mean))
        x = np.arange(plot_size)

        # Plot overall statistics
        plt.plot(x, overall_mean[:plot_size], label=method_name, color=config['color'])
        plt.fill_between(
            x,
            overall_mean[:plot_size] - overall_std[:plot_size],
            overall_mean[:plot_size] + overall_std[:plot_size],
            alpha=0.2,
            color=config['color']
        )

        # Store final statistics
        stats[method_name] = {
            'final_mean_hv': float(overall_mean[plot_size - 1]),
            'final_std_hv': float(overall_std[plot_size - 1])
        }

    plt.title(f'Average Hypervolume Progress Across All Tasks - {problem_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Hypervolume')
    plt.legend()
    plt.grid(True)

    # Save the summary plot
    summary_filename = f'summary_comparison_{problem_name}_{method_names_str}.png'
    summary_save_path = Path(base_path) / summary_filename
    plt.savefig(summary_save_path)
    plt.close()

    return stats


# Example usage of the new function:
def example_usage():
    problem_name = "dtlz3"
    base_path = "result/{}".format(problem_name)
    n_runs = 5

    # Define the methods to compare
    methods_config = [
        {
            'name': 'MOBO',
            'method_dir': 'MOBO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_hv",
            'timestamp_params': {
                'problem': problem_name,
                'dim': 5,
                'obj': 2,
                'epsilon': 0.01
            },
            'color': 'blue'
        },
        {
            'name': 'P-MOBO',
            'method_dir': 'CMOBO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_CustomGP_hv_constrain",
            'timestamp_params': {
                'problem': problem_name,
                'dim': 5,
                'obj': 2,
                'epsilon': 1.00
            },
            'color': 'red'
        },
        {
            'name': 'P-MOBO-VAE',
            'method_dir': 'betaVAE-CMOBO-nosigmoid_aug_2_0.2',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_CustomGP_hv_constrain",
            'timestamp_params': {
                'problem': problem_name,
                'dim': 5,
                'obj': 2,
                'epsilon': 1.00
            },
            'color': 'green'
        },
        {
            'name': 'PAREGO',
            'method_dir': 'ParEGO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_parego_test_hv",
            'timestamp_params': {
                'problem': problem_name,
                'dim': 5,
                'obj': 2,
                'epsilon': 1.00
            },
            'color': 'purple'
        },
        {
            'name': 'EHVI',
            'method_dir': 'EHVI3',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_ehvi_test_hv",
            'timestamp_params': {
                'problem': problem_name,
                'dim': 5,
                'obj': 2,
                'epsilon': 1.00
            },
            'color': 'black'
        },
        # {
        #     'name': 'PSLMOBO',
        #     'method_dir': 'PSLMOBO2',
        #     'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_pslmobo_test_hv",
        #     'timestamp_params': {
        #         'problem': problem_name,
        #         'dim': 5,
        #         'obj': 2,
        #         'epsilon': 1.00
        #     },
        #     'color': 'orange'
        # },
        # Add more methods as needed
    ]

    # Run the analysis
    stats = analyze_and_plot_results(base_path, problem_name, n_runs, methods_config)

    # Print the final statistics
    if stats:
        print("\nFinal Hypervolume Statistics:")
        for method, method_stats in stats.items():
            print(f"{method}: {method_stats['final_mean_hv']:.4f} ± {method_stats['final_std_hv']:.4f}")


# Note: These functions from the original code are assumed to be defined elsewhere
# def load_run_data(base_path, method_dir, timestamp, n_runs):
#     # Implementation here
#     pass

# def process_hypervolumes(data):
#     # Implementation here
#     pass

# if __name__ == "__main__":
#     example_usage()

def load_run_data(base_path, method, timestamp, n_runs):
    """Load data from all runs for a given method."""
    all_runs_data = []
    for run in range(n_runs):
        filename = f'{method}_optimization_history_{timestamp}_run_{run}.pth'
        filepath = Path(base_path) / filename
        if filepath.exists():
            run_data = torch.load(filepath)  # Load using torch.load since data contains tensors
            all_runs_data.append(run_data)
        else:
            print(f"Warning: Could not find file {filepath}")
    return all_runs_data


def process_hypervolumes(all_runs_data):
    """Process hypervolume data from all runs and contexts."""
    # First, get all context keys from the first run
    context_keys = list(all_runs_data[0].keys())

    # Initialize dictionary to store HV data for each context
    context_hv_data = {context: [] for context in context_keys}

    # Collect HV histories for each context across all runs
    for run_data in all_runs_data:
        for context in context_keys:
            if context in run_data:
                # Convert tensor to numpy array if it's a tensor
                hv_history = run_data[context]['hv_history']
                if torch.is_tensor(hv_history):
                    hv_history = hv_history.cpu().numpy()
                context_hv_data[context].append(hv_history)

    # Convert lists to numpy arrays and ensure all sequences have the same length
    for context in context_keys:
        # Pad shorter sequences with their last value
        max_len = max(len(hv) for hv in context_hv_data[context])
        padded_hvs = []
        for hv_seq in context_hv_data[context]:
            if len(hv_seq) < max_len:
                padding = [hv_seq[-1]] * (max_len - len(hv_seq))
                padded_hvs.append(np.concatenate([hv_seq, padding]))
            else:
                padded_hvs.append(hv_seq)
        context_hv_data[context] = np.array(padded_hvs)

    return context_hv_data


def analyze_multi_problems_with_custom_hv(problems_config, methods_config, cut_size=50,
                                          use_final_pareto=False, figure_size=(20, 12),
                                          save_filename=None):
    """
    Analyze and plot results for multiple problems using custom hypervolume calculations.
    Creates a single figure with 7 subplots (4 top, 3 bottom) and shared legend.

    Parameters:
    -----------
    problems_config : list of dict
        List of problem configurations, each with keys:
        - 'name': problem name
        - 'base_path': base path to results
        - 'n_runs': number of runs
        - 'reference_point': reference point for HV calculation
        - 'dim': problem dimension
        - 'obj': number of objectives
        - 'epsilon': epsilon value
        - 'title': (optional) custom title for subplot
    methods_config : list of dict
        List of method configurations (same format as original)
    cut_size : int, optional
        Number of iterations to include in plots (default: 50)
    use_final_pareto : bool, optional
        If True, compute HV based only on the final Pareto front
    figure_size : tuple, optional
        Figure size (width, height)
    save_filename : str, optional
        Custom filename for saving. If None, auto-generates filename

    Returns:
    --------
    dict
        Statistics for each problem and method combination
    """

    if len(problems_config) != 7:
        raise ValueError("This function is designed for exactly 7 problems")

    # Set up colors for methods if not provided
    color_cycle = list(mcolors.TABLEAU_COLORS)
    for i, config in enumerate(methods_config):
        if 'color' not in config:
            config['color'] = color_cycle[i % len(color_cycle)]

    # Create figure with 7 subplots: 4 on top, 3 on bottom (centered)
    fig = plt.figure(figsize=figure_size)

    # Use GridSpec for precise control over subplot positioning
    from matplotlib.gridspec import GridSpec

    # Create a 2x12 grid (12 columns allows for fine positioning control)
    # gs = GridSpec(2, 12, figure=fig, hspace=0.3, wspace=0.3)
    gs = GridSpec(2, 12, figure=fig, hspace=0.23, wspace=0.6)

    # Define subplot positions for perfectly centered layout
    # Top row: 4 subplots, each taking 3 columns (3*4=12, perfectly fills the row)
    top_positions = [
        gs[0, 0:3],  # Subplot 1: columns 0-2
        gs[0, 3:6],  # Subplot 2: columns 3-5
        gs[0, 6:9],  # Subplot 3: columns 6-8
        gs[0, 9:12]  # Subplot 4: columns 9-11
    ]

    # Bottom row: 3 subplots, each taking 3 columns, with 1.5 column gaps for centering
    # Gap of 1.5 columns on each side: 1.5 + 3*3 + 1.5*2 = 1.5 + 9 + 3 = 13.5
    # We use 12 columns, so we use gaps of 1.5 ≈ 1-2 columns
    bottom_positions = [
        gs[1, 1:4],  # Subplot 5: columns 1-3 (1 column left gap)
        gs[1, 4:7],  # Subplot 6: columns 4-6 (centered)
        gs[1, 7:10]  # Subplot 7: columns 7-9 (1 column right gap)
    ]

    # Combine all positions
    subplot_positions = top_positions + bottom_positions

    all_stats = {}

    # Process each problem
    for prob_idx, problem_config in enumerate(problems_config):
        print(f"Processing problem {prob_idx + 1}/7: {problem_config['name']}")

        # Get problem-specific method configurations
        current_methods_config = get_problem_specific_methods_config(
            methods_config, problem_config, prob_idx
        )

        # Load data for all methods for this problem
        problem_hv_data = {}

        for config in current_methods_config:
            # Format the timestamp
            formatted_params = format_timestamp_params(config['timestamp_params'])
            timestamp = config['timestamp_template'].format(**formatted_params)

            # Load the data
            data = load_run_data(
                problem_config['base_path'],
                config['method_dir'],
                timestamp,
                problem_config['n_runs']
            )

            if not data:
                print(f"Warning: No data found for method {config['name']} in problem {problem_config['name']}")
                continue

            # Compute hypervolumes
            problem_hv_data[config['name']] = extract_pareto_fronts_and_compute_hv(
                data, problem_config['reference_point'], use_final_pareto
            )

        if not problem_hv_data:
            print(f"Error: No data found for any method in problem {problem_config['name']}")
            continue

        # Create subplot for this problem using GridSpec position
        ax = fig.add_subplot(subplot_positions[prob_idx])

        # Calculate and plot overall statistics for each method
        problem_stats = {}

        for method_name, method_data in problem_hv_data.items():
            # Find the original method config for color
            original_config = next((m for m in current_methods_config if m['name'] == method_name), None)
            if not original_config:
                continue

            # Calculate average across all contexts
            overall_mean = np.mean([np.mean(data, axis=0) for data in method_data.values()], axis=0)
            overall_std = np.mean([np.std(data, axis=0) for data in method_data.values()], axis=0)

            # Ensure we don't exceed the available data length
            plot_size = min(cut_size, len(overall_mean))
            x = np.arange(plot_size)

            # Plot overall statistics
            # ax.plot(x, overall_mean[:plot_size], label=method_name,
            #         color=original_config['color'], linewidth=2)
            # ax.fill_between(
            #     x,
            #     overall_mean[:plot_size] - overall_std[:plot_size] / 2,
            #     overall_mean[:plot_size] + overall_std[:plot_size] / 2,
            #     alpha=0.2,
            #     color=original_config['color']
            # )
            line = ax.plot(x, overall_mean[:plot_size], label=method_name,
                           color=original_config['color'], linewidth=2.5)

            # Add dots every few points to make the curve clearer
            dot_interval = max(1, plot_size // 15)  # Show ~15 dots maximum
            dot_indices = np.arange(0, plot_size, dot_interval)
            ax.scatter(dot_indices, overall_mean[dot_indices],
                       color=original_config['color'], s=25, zorder=5, alpha=0.8)

            ax.fill_between(
                x,
                overall_mean[:plot_size] - overall_std[:plot_size] / 2,
                overall_mean[:plot_size] + overall_std[:plot_size] / 2,
                alpha=0.15,  # Slightly more transparent to reduce visual clutter
                color=original_config['color']
            )

            # Store final statistics
            problem_stats[method_name] = {
                'final_mean_hv': float(overall_mean[plot_size - 1]),
                'final_std_hv': float(overall_std[plot_size - 1])
            }

        # Customize subplot
        # AFTER:
        subplot_title = problem_config.get('title', problem_config['name'])
        ax.set_title(f'{subplot_title}', fontsize=15, fontweight='bold', pad=15)
        # ax.set_xlabel('Iteration', fontsize=11)
        # ax.set_ylabel('Hypervolume', fontsize=11)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Improve tick formatting
        ax.tick_params(axis='both', which='major', labelsize=9)

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_linewidth(1)

        # Store stats for this problem
        all_stats[problem_config['name']] = problem_stats

    # Create a single shared legend at the bottom
    # Get handles and labels from the last subplot that has data
    handles, labels = ax.get_legend_handles_labels()

    # Add the shared legend below all subplots
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.00),
               ncol=len(labels), fontsize=16.5, frameon=True, fancybox=True, shadow=True)

    # Add main title
    # fig.suptitle('Average Hypervolume Progress Across All Tasks',
    #              fontsize=16, fontweight='bold', y=0.95)

    # Adjust layout to make room for legend
    # plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save the figure
    if save_filename is None:
        problem_names_str = "_".join([p['name'] for p in problems_config])
        method_names_str = "_".join([m['name'].replace(" ", "-") for m in methods_config])
        save_filename = f'multi_problem_hv_comparison_{problem_names_str}_{method_names_str}.png'

    # Save to the first problem's base path or current directory
    if problems_config:
        save_path = Path(problems_config[0]['base_path']).parent / save_filename
    else:
        save_path = Path(save_filename)

    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Multi-problem comparison saved to: {save_path}")

    return all_stats


def get_problem_specific_methods_config(base_methods_config, problem_config, problem_index):
    """
    Get method configurations with problem-specific overrides.

    Parameters:
    -----------
    base_methods_config : list of dict
        Base method configurations
    problem_config : dict
        Current problem configuration
    problem_index : int
        Index of current problem (0-based)

    Returns:
    --------
    list of dict
        Method configurations with problem-specific overrides applied
    """
    # Deep copy to avoid modifying original configs
    methods_config = deepcopy(base_methods_config)

    # Apply problem-specific parameters to all methods
    for config in methods_config:
        if 'timestamp_params' in config:
            # Update common parameters from problem config
            config['timestamp_params'].update({
                'problem': problem_config['name'],
                'dim': problem_config['dim'],
                'obj': problem_config['obj'],
                'epsilon': problem_config['epsilon']
            })

    # Apply method-specific overrides for this problem
    if 'method_overrides' in problem_config:
        for method_name, overrides in problem_config['method_overrides'].items():
            # Find the method configuration
            method_config = next((m for m in methods_config if m['name'] == method_name), None)
            if method_config:
                # Apply overrides
                for key, value in overrides.items():
                    if key == 'timestamp_params':
                        method_config['timestamp_params'].update(value)
                    else:
                        method_config[key] = value

    return methods_config


def format_timestamp_params(timestamp_params):
    """Format timestamp parameters with proper floating-point formatting."""
    formatted_params = {}
    for key, value in timestamp_params.items():
        if isinstance(value, float):
            formatted_params[key] = f"{value:.2f}"
        else:
            formatted_params[key] = value
    return formatted_params


def print_multi_problem_stats(all_stats):
    """
    Print formatted statistics for all problems and methods.

    Parameters:
    -----------
    all_stats : dict
        Statistics returned from analyze_multi_problems_with_custom_hv
    """
    print("\n" + "=" * 80)
    print("MULTI-PROBLEM HYPERVOLUME STATISTICS")
    print("=" * 80)

    for problem_name, problem_stats in all_stats.items():
        print(f"\n{problem_name.upper()}:")
        print("-" * (len(problem_name) + 1))

        for method_name, stats in problem_stats.items():
            print(f"  {method_name:15}: {stats['final_mean_hv']:.4f} ± {stats['final_std_hv']:.4f}")


# Example usage function
def example_multi_problem_usage():
    """
    Example of how to use the multi-problem analysis function.
    """

    # Define the 7 problems to analyze
    problems_config = [
        {
            'name': 'dtlz1',
            'base_path': 'result/dtlz1',
            'n_runs': 5,
            'reference_point': np.array([2.0, 2.0]) * 100,
            'dim': 8,
            'obj': 2,
            'epsilon': 1.00,
            'title': 'DTLZ-1',
            # Example: Override MOBO settings for this problem only
            'method_overrides': {
                'UCB-MOBO': {
                    'timestamp_params': {'epsilon': 0.01}  # Different epsilon for MOBO on DTLZ2
                },
                'PMT-MOBO': {
                    'timestamp_params': {'gp_type': 'ExactGP'}  # Different epsilon for MOBO on DTLZ2
                },
                'PMT-MOBO-DDIM':{
                    'method_dir':'DDIM-CMOBO_20steps_200_8_2_0.1'
                }
            }
        },
        {
            'name': 'dtlz2',
            'base_path': 'result/dtlz2',
            'n_runs': 5,
            'reference_point': np.array([2.0, 2.0]),
            'dim': 8,
            'obj': 2,
            'epsilon': 1.00,
            'title': 'DTLZ-2',
            'method_overrides': {
                'UCB-MOBO': {
                    'timestamp_params': {'epsilon': 0.01}  # Different epsilon for MOBO on DTLZ2
                },
                 'PMT-MOBO': {
                    'timestamp_params': {'gp_type': 'ExactGP'}  # Different epsilon for MOBO on DTLZ2
                },
                # 'PMT-MOBO-DDIM': {
                #     'method_dir': 'DDIM-CMOBO_20steps_200_8_2_0.1'
                # }
            }
        },
        {
            'name': 'dtlz3',
            'base_path': 'result/dtlz3',
            'n_runs': 5,
            'reference_point': np.array([2.0, 2.0]) * 100,
            'dim': 8,
            'obj': 2,
            'epsilon': 1.00,
            'title': 'DTLZ-3',
            'method_overrides': {
                'UCB-MOBO': {
                    'timestamp_params': {'epsilon': 0.01}  # Different epsilon for MOBO on DTLZ2
                },
                'PMT-MOBO': {
                    'timestamp_params': {'gp_type': 'ExactGP'}  # Different epsilon for MOBO on DTLZ2
                },
                'PMT-MOBO-DDIM': {
                    'method_dir': 'DDIM-CMOBO_20steps_200_8_2_0.1'
                }
            }
        },
        {
            'name': 'lamp',
            'base_path': 'result/lamp',
            'n_runs': 5,
            'reference_point': np.array([1.0, 1.0, 1.0]),
            'dim': 9,
            'obj': 3,
            'epsilon': 1.00,
            'title': 'Lamp Generation',
            'method_overrides': {
                'PMT-MOBO-DDIM': {
                    'method_dir': 'DDIM-CMOBO_20steps_200_8_2_0.1'
                },
                'PMT-MOBO': {
                    'timestamp_params': {'gp_type': 'ExactGP',
                                         'epsilon': 0.01}  # Different epsilon for MOBO on DTLZ2
                },
            }
        },
        {
            'name': 'gridshell',
            'base_path': 'result/gridshell',
            'n_runs': 5,
            'reference_point': np.array([1.0, 1.0]),
            'dim': 9,
            'obj': 2,
            'epsilon': 1.00,
            'title': 'Solar Rooftop Generation',
            'method_overrides': {
                'PMT-MOBO': {
                    'timestamp_params': {'gp_type': 'ExactGP',
                                         'epsilon': 0.01}  # Different epsilon for MOBO on DTLZ2
                },
                # 'PMT-MOBO-DDIM': {
                #     'method_dir': 'DDIM-CMOBO_20steps_200_8_2_0.1'
                # }
            }
        },
        {
            'name': 'magnetic_sifter',
            'base_path': 'result/magnetic_sifter',
            'n_runs': 5,
            'reference_point': np.array([1.0, 1.0, 1.0]),
            'dim': 3,
            'obj': 3,
            'epsilon': 1.00,
            'title': 'Magnetic Sifter Design'
        },
        {
            'name': 'bicopter',
            'base_path': 'result/bicopter',
            'n_runs': 5,
            'reference_point': np.array([1.0, 1.0]),
            'dim': 12,
            'obj': 2,
            'epsilon': 1.00,
            'title': 'UAV Controller Design',
            'method_overrides': {
                'PMT-MOBO': {
                    'timestamp_params': {'gp_type': 'ExactGP',
                                         'epsilon': 0.01}  # Different epsilon for MOBO on DTLZ2
                },
                # 'PMT-MOBO-DDIM': {
                #     'method_dir': 'DDIM-CMOBO_20steps_200_8_2_0.1'
                # }
            }
        },
    ]

    # Define base methods configuration (same as original)
    methods_config = [
        {
            'name': 'UCB-MOBO',
            'method_dir': 'MOBO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_hv",
            'timestamp_params': {},  # Will be filled by problem config
            'color': 'blue'
        },
        {
            'name': 'qPAREGO',
            'method_dir': 'ParEGO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_parego_test_hv",
            'timestamp_params': {},
            'color': 'purple'
        },
        {
            'name': 'qEHVI',
            'method_dir': 'EHVI3',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_ehvi_test_hv",
            'timestamp_params': {},
            'color': 'brown'
        },
        {
            'name': 'PSL-MOBO',
            'method_dir': 'PSLMOBO3',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_pslmobo_test_hv",
            'timestamp_params': {},
            'color': 'pink'
        },
        {
            'name': 'PMT-MOBO',
            'method_dir': 'CMOBO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_{gp_type}_hv_constrain",
            'timestamp_params': {
                'gp_type': 'CustomGP'  # Default GP type for P-MOBO-VAE
            },
            'color': 'orange'
        },
        {
            'name': 'PMT-MOBO-VAE',
            'method_dir': 'betaVAE-CMOBO-nosigmoid_aug_2_0.1',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_{gp_type}_hv_constrain",
            'timestamp_params': {
                'gp_type': 'CustomGP'  # Default GP type for P-MOBO-VAE
            },
            'color': 'green'
        },
        {
            'name': 'PMT-MOBO-DDIM',
            'method_dir': 'DDIM-CMOBO_20steps_100_16_0.5_3_0.1',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_{gp_type}_hv_constrain",
            'timestamp_params': {
                'gp_type': 'CustomGP'  # Default GP type for P-MOBO-VAE
            },
            'color': 'red'
        },
    ]

    # Run the multi-problem analysis
    all_stats = analyze_multi_problems_with_custom_hv(
        problems_config=problems_config,
        methods_config=methods_config,
        cut_size=50,
        use_final_pareto=False,
        figure_size=(20, 9),
        save_filename='comprehensive_multi_problem_comparison.png'
    )

    # Print the statistics
    print_multi_problem_stats(all_stats)

    return all_stats


if __name__ == "__main__":
    stats = example_multi_problem_usage()
    # example_custom_hv_usage()
    # Configuration
    # problem_name = "dtlz3"
    # problem_dim = 5
    # problem_obj = 2
    # problem_beta = 1.00
    # base_path = f"result/{problem_name}"
    # timestamp = "{}_{}_{}_{:.2f}_test".format(problem_name, problem_dim, problem_obj, problem_beta)
    # n_runs = 5  # Adjust based on your actual number of runs

    # Run analysis
    # stats = analyze_and_plot_results(base_path, timestamp, n_runs, problem_name)
    # tot_path = "./result/CMOBO_optimization_history_dtlz-2_5_2_run_0.pth"
    # result = torch.load(tot_path)
    # print(result.keys)

    # if stats:
    #     # Print overall statistics
    #     print("\nOverall Statistics:")
    #     print("\nMOBO:")
    #     print(f"Final Mean HV: {stats['mobo']['final_mean_hv']:.4f}")
    #     print(f"Final Std HV: {stats['mobo']['final_std_hv']:.4f}")
    #     print("\nCMOBO-e:")
    #     print(f"Final Mean HV: {stats['cmobo_e']['final_mean_hv']:.4f}")
    #     print(f"Final Std HV: {stats['cmobo_e']['final_std_hv']:.4f}")
    #     print("\nCMOBO-r:")
    #     print(f"Final Mean HV: {stats['cmobo_r']['final_mean_hv']:.4f}")
    #     print(f"Final Std HV: {stats['cmobo_r']['final_std_hv']:.4f}")