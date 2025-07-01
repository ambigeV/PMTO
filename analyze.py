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
    comparison_filename = f'custom_hv_comparison_plot_fine_{dim}_{problem_name}_{method_names_str}_ref_{ref_point_str}.png'
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

    plt.title(f'Average Custom Hypervolume Progress Across All Tasks - {problem_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Hypervolume')
    plt.legend()
    plt.grid(True)

    # Save the summary plot
    summary_filename = f'custom_hv_summary_comparison_fine_{dim}_{problem_name}_{method_names_str}_ref_{ref_point_str}.png'
    summary_save_path = Path(base_path) / summary_filename
    plt.savefig(summary_save_path)
    plt.close()

    return stats


# Example usage
def example_custom_hv_usage():
    problem_name = "dtlz3"
    # problem_name = "lamp"
    base_path = f"result/{problem_name}"
    n_runs = 5
    example_dim = 8
    example_obj = 2

    # Define reference point for hypervolume calculation
    # Adjust this reference point according to your specific problem
    # reference_point = np.array([10.0, 10.0]) * 0.2  # For 2-objective problems
    reference_point = np.array([100.0, 100.0])*2.0  # For 2-objective problems
    # reference_point = torch.ones(3).numpy() * 1.0

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
                'epsilon': 0.01
            },
        },
        {
            'name': 'P-MOBO',
            'method_dir': 'CMOBO',
            'timestamp_template': "{problem}_{dim}_{obj}_{epsilon}_test_ExactGP_hv_constrain",
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
            'name': 'P-MOBO-DDIM',
            'method_dir': 'DDIM-CMOBO_20steps_200_8_2_0.1',
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
            std_values = np.std(method_data[context], axis=0) / 2

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


# def analyze_and_plot_results(base_path, timestamp, n_runs, problem_name):
#     """Analyze and plot results for both MOBO and CMOBO methods."""
#     # Load data for both methods
#     timestamp = "{}_{}_{}_{:.2f}_test_hv".format(problem_name, 5, 2, 0.01)
#     mobo_data = load_run_data(base_path, 'MOBO', timestamp, n_runs)
#     timestamp = "{}_{}_{}_{:.2f}_test".format(problem_name, 5, 2, 1.00)
#     exact_timestamp = timestamp + "_CustomGP_hv_constrain"
#     cmobo_data_exactgp = load_run_data(base_path, 'CMOBO', exact_timestamp, n_runs)
#     ard_timestamp = timestamp + "_CustomGP_hv_constrain"
#     cmobo_data_ardgp = load_run_data(base_path, 'betaVAE-CMOBO-nosigmoid_aug_2_0.1', ard_timestamp, n_runs)
#     # timestamp = "{}_{}_{}_{:.2f}_test_norm".format("dtlz2", 3, 2, 0.80)
#
#     if not mobo_data or not cmobo_data_exactgp or not cmobo_data_ardgp:
#         print("Error: No data found for one or both methods")
#         return None
#
#     # Process hypervolume data
#     mobo_hv_data = process_hypervolumes(mobo_data)
#     cmobo_hv_data_exactgp = process_hypervolumes(cmobo_data_exactgp)
#     cmobo_hv_data_ardgp = process_hypervolumes(cmobo_data_ardgp)
#
#     # Set up the plotting style
#     # plt.style.use('seaborn')
#     # sns.set_palette("husl")
#
#     # Create subplots for each context
#     n_contexts = len(mobo_hv_data)
#     n_cols = 5
#     n_rows = (n_contexts + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
#     fig.suptitle('Comparison of MOBO and CMOBO Hypervolume Progress', fontsize=16)
#
#     # Ensure axes is always 2D
#     if n_rows == 1:
#         axes = axes.reshape(1, -1)
#
#     # Plot each context
#     for idx, context in enumerate(mobo_hv_data.keys()):
#         row = idx // n_cols
#         col = idx % n_cols
#         ax = axes[row, col]
#
#         # Calculate statistics for MOBO
#         mobo_mean = np.mean(mobo_hv_data[context], axis=0)
#         mobo_std = np.std(mobo_hv_data[context], axis=0) / 2
#
#         # Calculate statistics for CMOBO
#         cmobo_mean_exact = np.mean(cmobo_hv_data_exactgp[context], axis=0)
#         cmobo_std_exact = np.std(cmobo_hv_data_exactgp[context], axis=0) / 2
#
#         # Calculate statistics for CMOBO
#         cmobo_mean_ard = np.mean(cmobo_hv_data_ardgp[context], axis=0)
#         cmobo_std_ard = np.std(cmobo_hv_data_ardgp[context], axis=0) / 2
#
#         # Plot means and standard deviations
#         x = np.arange(len(mobo_mean))
#
#         # Plot MOBO
#         cut_size = 50
#         ax.plot(x[:cut_size], mobo_mean[:cut_size], label='MOBO', color='blue')
#         ax.fill_between(x[:cut_size], mobo_mean[:cut_size] - mobo_std[:cut_size], mobo_mean[:cut_size] + mobo_std[:cut_size],
#                         alpha=0.2, color='blue')
#
#         # Plot CMOBO
#         ax.plot(x[:cut_size], cmobo_mean_exact[:cut_size], label='CMOBO_e', color='red')
#         ax.fill_between(x[:cut_size], cmobo_mean_exact[:cut_size] - cmobo_std_exact[:cut_size],
#                         cmobo_mean_exact[:cut_size] + cmobo_std_exact[:cut_size],
#                         alpha=0.2, color='red')
#
#         # Plot CMOBO
#         ax.plot(x[:cut_size], cmobo_mean_ard[:cut_size], label='P-MOBO', color='green')
#         ax.fill_between(x[:cut_size], cmobo_mean_ard[:cut_size] - cmobo_std_ard[:cut_size],
#                         cmobo_mean_ard[:cut_size] + cmobo_std_ard[:cut_size],
#                         alpha=0.2, color='green')
#
#         ax.set_title(f'Context {idx}')
#         ax.set_xlabel('Iteration')
#         ax.set_ylabel('Hypervolume')
#         ax.legend()
#         ax.grid(True)
#
#     # Remove empty subplots if any
#     for idx in range(len(mobo_hv_data), n_rows * n_cols):
#         row = idx // n_cols
#         col = idx % n_cols
#         fig.delaxes(axes[row, col])
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     # Save the comparison plot
#     save_path = Path(base_path) / f'comparison_plot_betavae_0.1_{timestamp}_hv.png'
#     plt.savefig(save_path)
#     plt.close()
#
#     # Create a summary plot of average performance across all contexts
#     plt.figure(figsize=(10, 6))
#
#     # Calculate average across all contexts
#     mobo_overall_mean = np.mean([np.mean(data, axis=0) for data in mobo_hv_data.values()], axis=0)
#     mobo_overall_std = np.mean([np.std(data, axis=0) for data in mobo_hv_data.values()], axis=0)
#
#     cmobo_overall_mean_exact = np.mean([np.mean(data, axis=0) for data in cmobo_hv_data_exactgp.values()], axis=0)
#     cmobo_overall_std_exact = np.mean([np.std(data, axis=0) for data in cmobo_hv_data_exactgp.values()], axis=0)
#
#     cmobo_overall_mean_ard = np.mean([np.mean(data, axis=0) for data in cmobo_hv_data_ardgp.values()], axis=0)
#     cmobo_overall_std_ard = np.mean([np.std(data, axis=0) for data in cmobo_hv_data_ardgp.values()], axis=0)
#
#     x = np.arange(len(mobo_overall_mean))
#     y = np.arange(len(cmobo_overall_mean_exact))
#     z = np.arange(len(cmobo_overall_mean_ard))
#
#     plt.plot(x[:cut_size], mobo_overall_mean[:cut_size], label='MOBO', color='blue')
#     plt.fill_between(x[:cut_size], (mobo_overall_mean - mobo_overall_std)[:cut_size],
#                      (mobo_overall_mean + mobo_overall_std)[:cut_size],
#                      alpha=0.2, color='blue')
#
#     plt.plot(y[:cut_size], cmobo_overall_mean_exact[:cut_size], label='P-MOBO', color='red')
#     plt.fill_between(y[:cut_size], (cmobo_overall_mean_exact - cmobo_overall_std_exact)[:cut_size],
#                      (cmobo_overall_mean_exact + cmobo_overall_std_exact)[:cut_size],
#                      alpha=0.2, color='red')
#
#     plt.plot(z[:cut_size], cmobo_overall_mean_ard[:cut_size], label='P-MOBO-VAE', color='green')
#     plt.fill_between(z[:cut_size], (cmobo_overall_mean_ard - cmobo_overall_std_ard)[:cut_size],
#                      (cmobo_overall_mean_ard + cmobo_overall_std_ard)[:cut_size],
#                      alpha=0.2, color='green')
#
#     plt.title('Average Hypervolume Progress Across All Tasks')
#     plt.xlabel('Iteration')
#     plt.ylabel('Average Hypervolume')
#     plt.legend()
#     plt.grid(True)
#
#     timestamp = timestamp + "_clean"
#     # Save the summary plot
#     summary_save_path = Path(base_path) / f'summary_comparison_plot_betavae_0.1_{timestamp}_hv.png'
#     plt.savefig(summary_save_path)
#     plt.close()
#
#     # Calculate and return overall statistics
#     stats = {
#         'mobo': {
#             'final_mean_hv': float(mobo_overall_mean[cut_size - 1]),
#             'final_std_hv': float(mobo_overall_std[cut_size - 1])
#         },
#         'cmobo_e': {
#             'final_mean_hv': float(cmobo_overall_mean_exact[cut_size - 1]),
#             'final_std_hv': float(cmobo_overall_std_exact[cut_size - 1])
#         },
#         'cmobo_r': {
#             'final_mean_hv': float(cmobo_overall_mean_ard[cut_size - 1]),
#             'final_std_hv': float(cmobo_overall_std_ard[cut_size - 1])
#         }
#     }
#
#     return stats


if __name__ == "__main__":
    # example_usage()
    example_custom_hv_usage()
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