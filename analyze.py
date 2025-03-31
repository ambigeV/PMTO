import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import seaborn as sns


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


def analyze_and_plot_results(base_path, timestamp, n_runs, problem_name):
    """Analyze and plot results for both MOBO and CMOBO methods."""
    # Load data for both methods
    timestamp = "{}_{}_{}_{:.2f}_test_hv".format(problem_name, 5, 2, 0.01)
    mobo_data = load_run_data(base_path, 'MOBO', timestamp, n_runs)
    timestamp = "{}_{}_{}_{:.2f}_test".format(problem_name, 5, 2, 1.00)
    exact_timestamp = timestamp + "_CustomGP_hv_constrain"
    cmobo_data_exactgp = load_run_data(base_path, 'CMOBO', exact_timestamp, n_runs)
    ard_timestamp = timestamp + "_CustomGP_hv_constrain"
    cmobo_data_ardgp = load_run_data(base_path, 'betaVAE-CMOBO-nosigmoid_aug_2_0.1', ard_timestamp, n_runs)
    # timestamp = "{}_{}_{}_{:.2f}_test_norm".format("dtlz2", 3, 2, 0.80)

    if not mobo_data or not cmobo_data_exactgp or not cmobo_data_ardgp:
        print("Error: No data found for one or both methods")
        return None

    # Process hypervolume data
    mobo_hv_data = process_hypervolumes(mobo_data)
    cmobo_hv_data_exactgp = process_hypervolumes(cmobo_data_exactgp)
    cmobo_hv_data_ardgp = process_hypervolumes(cmobo_data_ardgp)

    # Set up the plotting style
    # plt.style.use('seaborn')
    # sns.set_palette("husl")

    # Create subplots for each context
    n_contexts = len(mobo_hv_data)
    n_cols = 5
    n_rows = (n_contexts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle('Comparison of MOBO and CMOBO Hypervolume Progress', fontsize=16)

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each context
    for idx, context in enumerate(mobo_hv_data.keys()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Calculate statistics for MOBO
        mobo_mean = np.mean(mobo_hv_data[context], axis=0)
        mobo_std = np.std(mobo_hv_data[context], axis=0) / 2

        # Calculate statistics for CMOBO
        cmobo_mean_exact = np.mean(cmobo_hv_data_exactgp[context], axis=0)
        cmobo_std_exact = np.std(cmobo_hv_data_exactgp[context], axis=0) / 2

        # Calculate statistics for CMOBO
        cmobo_mean_ard = np.mean(cmobo_hv_data_ardgp[context], axis=0)
        cmobo_std_ard = np.std(cmobo_hv_data_ardgp[context], axis=0) / 2

        # Plot means and standard deviations
        x = np.arange(len(mobo_mean))

        # Plot MOBO
        cut_size = 50
        ax.plot(x[:cut_size], mobo_mean[:cut_size], label='MOBO', color='blue')
        ax.fill_between(x[:cut_size], mobo_mean[:cut_size] - mobo_std[:cut_size], mobo_mean[:cut_size] + mobo_std[:cut_size],
                        alpha=0.2, color='blue')

        # Plot CMOBO
        ax.plot(x[:cut_size], cmobo_mean_exact[:cut_size], label='CMOBO_e', color='red')
        ax.fill_between(x[:cut_size], cmobo_mean_exact[:cut_size] - cmobo_std_exact[:cut_size],
                        cmobo_mean_exact[:cut_size] + cmobo_std_exact[:cut_size],
                        alpha=0.2, color='red')

        # Plot CMOBO
        ax.plot(x[:cut_size], cmobo_mean_ard[:cut_size], label='P-MOBO', color='green')
        ax.fill_between(x[:cut_size], cmobo_mean_ard[:cut_size] - cmobo_std_ard[:cut_size],
                        cmobo_mean_ard[:cut_size] + cmobo_std_ard[:cut_size],
                        alpha=0.2, color='green')

        ax.set_title(f'Context {idx}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypervolume')
        ax.legend()
        ax.grid(True)

    # Remove empty subplots if any
    for idx in range(len(mobo_hv_data), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the comparison plot
    save_path = Path(base_path) / f'comparison_plot_betavae_0.1_{timestamp}_hv.png'
    plt.savefig(save_path)
    plt.close()

    # Create a summary plot of average performance across all contexts
    plt.figure(figsize=(10, 6))

    # Calculate average across all contexts
    mobo_overall_mean = np.mean([np.mean(data, axis=0) for data in mobo_hv_data.values()], axis=0)
    mobo_overall_std = np.mean([np.std(data, axis=0) for data in mobo_hv_data.values()], axis=0)

    cmobo_overall_mean_exact = np.mean([np.mean(data, axis=0) for data in cmobo_hv_data_exactgp.values()], axis=0)
    cmobo_overall_std_exact = np.mean([np.std(data, axis=0) for data in cmobo_hv_data_exactgp.values()], axis=0)

    cmobo_overall_mean_ard = np.mean([np.mean(data, axis=0) for data in cmobo_hv_data_ardgp.values()], axis=0)
    cmobo_overall_std_ard = np.mean([np.std(data, axis=0) for data in cmobo_hv_data_ardgp.values()], axis=0)

    x = np.arange(len(mobo_overall_mean))
    y = np.arange(len(cmobo_overall_mean_exact))
    z = np.arange(len(cmobo_overall_mean_ard))

    plt.plot(x[:cut_size], mobo_overall_mean[:cut_size], label='MOBO', color='blue')
    plt.fill_between(x[:cut_size], (mobo_overall_mean - mobo_overall_std)[:cut_size],
                     (mobo_overall_mean + mobo_overall_std)[:cut_size],
                     alpha=0.2, color='blue')

    plt.plot(y[:cut_size], cmobo_overall_mean_exact[:cut_size], label='P-MOBO', color='red')
    plt.fill_between(y[:cut_size], (cmobo_overall_mean_exact - cmobo_overall_std_exact)[:cut_size],
                     (cmobo_overall_mean_exact + cmobo_overall_std_exact)[:cut_size],
                     alpha=0.2, color='red')

    plt.plot(z[:cut_size], cmobo_overall_mean_ard[:cut_size], label='P-MOBO-VAE', color='green')
    plt.fill_between(z[:cut_size], (cmobo_overall_mean_ard - cmobo_overall_std_ard)[:cut_size],
                     (cmobo_overall_mean_ard + cmobo_overall_std_ard)[:cut_size],
                     alpha=0.2, color='green')

    plt.title('Average Hypervolume Progress Across All Tasks')
    plt.xlabel('Iteration')
    plt.ylabel('Average Hypervolume')
    plt.legend()
    plt.grid(True)

    timestamp = timestamp + "_clean"
    # Save the summary plot
    summary_save_path = Path(base_path) / f'summary_comparison_plot_betavae_0.1_{timestamp}_hv.png'
    plt.savefig(summary_save_path)
    plt.close()

    # Calculate and return overall statistics
    stats = {
        'mobo': {
            'final_mean_hv': float(mobo_overall_mean[cut_size - 1]),
            'final_std_hv': float(mobo_overall_std[cut_size - 1])
        },
        'cmobo_e': {
            'final_mean_hv': float(cmobo_overall_mean_exact[cut_size - 1]),
            'final_std_hv': float(cmobo_overall_std_exact[cut_size - 1])
        },
        'cmobo_r': {
            'final_mean_hv': float(cmobo_overall_mean_ard[cut_size - 1]),
            'final_std_hv': float(cmobo_overall_std_ard[cut_size - 1])
        }
    }

    return stats


if __name__ == "__main__":
    # Configuration
    problem_name = "dtlz3"
    problem_dim = 5
    problem_obj = 2
    problem_beta = 1.00
    base_path = f"result/{problem_name}"
    timestamp = "{}_{}_{}_{:.2f}_test".format(problem_name, problem_dim, problem_obj, problem_beta)
    n_runs = 5  # Adjust based on your actual number of runs

    # Run analysis
    stats = analyze_and_plot_results(base_path, timestamp, n_runs, problem_name)
    # tot_path = "./result/CMOBO_optimization_history_dtlz-2_5_2_run_0.pth"
    # result = torch.load(tot_path)
    # print(result.keys)

    if stats:
        # Print overall statistics
        print("\nOverall Statistics:")
        print("\nMOBO:")
        print(f"Final Mean HV: {stats['mobo']['final_mean_hv']:.4f}")
        print(f"Final Std HV: {stats['mobo']['final_std_hv']:.4f}")
        print("\nCMOBO-e:")
        print(f"Final Mean HV: {stats['cmobo_e']['final_mean_hv']:.4f}")
        print(f"Final Std HV: {stats['cmobo_e']['final_std_hv']:.4f}")
        print("\nCMOBO-r:")
        print(f"Final Mean HV: {stats['cmobo_r']['final_mean_hv']:.4f}")
        print(f"Final Std HV: {stats['cmobo_r']['final_std_hv']:.4f}")