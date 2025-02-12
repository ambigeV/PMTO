from PMTO import BayesianOptimization, ObjectiveFunction, \
    MultiObjectiveFunction, MultiObjectiveBayesianOptimization, \
    ContextualBayesianOptimization, ContextualMultiObjectiveFunction, \
    ContextualMultiObjectiveBayesianOptimization, PseudoObjectiveFunction
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import os

# Test Case for GP-SOO
def test_bo():
    cur_dim = 5
    objective = ObjectiveFunction(func_name='ackley', dim=cur_dim)
    X_train = torch.rand(50, cur_dim)
    y_train = torch.tensor([objective.evaluate(x) for x in X_train])

    # print(X_train.shape, y_train.shape)

    inducing_points = torch.rand(20, cur_dim)
    bo = BayesianOptimization(objective_func=objective,
                              inducing_points=inducing_points,
                              model_type="EXACT-GP",
                              optimizer_type="adam")

    X_final, y_final, best_y = bo.optimize(X_train, y_train, n_iter=100, beta=1.0)


def test_multiobjective_functions():
    """Test cases for DTLZ test problems"""
    print("Testing DTLZ test problems...")

    # Test DTLZ1 with 2 objectives
    n_var = 7  # n = m + k - 1, where m=2, k=6
    problem = MultiObjectiveFunction(
        func_name='dtlz1',
        n_objectives=2,
        n_variables=n_var
    )

    # Generate test points
    X_test = torch.rand(50, n_var)
    Y_test = problem.evaluate(X_test)

    print(f"\nDTLZ1 Test:")
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {Y_test.shape}")
    print(f"Output range: [{Y_test.min():.3f}, {Y_test.max():.3f}]")

    # Test DTLZ2 with 3 objectives
    n_var = 12  # n = m + k - 1, where m=3, k=10
    problem = MultiObjectiveFunction(
        func_name='dtlz2',
        n_objectives=2,
        n_variables=n_var
    )

    X_test = torch.rand(50, n_var)
    Y_test = problem.evaluate(X_test)

    print(f"\nDTLZ2 Test:")
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {Y_test.shape}")
    print(f"Output range: [{Y_test.min():.3f}, {Y_test.max():.3f}]")

    # Visualize 2D results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    problem_2d = MultiObjectiveFunction('dtlz1', n_objectives=2)
    X_vis = torch.rand(100, problem_2d.n_variables)
    Y_vis = problem_2d.evaluate(X_vis)
    plt.scatter(Y_vis[:, 0].numpy(), Y_vis[:, 1].numpy())
    plt.title('DTLZ1 (2 objectives)')
    plt.xlabel('f1')
    plt.ylabel('f2')

    plt.subplot(1, 2, 2)
    problem_2d = MultiObjectiveFunction('dtlz2', n_objectives=2)
    Y_vis = problem_2d.evaluate(X_vis)
    plt.scatter(Y_vis[:, 0].numpy(), Y_vis[:, 1].numpy())
    plt.title('DTLZ2 (2 objectives)')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.tight_layout()
    plt.show()


def test_mobo():
    """Test cases for Multi-Objective Bayesian Optimization"""
    print("\nTesting Multi-Objective Bayesian Optimization...")

    # Create test problem
    n_var = 7
    n_obj = 2
    problem = MultiObjectiveFunction(
        func_name='dtlz2',
        n_objectives=n_obj,
        n_variables=n_var
    )

    # Initial training data
    X_train = torch.rand(50, n_var)
    Y_train = problem.evaluate(X_train)

    # Set reference point for hypervolume calculation
    # ref_point = torch.ones(n_obj) * 1.1  # Slightly above maximum possible value

    # Initialize MOBO
    # inducing_points = torch.rand(20, n_var)
    # print(problem.output_dim)
    mobo = MultiObjectiveBayesianOptimization(
        objective_func=problem,
        # reference_point=ref_point,
        # inducing_points=inducing_points,
        model_type="EXACT-GP",
        optimizer_type="adam"
    )

    # Run optimization
    X_final, Y_final = mobo.optimize(
        X_train=X_train,
        Y_train=Y_train,
        n_iter=50,
        beta=1.0
    )
    hv_history = mobo.hv_history

    print(f"\nMOBO Results:")
    print(f"Initial points: {len(X_train)}")
    print(f"Final points: {len(X_final)}")
    print(f"Initial HV: {hv_history[0]:.3f}")
    print(f"Final HV: {hv_history[-1]:.3f}")

    # Visualize optimization progress
    plt.figure(figsize=(15, 5))

    # Plot hypervolume history
    plt.subplot(1, 2, 1)
    plt.plot(hv_history)
    plt.title('Hypervolume Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')

    # Plot final Pareto front approximation
    plt.subplot(1, 2, 2)
    plt.scatter(Y_final[:, 0].numpy(), Y_final[:, 1].numpy(),
                c='blue', label='All points')
    pareto_front = mobo.pareto_front
    plt.scatter(pareto_front[:, 0].numpy(),
                pareto_front[:, 1].numpy(),
                c='red', label='Pareto front')
    plt.title('Final Pareto Front Approximation')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_contextual_effects(func_name='dtlz2', n_objectives=2, n_samples=5000):

    """Visualize the effects of different contexts on the objective space."""
    problem = ContextualMultiObjectiveFunction(
        func_name=func_name,
        n_objectives=n_objectives
    )

    # Verify context dimension
    k = problem.context_dim
    print(f"\nProblem settings:")
    print(f"n_variables (n): {problem.n_variables}")
    print(f"n_objectives (m): {problem.n_objectives}")
    print(f"context_dim (k): {k} (should be n-m+1)")

    # Generate different contexts
    contexts = [
        torch.zeros(1, k),  # Zero context
        torch.ones(1, k),  # One context
        torch.rand(1, k),  # Random context
        0.5 * torch.ones(1, k)  # Mid context
    ]
    context_names = ['Zero Context', 'One Context', 'Random Context', 'Mid Context']

    # Setup plotting
    if n_objectives == 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        fig.suptitle(f'Contextual {func_name.upper()} - 2D Objective Space')

        for idx, (context, name) in enumerate(zip(contexts, context_names)):
            # Generate random samples
            X = torch.rand(n_samples, problem.n_variables)
            C = context.repeat(n_samples, 1)

            # Verify dimensions
            assert C.shape[1] == k, f"Context dimension mismatch: {C.shape[1]} != {k}"

            inputs = torch.cat([X, C], dim=1)
            Y = problem.evaluate(inputs)

            axes[idx].scatter(Y[:, 0].numpy(), Y[:, 1].numpy(), alpha=0.5)
            axes[idx].set_xlabel('f1')
            axes[idx].set_ylabel('f2')
            axes[idx].set_title(f'{name}\nShift:{problem.get_context_shift(context)[0]:.2f}, '
                                f'Power:{problem.get_context_power(context)[0]:.2f}')

    elif n_objectives == 3:
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Contextual {func_name.upper()} - 3D Objective Space')

        for idx, (context, name) in enumerate(zip(contexts, context_names)):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

            X = torch.rand(n_samples, problem.n_variables)
            C = context.repeat(n_samples, 1)

            assert C.shape[1] == k, f"Context dimension mismatch: {C.shape[1]} != {k}"

            inputs = torch.cat([X, C], dim=1)
            Y = problem.evaluate(inputs)

            ax.scatter(Y[:, 0].numpy(), Y[:, 1].numpy(), Y[:, 2].numpy(), alpha=0.5)
            ax.set_xlabel('f1')
            ax.set_ylabel('f2')
            ax.set_zlabel('f3')
            ax.set_title(f'{name}\nShift:{problem.get_context_shift(context)[0]:.2f}, '
                         f'Power:{problem.get_context_power(context)[0]:.2f}')

    plt.tight_layout()
    plt.show()


def random_test_obj():
    obj_func = ContextualMultiObjectiveFunction(func_name='dtlz2', n_objectives=2, n_variables=5)
    x = torch.rand(1000000, obj_func.input_dim + obj_func.context_dim)
    y = obj_func.evaluate(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    print(torch.min(y, dim=0))
    print(torch.max(y, dim=0))
    print(obj_func.nadir_point)


def random_test_context():
    obj_func = ContextualMultiObjectiveFunction(func_name='dtlz2', n_objectives=2, n_variables=5)
    x_fixed = torch.rand(1, obj_func.input_dim)
    contexts = torch.tensor([[0, 0.5], [0.5, 0.5], [1, 0.5]])
    for c in contexts:
        x_c = torch.cat([x_fixed, c.unsqueeze(0)], dim=1)
        y = obj_func.evaluate(x_c)
        print(f"Context: {c}, Output: {y}")


def random_test_boundary():
    obj_func = ContextualMultiObjectiveFunction(func_name='dtlz2', n_objectives=2, n_variables=5)
    x_min = torch.zeros(1, obj_func.input_dim + obj_func.context_dim)
    x_max = torch.ones(1, obj_func.input_dim + obj_func.context_dim)
    y_min = obj_func.evaluate(x_min)
    y_max = obj_func.evaluate(x_max)
    print(f"Min input: {y_min}, Max input: {y_max}")


def context_influence_test_grid():
    obj_func = ContextualMultiObjectiveFunction(func_name='dtlz2', n_objectives=2, n_variables=5)
    x_fixed = torch.rand(1, obj_func.input_dim)

    # Create a grid of contexts
    n_points = 100  # Number of points along each dimension
    c1 = np.linspace(0, 1, n_points)
    c2 = np.linspace(0, 1, n_points)
    c1_grid, c2_grid = np.meshgrid(c1, c2)
    contexts = torch.tensor(np.column_stack((c1_grid.ravel(), c2_grid.ravel())), dtype=torch.float32)

    results = []
    for c in contexts:
        x_c = torch.cat([x_fixed, c.unsqueeze(0)], dim=1)
        y = obj_func.evaluate(x_c)
        results.append(y.numpy())

    results = np.array(results)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i in range(2):  # For each objective
        im = axes[i].scatter(contexts[:, 0], contexts[:, 1], c=results[:, 0, i], cmap='viridis')
        axes[i].set_xlabel('Context 1')
        axes[i].set_ylabel('Context 2')
        axes[i].set_title(f'Objective {i + 1}')
        fig.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Min values: {results.min(axis=0)[0]}")
    print(f"Max values: {results.max(axis=0)[0]}")
    print(f"Mean values: {results.mean(axis=0)[0]}")
    print(f"Std values: {results.std(axis=0)[0]}")


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


def plot_contexts(contexts):
    if contexts.shape[1] == 2:
        plt.figure(figsize=(8, 8))
        plt.scatter(contexts[:, 0], contexts[:, 1])
        plt.xlabel('Context Dimension 1')
        plt.ylabel('Context Dimension 2')
        plt.title('Distribution of Contexts')
        plt.savefig('context_distribution.png')
        plt.close()
    elif contexts.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(contexts[:, 0], contexts[:, 1], contexts[:, 2])
        ax.set_xlabel('Context Dimension 1')
        ax.set_ylabel('Context Dimension 2')
        ax.set_zlabel('Context Dimension 3')
        ax.set_title('Distribution of Contexts')
        plt.savefig('context_distribution.png')
        plt.close()
    else:
        print("Cannot visualize contexts with more than 3 dimensions")


def optimization_loop_test(n_runs=1, n_iter=5):
    # Initialize the objective function
    obj_func = ContextualMultiObjectiveFunction(func_name='dtlz2', n_objectives=2, n_variables=5)

    # Set up fixed contexts using LHS
    n_contexts = 16
    contexts_file = 'data/context_{}_{}.pth'.format(n_contexts, obj_func.context_dim)

    if os.path.exists(contexts_file):
        contexts = torch.load(contexts_file)
    else:
        contexts = generate_and_save_contexts(n_contexts, obj_func.context_dim, contexts_file)

    # Plot the contexts
    plot_contexts(contexts)

    timestamp = "dtlz-2_5_2"

    for run in range(n_runs):
        print(f"Starting run {run + 1}/{n_runs}")

        # Initialize the optimizer
        optimizer = ContextualMultiObjectiveBayesianOptimization(
            objective_func=obj_func,
        )

        # Generate initial points
        n_initial_points = 20
        X_init = torch.zeros(n_initial_points * n_contexts, obj_func.input_dim + obj_func.context_dim)
        for i in range(n_contexts):
            start_idx = i * n_initial_points
            end_idx = (i + 1) * n_initial_points
            X_init[start_idx:end_idx, :obj_func.input_dim] = torch.rand(n_initial_points, obj_func.input_dim)
            X_init[start_idx:end_idx, obj_func.input_dim:] = contexts[i].repeat(n_initial_points, 1)

        # Evaluate initial points
        Y_init = obj_func.evaluate(X_init)

        # Run optimization
        X_opt, Y_opt = optimizer.optimize(X_init, Y_init, contexts, n_iter=n_iter)

        # Store results for this run
        run_data = {}
        for i, context in enumerate(contexts):
            context_key = tuple(context.numpy())
            if context_key in optimizer.context_pareto_fronts:
                run_data[context_key] = {
                    'pareto_set_history': optimizer.context_pareto_sets[context_key],
                    'pareto_front_history': optimizer.context_pareto_fronts[context_key],
                    'hv_history': optimizer.context_hv[context_key]
                }
                print(f"Run {run + 1}, Context {i}: Final hypervolume = {optimizer.context_hv[context_key][-1]:.4f}")
            else:
                print(f"Run {run + 1}, Context {i}: No Pareto front found")

        # Save individual run data
        save_path = f'result/CMOBO_optimization_history_{timestamp}_run_{run}.pth'
        torch.save(run_data, save_path)
        print(f"Run {run + 1} data saved to {save_path}")

        # Plot hypervolume history for this run
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f'Hypervolume History for Each Context (Run {run + 1})', fontsize=16)

        for i, context in enumerate(contexts):
            row = i // 4
            col = i % 4
            ax = axes[row, col]

            context_key = tuple(context.numpy())
            if context_key in run_data:
                hv_history = run_data[context_key]['hv_history']
                ax.plot(range(len(hv_history)), hv_history, label=f'Run {run + 1}')

            ax.set_title(f'Context {i}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Hypervolume')
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'result/CMOBO_hypervolume_history_grid_{timestamp}_run_{run}.png')
        plt.close()


def run_mobo_test(n_runs=1, n_iter=5):
    # Initialize the objective function
    obj_func = ContextualMultiObjectiveFunction(func_name='dtlz2', n_objectives=2, n_variables=5)

    # Set up fixed contexts using LHS
    n_contexts = 16
    contexts_file = 'data/context_{}_{}.pth'.format(n_contexts, obj_func.context_dim)

    if os.path.exists(contexts_file):
        contexts = torch.load(contexts_file)
    else:
        contexts = generate_and_save_contexts(n_contexts, obj_func.context_dim, contexts_file)

    timestamp = "dtlz-2_5_2"

    for run in range(n_runs):
        print(f"Starting run {run + 1}/{n_runs}")

        run_data = {}
        for context_idx, context in enumerate(contexts):
            print(f" Optimizing for context {context_idx + 1}/{len(contexts)}")

            # Create a wrapper function for this specific context
            def context_specific_obj(x):
                x_with_context = torch.cat([x, context.unsqueeze(0).repeat(x.shape[0], 1)], dim=1)
                return obj_func.evaluate(x_with_context)

            mobo = MultiObjectiveBayesianOptimization(
                objective_func=PseudoObjectiveFunction(
                    func=context_specific_obj,
                    dim=obj_func.input_dim,
                    context_dim=obj_func.context_dim,
                    output_dim=obj_func.output_dim,
                    nadir_point=obj_func.nadir_point
                )
            )

            # Generate initial points for this context
            n_initial_points = 20
            X_init = torch.rand(n_initial_points, obj_func.input_dim)
            Y_init = context_specific_obj(X_init)

            # Run optimization
            X_opt, Y_opt = mobo.optimize(X_init, Y_init, n_iter=n_iter)

            # Store results for this context
            context_key = tuple(context.numpy())
            run_data[context_key] = {
                'pareto_set_history': mobo.pareto_set_history,
                'pareto_front_history': mobo.pareto_front_history,
                'hv_history': mobo.hv_history
            }

        # Save individual run data
        save_path = f'result/MOBO_optimization_history_{timestamp}_run_{run}.pth'
        torch.save(run_data, save_path)
        print(f"Run {run} data saved to {save_path}")

        # Plot results for this run
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f'MOBO Hypervolume History for Each Context (Run {run + 1})', fontsize=16)

        for context_idx, context in enumerate(contexts):
            row = context_idx // 4
            col = context_idx % 4
            ax = axes[row, col]

            context_key = tuple(context.numpy())
            hv_history = run_data[context_key]['hv_history']
            ax.plot(range(len(hv_history)), hv_history, label=f'Run {run + 1}')

            ax.set_title(f'Context {context_idx + 1}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Hypervolume')
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'result/MOBO_hypervolume_history_grid_{timestamp}_run_{run}.png')
        plt.close()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    # np.random.seed(42)

    # Run tests
    optimization_loop_test()
    run_mobo_test()
    # random_test_obj()
    # context_influence_test_grid()
    # random_test_boundary()
    # temp()
    # test_bo()
    # test_multiobjective_functions()
    # test_mobo()
    # print("\nRunning visualization tests...")
    # for func_name in ['dtlz2', 'dtlz3']:
    #     print(f"\nTesting {func_name.upper()}")
    #     for n_obj in [2]:
    #         print(f"Testing with {n_obj} objectives")
    #         visualize_contextual_effects(func_name, n_objectives=n_obj)



