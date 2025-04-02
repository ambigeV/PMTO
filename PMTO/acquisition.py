import torch
import numpy as np
from typing import List, Callable
from scipy.optimize import minimize
import gpytorch


def optimize_with_ehvi(models, pareto_front, nadir_point, input_dim, x_mean=None, x_std=None, minimize=True):
    """
    Compute the next point to sample using EHVI via BoTorch.

    Args:
        models: List of trained GP models
        pareto_front: Current Pareto front points [n_pareto, n_objectives]
        nadir_point: Nadir point (worst values for each objective)
        input_dim: Dimension of input space
        x_mean: Mean for input normalization
        x_std: Standard deviation for input normalization
        minimize: Whether this is a minimization problem (True) or maximization (False)

    Returns:
        next_x: Next point to sample
    """
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
    from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
    from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
    from botorch.optim import optimize_acqf

    # Reference point (nadir point) handling
    ref_point = nadir_point.clone()

    # Add a small margin to the reference point to ensure it's worse than all observed points
    margin = 0.01 * torch.abs(ref_point)
    if minimize:
        ref_point = ref_point + margin  # For minimization, worse means larger
    else:
        ref_point = ref_point - margin  # For maximization, worse means smaller

    # Convert pareto front and reference point based on minimization/maximization
    if minimize:
        # For minimization, we negate both because BoTorch assumes maximization
        ref_point_botorch = -ref_point
        if pareto_front is not None and len(pareto_front) > 0:
            neg_pareto_front = -pareto_front.clone()
        else:
            neg_pareto_front = None
    else:
        # For maximization, use as is
        ref_point_botorch = ref_point
        if pareto_front is not None and len(pareto_front) > 0:
            neg_pareto_front = pareto_front.clone()
        else:
            neg_pareto_front = None

    def convert_gpytorch_to_botorch(model_obj, X_train, y_train, minimize=True):
        """
        Convert a trained GPyTorch model to a BoTorch SingleTaskGP model with copied parameters
        """
        gpytorch_model = model_obj.model
        likelihood = model_obj.likelihood

        # For minimization, negate the targets for BoTorch (which assumes maximization)
        if minimize:
            y_botorch = -y_train.clone()
        else:
            y_botorch = y_train.clone()

        # Make sure y_botorch has the right shape (n, 1)
        if y_botorch.dim() == 1:
            y_botorch = y_botorch.unsqueeze(-1)

        # Ensure X_train has the right shape and is contiguous
        X_train = X_train.contiguous()

        # Create a new BoTorch model with the same training data
        botorch_model = SingleTaskGP(X_train, y_botorch)

        # Make sure the model is in eval mode when copying parameters
        gpytorch_model.eval()
        likelihood.eval()
        botorch_model.eval()

        # Copy over the parameters from the trained GPyTorch model
        try:
            # For kernel parameters
            botorch_model.covar_module.base_kernel.lengthscale = gpytorch_model.covar_module.base_kernel.lengthscale.clone()
            botorch_model.covar_module.outputscale = gpytorch_model.covar_module.outputscale.clone()

            # For mean function parameters
            if hasattr(gpytorch_model.mean_module, 'constant'):
                botorch_model.mean_module.constant = gpytorch_model.mean_module.constant.clone()

            # For noise
            botorch_model.likelihood.noise = likelihood.noise.clone()
        except Exception as e:
            print(f"Warning: Could not copy all model parameters. Error: {e}")
            print("Using default BoTorch model parameters.")

        # # Verify that the posterior method works
        # try:
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         posterior = botorch_model.posterior(X_train[:1])  # Test with first input point
        #         print(f"Posterior mean shape: {posterior.mean.shape}")
        # except Exception as e:
        #     print(f"Error obtaining posterior: {e}")
        #     raise

        return botorch_model

    # Create BoTorch models with copied parameters
    botorch_models = []
    for model in models:
        # Get training data
        X_train = model.model.train_inputs[0]
        y_train = model.model.train_targets

        # Create BoTorch model with copied parameters
        botorch_model = convert_gpytorch_to_botorch(model, X_train, y_train, minimize)
        botorch_models.append(botorch_model)

    # Create model list
    model_list = ModelListGP(*botorch_models)

    # Create training outputs for all models (for partitioning)
    train_y_list = []
    for i, model in enumerate(models):
        y_train = model.model.train_targets

        # For minimization, negate the outputs
        if minimize:
            train_y_list.append(-y_train)
        else:
            train_y_list.append(y_train)

    # Stack outputs to create a multi-objective output tensor
    train_Y = torch.stack(train_y_list, dim=-1)

    # Create FastNondominatedPartitioning using all training data
    n_objectives = ref_point.shape[0]
    partitioning = NondominatedPartitioning(
        ref_point=ref_point_botorch,
        Y=train_Y
    )

    # Create EHVI acquisition function with FastNondominatedPartitioning
    ehvi = ExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=ref_point_botorch.tolist(),
        partitioning=partitioning
    )

    # Bounds for optimization
    bounds = torch.stack([torch.zeros(input_dim), torch.ones(input_dim)])

    # Optimize acquisition function
    candidate, acq_value = optimize_acqf(
        acq_function=ehvi,
        bounds=bounds,
        q=1,
        num_restarts=3,
        raw_samples=128,
    )

    # Denormalize if needed
    if x_mean is not None and x_std is not None:
        next_x = candidate.squeeze(0) * x_std + x_mean
    else:
        next_x = candidate.squeeze(0)

    print(f"EHVI value: {acq_value.item():.6e}")

    return next_x


def expected_improvement(model, likelihood, X, best_value):
    """
    Calculate expected improvement acquisition function values

    Args:
        model: GP model
        likelihood: GP likelihood
        X: Input points [batch_size, input_dim]
        best_value: Current best observed value (scalar)

    Returns:
        Expected improvement values [batch_size]
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get predictive distribution
        output = model(X)
        mean = output.mean
        std = torch.sqrt(output.variance)

        # For minimization (lower is better)
        improvement = best_value - mean

        # Calculate normalized improvement
        z = improvement / (std + 1e-8)  # Add small constant for numerical stability

        # Compute EI using the normal CDF and PDF
        normal = torch.distributions.Normal(0, 1)
        cdf = normal.cdf(z)
        pdf = torch.exp(normal.log_prob(z))

        ei = improvement * cdf + std * pdf

        # Set EI to 0 where std is very small to avoid numerical issues
        ei = torch.where(std > 1e-6, ei, torch.zeros_like(ei))

        return ei


def optimize_scalarized_acquisition_ei(
        model,
        best_value,
        input_dim,
        x_mean=None,
        x_std=None,
        n_restarts=5,
        bounds=None
) -> torch.Tensor:
    """
    Optimize the expected improvement acquisition function

    Args:
        model: BO model containing GP model and likelihood
        best_value: Current best observed scalarized value
        input_dim: Input dimension
        x_mean: Mean for input normalization
        x_std: Standard deviation for input normalization
        n_restarts: Number of random restarts for optimization
        bounds: Bounds for optimization variables, default is [0,1] for each dimension

    Returns:
        Optimal input point
    """
    bounds = bounds or [(0, 1)] * input_dim

    def objective(x):
        """Objective function to minimize (negative EI)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure proper dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Normalize input if needed
        if x_mean is not None and x_std is not None:
            x_norm = (x - x_mean) / x_std
        else:
            x_norm = x

        return -expected_improvement(model.model, model.likelihood, x_norm, best_value)

    best_x = None
    best_value_found = float('inf')

    # Multiple random restarts
    for _ in range(n_restarts):
        x0 = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=input_dim
        )

        res = minimize(
            objective,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )

        if res.fun < best_value_found:
            best_value_found = res.fun
            best_x = res.x

    if best_x is None:
        # Fallback if optimization fails
        best_x = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=input_dim
        )

    print("Best EI value: {}".format(-best_value_found))

    return torch.tensor(best_x, dtype=torch.float32)


def ucb_acquisition_group(model, likelihood, x, beta=2.0):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))
        mean = pred.mean
        std = pred.variance.sqrt()
    return (mean - beta * std).detach().numpy()  # Negative for maximization


def ucb_acquisition(model, likelihood, x, beta=2.0):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if not isinstance(x, torch.Tensor):
            x = torch.tensor([x], dtype=torch.float32).reshape(1, -1)
        else:
            x = x.reshape(1, -1)
        pred = likelihood(model(x))
        mean = pred.mean
        std = pred.variance.sqrt()
    return (mean - beta * std).detach().numpy()  # Negative for maximization


# Optimizer to Maximize UCB
def optimize_acquisition(model,
                         likelihood,
                         bounds=None,
                         n_restarts=10,
                         beta=2.0,
                         dim=1000,
                         x_mean=None,
                         x_std=None):
    best_x = None
    best_value = float('inf')

    # Acquisition function (UCB with normalization)
    def min_obj(x):
        x = torch.tensor(x, dtype=torch.float32)

        # Normalize candidate points if x_mean and x_std are provided
        if x_mean is not None and x_std is not None:
            x = (x - x_mean) / x_std

        return ucb_acquisition(model, likelihood, x, beta)

    # Multiple Restarts to Avoid Local Optima
    for _ in range(n_restarts):
        x0 = np.random.uniform(0, 1, size=(dim))
        res = minimize(min_obj, x0, bounds=[(0, 1)] * dim, method='L-BFGS-B')

        if res.fun < best_value:
            best_value = res.fun
            best_x = res.x

    best_x = torch.tensor(best_x, dtype=torch.float32)

    return best_x


def optimize_scalarized_acquisition(
        models: List,
        scalarization_func: Callable,
        weights: torch.Tensor,
        input_dim: int,
        beta: float = 2.0,
        n_restarts: int = 10,
        bounds: List[tuple] = None,
        x_mean: torch.Tensor = None,
        x_std: torch.Tensor = None
) -> torch.Tensor:

    bounds = bounds or [(0, 1)] * input_dim

    def combined_acquisition(x: torch.Tensor) -> float:
        """Combine multiple acquisition values using scalarization"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Normalize input if needed
        if x_mean is not None and x_std is not None:
            x_norm = (x - x_mean) / x_std
        else:
            x_norm = x

        # Get acquisition values for each objective
        acq_values = []
        for model in models:
            acq_value = ucb_acquisition(model.model, model.likelihood, x_norm, beta)
            acq_values.append(torch.tensor(acq_value))

        # Stack and scalarize
        stacked_acq = torch.stack(acq_values, dim=-1)
        scalarized = scalarization_func(stacked_acq, weights)

        return scalarized.item()  # Negative for minimization

    best_x = None
    best_value = float('inf')

    # Multiple random restarts
    for _ in range(n_restarts):
        x0 = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=input_dim
        )
        res = minimize(
            combined_acquisition,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )

        if res.fun < best_value:
            best_value = res.fun
            best_x = res.x

    print("best_value is {}".format(best_value))

    return torch.tensor(best_x, dtype=torch.float32)


def optimize_acquisition_for_context(
        model,
        likelihood,
        context: torch.Tensor,
        x_dim: int,
        beta: float = 2.0,
        n_restarts: int = 5,
        bounds: tuple = (0, 1),
        x_mean: torch.Tensor = None,
        x_std: torch.Tensor = None
) -> torch.Tensor:

    def objective(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Concatenate x with context
        x_c = torch.cat([x, context])

        # Normalize if needed
        if x_mean is not None and x_std is not None:
            x_c = (x_c - x_mean) / x_std

        return ucb_acquisition(model, likelihood, x_c, beta)

    best_x = None
    best_value = float('inf')

    # Multiple random restarts
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[0], bounds[1], size=x_dim)
        res = minimize(
            objective,
            x0,
            bounds=[bounds] * x_dim,
            method='L-BFGS-B'
        )

        if res.fun < best_value:
            best_value = res.fun
            best_x = res.x

    return torch.tensor(best_x, dtype=torch.float32)


def optimize_scalarized_acquisition_for_context(
        models: List,
        context: torch.Tensor,
        x_dim: int,
        scalarization_func: Callable,
        weights: torch.Tensor,
        beta: float = 2.0,
        n_restarts: int = 20,
        bounds: tuple = (0, 1),
        x_mean: torch.Tensor = None,
        x_std: torch.Tensor = None
) -> torch.Tensor:

    def objective(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Concatenate x with context
        x_c = torch.cat([x, context])

        # Normalize if needed
        if x_mean is not None and x_std is not None:
            x_c = (x_c - x_mean) / x_std

        # Get acquisition values for each objective
        acq_values = []
        for model in models:
            acq_value = ucb_acquisition(model["model"], model["likelihood"], x_c, beta)
            acq_values.append(torch.tensor(acq_value))

        # Stack and scalarize
        stacked_acq = torch.stack(acq_values, dim=-1)
        scalarized = scalarization_func(stacked_acq, weights)

        return scalarized.item()  # Negative for minimization

    best_x = None
    best_value = float('inf')

    # Multiple random restarts
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[0], bounds[1], size=x_dim)
        res = minimize(
            objective,
            x0,
            bounds=[bounds] * x_dim,
            method='L-BFGS-B'
        )

        if res.fun < best_value:
            best_value = res.fun
            best_x = res.x

    print("context best_value is {}".format(best_value))

    return torch.tensor(best_x, dtype=torch.float32)

