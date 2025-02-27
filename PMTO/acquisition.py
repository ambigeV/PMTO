import torch
import numpy as np
from typing import List, Callable
from scipy.optimize import minimize
import gpytorch


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
        n_restarts: int = 10,
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
        n_restarts: int = 10,
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

