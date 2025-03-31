import torch
import numpy as np
import gpytorch
from math import log
import math
from .models import SVGPModel, ExactGPModel, ArdGPModel, CustomGPModel
from .acquisition import optimize_acquisition, optimize_scalarized_acquisition, \
    optimize_acquisition_for_context, optimize_scalarized_acquisition_for_context, \
    ucb_acquisition_group, expected_improvement, optimize_scalarized_acquisition_ei, optimize_with_ehvi
from typing import Callable, Optional, Tuple, List, Dict
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# TensorBoard Library
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# Optimizer
from .LBFGS import FullBatchLBFGS
# CVAE Model
from .gen_models import ParetoVAETrainer
import os


IF_PLOT = False
IF_GLOBAL = True
SCALAR = "HV"
NOISE = False
# "AT"

class GPMonitor:
    def __init__(self, log_dir=None, dir_name=None, if_plot=IF_PLOT):
        # Create unique run name with timestamp
        self.if_plot = if_plot
        if self.if_plot:
            timestamp = dir_name
            if log_dir is None:
                # log_dir = f'runs/gp_monitoring_constraint_prior_{timestamp}'
                self.log_dir = f'dtlz1_runs/gp_monitoring_hv_{timestamp}'
            self.writer = SummaryWriter(self.log_dir)
        else:
            pass

    def log_kernel_params(self, model, objective, iteration):
        if not self.if_plot:
            return None
        else:
            pass

        if hasattr(model.covar_module.base_kernel, 'get_lengthscales'):
            lengthscales = model.covar_module.base_kernel.get_lengthscales()
        else:
            """Log kernel parameters for both decision and context spaces with ARD support"""
            # Get lengthscales - for ARD this will be a tensor with multiple values
            lengthscales = model.covar_module.base_kernel.lengthscale.detach()

        # If it's a single lengthscale (non-ARD)
        if lengthscales.shape[1] == 1:
            self.writer.add_scalar(
                f'Kernel/decision_lengthscale_{objective}',
                lengthscales.item(),
                iteration
            )
        # If it's ARD kernel with multiple lengthscales
        else:
            # Log each dimension's lengthscale separately
            for dim, ls in enumerate(lengthscales.squeeze()):
                self.writer.add_scalar(
                    f'Kernel/decision_lengthscale_{objective}_dim{dim}',
                    ls.item(),
                    iteration
                )

    def log_predictions(self, model, likelihood, X_sample, Y_true, iteration, objective_idx):
        """Log prediction statistics for a sample of points"""
        if not self.if_plot:
            return None
        else:
            pass

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get predictions
            pred = likelihood(model(X_sample))
            mean = pred.mean
            std = pred.variance.sqrt()

            # Compute prediction errors
            pred_errors = (mean - Y_true).abs()  # Absolute error
            rel_errors = pred_errors / (Y_true.abs() + 1e-8)  # Relative error

            # Log statistics
            # Log error statistics
            self.writer.add_scalar(f'Prediction_Errors/mean_abs_error_obj{objective_idx}',
                                   pred_errors.mean().item(), iteration)
            self.writer.add_scalar(f'Prediction_Errors/max_abs_error_obj{objective_idx}',
                                   pred_errors.max().item(), iteration)
            self.writer.add_scalar(f'Prediction_Errors/mean_rel_error_obj{objective_idx}',
                                   rel_errors.mean().item(), iteration)
            self.writer.add_scalar(f'Predictions/std_obj{objective_idx}',
                                   std.mean().item(), iteration)
            self.writer.add_scalar(f'Predictions/max_std_obj{objective_idx}',
                                   std.max().item(), iteration)

            # Log uncertainty metrics
            uncertainty_calibration = pred_errors / std  # Should be close to 1 if well-calibrated
            self.writer.add_scalar(f'Prediction_Errors/uncertainty_calibration_obj{objective_idx}',
                                   uncertainty_calibration.mean().item(), iteration)

            self.writer.add_histogram(f'Distributions/abs_error_obj{objective_idx}',
                                      pred_errors.numpy(), iteration)
            self.writer.add_histogram(f'Distributions/rel_error_obj{objective_idx}',
                                      rel_errors.numpy(), iteration)
            self.writer.add_histogram(f'sDistributions/uncertainty_calibration_obj{objective_idx}',
                                      uncertainty_calibration.numpy(), iteration)
            self.writer.add_histogram(f'Distributions/std_obj{objective_idx}',
                                      std.numpy(), iteration)

    def log_acquisition_values(self, acq_values, iteration):
        """Log statistics of acquisition function values"""
        if not self.if_plot:
            return None

        self.writer.add_scalar('Acquisition/mean', np.mean(acq_values), iteration)
        self.writer.add_scalar('Acquisition/max', np.max(acq_values), iteration)
        self.writer.add_histogram('Distributions/acquisition', acq_values, iteration)

    def log_optimization_metrics(self, metrics, iteration):
        """Log optimization-related metrics including Pareto front"""
        if not self.if_plot:
            return None

        if 'hypervolume' in metrics:
            self.writer.add_scalar('Optimization/hypervolume',
                                   metrics['hypervolume'], iteration)

        if 'pareto_points' in metrics:
            pareto_points = metrics['pareto_points']
            # Log number of Pareto points
            self.writer.add_scalar('Optimization/pareto_points',
                                   len(pareto_points), iteration)

            # Create scatter plot of Pareto front
            if isinstance(pareto_points, (np.ndarray, torch.Tensor)):
                points = pareto_points
            else:
                points = np.array(pareto_points)

            # Create figure for Pareto front
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(points[:, 0], points[:, 1], c='blue', label='Pareto points')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title(f'Pareto Front at Iteration {iteration}')
            plt.grid(True)
            plt.legend()

            # Add figure to tensorboard
            self.writer.add_figure('Pareto_Front/scatter', fig, iteration)
            plt.close(fig)

            # Optional: Add animation frames
            if hasattr(self, 'previous_fronts'):
                fig = plt.figure(figsize=(8, 8))
                for prev_iter, prev_points in self.previous_fronts[-5:]:  # Show last 5 fronts
                    alpha = 0.2 + 0.8 * (prev_iter / iteration)  # Fade older fronts
                    plt.scatter(prev_points[:, 0], prev_points[:, 1],
                                alpha=alpha, label=f'Iter {prev_iter}')
                plt.scatter(points[:, 0], points[:, 1],
                            c='red', label=f'Current (Iter {iteration})')
                plt.xlabel('Objective 1')
                plt.ylabel('Objective 2')
                plt.title('Pareto Front Evolution')
                plt.grid(True)
                plt.legend()
                self.writer.add_figure('Pareto_Front/evolution', fig, iteration)
                plt.close(fig)

            # Store current front for animation
            if not hasattr(self, 'previous_fronts'):
                self.previous_fronts = []
            self.previous_fronts.append((iteration, points))
            # Keep only last 10 fronts to manage memory
            if len(self.previous_fronts) > 10:
                self.previous_fronts.pop(0)

    def close(self):
        if not self.if_plot:
            return None
        else:
            pass

        """Close the writer"""
        self.writer.close()


class HypervolumeScalarization:
    """
    Hypervolume scalarization for minimization using a nadir point,
    where the per-coordinate ratio is exponentiated first and then the
    minimum over objectives is taken.

    Given an objective vector y (to be minimized), a nadir point (an upper bound),
    and a weight vector, we define the scalarization as:

        s_λ(y) = min_i [ max(0, (nadir_i - y_i) / weights_i) ^ exponent ]

    We then return its negative so that a lower scalarized value corresponds to a better candidate.

    Args:
        nadir_point (torch.Tensor): The nadir point vector (upper bounds) for each objective.
        exponent (float): The exponent to apply to each coordinate ratio before minimization.
    """

    def __init__(self, nadir_point: torch.Tensor, exponent: float = 2.0):
        self.nadir_point = nadir_point
        self.exponent = exponent

    def __call__(self, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Compute the improvement relative to the nadir point.
        # Assuming y <= nadir_point element-wise, diff = nadir_point - y.
        diff = self.nadir_point - y
        # Compute the per-objective ratio.
        ratio = diff / (weights + 1e-8)
        # Ensure non-negativity.
        ratio = torch.clamp(ratio, min=0.0)
        # First, raise each element to the specified exponent.
        exp_ratio = ratio ** self.exponent
        # Then, take the minimum over the objective dimensions.
        min_val, _ = torch.min(exp_ratio, dim=-1)
        # Return the negative so that minimizing the scalarized value corresponds to better (lower) objective values.
        return -min_val


class AugmentedTchebycheff:
    """Augmented Tchebycheff scalarization"""

    def __init__(self, reference_point: torch.Tensor, rho: float = 0.05):
        self.reference_point = reference_point
        self.rho = rho

    def __call__(self, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_diff = weights * (y - self.reference_point)
        weighted_y = weights * y
        max_term = torch.max(weighted_diff, dim=-1)[0]
        sum_term = self.rho * torch.sum(weighted_y, dim=-1)
        return max_term + sum_term


class PseudoObjectiveFunction:
    """Wrapper for objective functions"""
    def __init__(self,
                 func: Callable,
                 dim: int = 0,
                 context_dim: int = 0,
                 output_dim: int = 0,
                 nadir_point: torch.Tensor = None):
        self.func = func
        self.input_dim = dim
        self.dim = dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.nadir_point = nadir_point

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class BayesianOptimization:
    def __init__(self,
                 objective_func,
                 inducing_points=None,
                 train_steps=500,
                 model_type='SVGP',
                 optimizer_type='adam'):
        self.objective_func = objective_func
        self.inducing_points = inducing_points
        self.train_steps = train_steps
        self.model_type = model_type
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.optimizer_type = optimizer_type.lower()

        self.dim = objective_func.dim
        self.model = None

        # Placeholders for normalization parameters
        self.x_mean, self.x_std = None, None
        self.y_mean, self.y_std = None, None

        # ------------------------------
        # Z-score Normalization Function
        # ------------------------------
    def normalize_data(self, X, Y):
        """
        Perform Z-score normalization for train_X and train_Y.

        Parameters:
        - X: Training input data (train_X).
        - Y: Training output data (train_Y).

        Returns:
        - Normalized X and Y tensors.
        """
        # Compute mean and std for X and Y
        x_mean, x_std = X.mean(dim=0), X.std(dim=0)
        y_mean, y_std = Y.mean(dim=0), Y.std(dim=0)

        # Z-score normalization
        X_normalized = (X - x_mean) / x_std
        Y_normalized = (Y - y_mean) / y_std

        # Store the current normalization parameters
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

        return X_normalized, Y_normalized

    def normalize_inference(self, X):
        """
        Normalize new input points during inference using stored scaling factors.

        Parameters:
        - X: New input points.

        Returns:
        - Normalized X tensor.
        """
        if self.x_mean is not None and self.x_std is not None:
            return (X - self.x_mean) / self.x_std
        return X

    def denormalize_input(self, X):
        """
        Denormalize input points to original space.

        Parameters:
        - X: Normalized input points.

        Returns:
        - Denormalized X tensor.
        """
        if self.x_mean is not None and self.x_std is not None:
            return X * self.x_std + self.x_mean
        return X

    def denormalize_output(self, Y):
        """
        Denormalize predictions to the original scale.

        Parameters:
        - Y: Normalized predictions.

        Returns:
        - Denormalized Y tensor.
        """
        if self.y_mean is not None and self.y_std is not None:
            return Y * self.y_std + self.y_mean
        return Y

    def normalize_output(self, Y):
        """
        Normalize predictions to the original scale.

        Parameters:
        - Y: Normalized predictions.

        Returns:
        - Denormalized Y tensor.
        """
        if self.y_mean is not None and self.y_std is not None:
            return (Y - self.y_mean) / self.y_std
        return Y

    def build_model(self, X_train, y_train):
        if self.model_type == 'SVGP':
            model = SVGPModel(self.inducing_points, input_dim=self.dim)
        elif self.model_type == 'ArdGP':
            model = ArdGPModel(X_train, y_train, self.likelihood)
        else:
            model = ExactGPModel(X_train, y_train, self.likelihood)
        return model

    def optimize(self, X_train, y_train, n_iter=50, beta=2.0):
        best_y = []

        for i in range(n_iter):
            X_train_norm, y_train_norm = self.normalize_data(X_train.clone(), y_train.clone())
            # print(X_train_norm)
            # print(y_train_norm)

            model = self.build_model(X_train_norm, y_train_norm)
            model.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=0.01) if self.optimizer_type == 'adam' else torch.optim.LBFGS(
                model.parameters(), lr=0.1, max_iter=20)

            # mll = gpytorch.mlls.VariationalELBO(self.likelihood, model, num_data=y_train.size(0))
            if self.model_type == 'SVGP':
                mll = gpytorch.mlls.VariationalELBO(self.likelihood, model, num_data=y_train.size(0))
            else:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

            if self.optimizer_type == 'lbfgs':
                prev_loss = float('inf')
                for _ in range(20):
                    def closure():
                        optimizer.zero_grad()
                        output = model(X_train_norm)
                        loss = -mll(output, y_train_norm)
                        loss.backward()
                        return loss

                    curr_loss = optimizer.step(closure)
                    if abs(prev_loss - curr_loss.item()) < 1e-5:
                        break
                    prev_loss = curr_loss.item()

            else:
                prev_loss = float('inf')
                for _ in range(self.train_steps):
                    optimizer.zero_grad()
                    output = model(X_train_norm)
                    loss = -mll(output, y_train_norm)
                    loss.backward()
                    optimizer.step()
                    prev_loss = loss.item()

            next_x = optimize_acquisition(model, likelihood=self.likelihood, beta=beta,
                                          dim=self.dim, x_mean=self.x_mean, x_std=self.x_std)
            next_y = self.objective_func.evaluate(next_x)

            X_train = torch.cat([X_train, next_x.unsqueeze(0)])
            y_train = torch.cat([y_train, next_y.unsqueeze(0)])
            best_y.append(y_train.min().item())

            if i % 5 == 0:
                print(f'Iteration {i}/{n_iter}, Best y: {y_train.min().item():.3f}')

            self.model = model

        return X_train, y_train, best_y


class ContextualBayesianOptimization:
    def __init__(
            self,
            objective_func,
            inducing_points: Optional[torch.Tensor] = None,
            train_steps: int = 500,
            model_type: str = 'SVGP',
            optimizer_type: str = 'adam'
    ):
        self.objective_func = objective_func
        self.x_dim = objective_func.dim
        self.context_dim = objective_func.context_dim
        self.dim = self.x_dim + self.context_dim

        self.inducing_points = inducing_points
        self.train_steps = train_steps
        self.model_type = model_type
        self.optimizer_type = optimizer_type.lower()

        self.likelihood_new = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3)
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model = None

        # Placeholders for normalization parameters
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # Dictionary to track best values per context
        self.context_best_values = {}

    def normalize_data(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize input and output data."""
        # Compute mean and std for X and Y
        x_mean, x_std = X.mean(dim=0), X.std(dim=0)
        y_mean, y_std = Y.mean(dim=0), Y.std(dim=0)

        # Z-score normalization
        X_normalized = (X - x_mean) / x_std
        Y_normalized = (Y - y_mean) / y_std

        # Store normalization parameters
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

        return X_normalized, Y_normalized

    def normalize_output(self, Y):
        """
        Normalize predictions to the original scale.

        Parameters:
        - Y: Normalized predictions.

        Returns:
        - Denormalized Y tensor.
        """
        if self.y_mean is not None and self.y_std is not None:
            return (Y - self.y_mean) / self.y_std
        return Y

    def build_model(self, X_train: torch.Tensor, y_train: torch.Tensor, if_noise=False):
        """Build GP model based on specified type."""
        if self.model_type == 'SVGP':
            model = SVGPModel(self.inducing_points, input_dim=self.dim)
        elif self.model_type == 'ArdGP':
            model = ArdGPModel(X_train, y_train, self.likelihood)
        elif self.model_type == 'CustomGP':
            if if_noise:
                model = CustomGPModel(X_train, y_train, self.likelihood_new, self.x_dim - self.context_dim, self.context_dim)
            else:
                model = CustomGPModel(X_train, y_train, self.likelihood, self.x_dim - self.context_dim, self.context_dim)
        else:
            if if_noise:
                model = ExactGPModel(X_train, y_train, self.likelihood_new)
            else:
                model = ExactGPModel(X_train, y_train, self.likelihood)

        self.model = model
        return model

    def update_context_best_values(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            contexts: torch.Tensor
    ):
        """Update best values for each context."""
        for context in contexts:
            context_key = tuple(context.numpy())

            # Find all points with this context
            context_mask = torch.all(X[:, self.x_dim:] == context, dim=1)
            if torch.any(context_mask):
                context_values = Y[context_mask]
                current_best = context_values.min().item()

                # Update if better than previous best or no previous best exists
                if context_key not in self.context_best_values:
                    self.context_best_values[context_key] = []
                self.context_best_values[context_key].append(current_best)

    def optimize(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            contexts: torch.Tensor,
            n_iter: int = 50,
            beta: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[tuple, List[float]]]:

        # Initialize best values tracking
        self.update_context_best_values(X_train, y_train, contexts)

        for iteration in range(n_iter):
            # Normalize data
            # TODO: The normalization is conducted uniformly for all contexts
            X_train_norm, y_train_norm = self.normalize_data(X_train.clone(), y_train.clone())

            # Build and train model
            model = self.build_model(X_train_norm, y_train_norm)
            model.train()
            self.likelihood.train()

            # Set up optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=0.01) if self.optimizer_type == 'adam' else torch.optim.LBFGS(
                model.parameters(), lr=0.1, max_iter=20)

            if self.model_type == 'SVGP':
                mll = gpytorch.mlls.VariationalELBO(self.likelihood, model, num_data=y_train.size(0))
            else:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

            # Train the model
            if self.optimizer_type == 'lbfgs':
                prev_loss = float('inf')
                for _ in range(20):
                    def closure():
                        optimizer.zero_grad()
                        output = model(X_train_norm)
                        loss = -mll(output, y_train_norm)
                        loss.backward()
                        return loss

                    curr_loss = optimizer.step(closure)
                    if abs(prev_loss - curr_loss.item()) < 1e-5:
                        break
                    prev_loss = curr_loss.item()

            else:
                prev_loss = float('inf')
                for _ in range(self.train_steps):
                    optimizer.zero_grad()
                    output = model(X_train_norm)
                    loss = -mll(output, y_train_norm)
                    loss.backward()
                    optimizer.step()
                    prev_loss = loss.item()

            # Optimize acquisition for each context
            next_points = []
            next_values = []

            for context in contexts:
                # Find best x for this context
                next_x = optimize_acquisition_for_context(
                    model=model,
                    likelihood=self.likelihood,
                    context=context,
                    x_dim=self.x_dim,
                    beta=beta,
                    x_mean=self.x_mean,
                    x_std=self.x_std
                )

                # Evaluate objective
                x_c = torch.cat([next_x, context])
                next_y = self.objective_func.evaluate(x_c)

                next_points.append(x_c)
                next_values.append(next_y)

            # Stack new points and values
            next_points = torch.stack(next_points)
            next_values = torch.stack(next_values)

            # Update training data
            X_train = torch.cat([X_train, next_points])
            y_train = torch.cat([y_train, next_values])

            # Update best values per context
            self.update_context_best_values(next_points, next_values, contexts)

            if iteration % 3 == 0:
                print(f'Iteration {iteration}/{n_iter}')
                for context in contexts:
                    context_key = tuple(context.numpy())
                    print(f'Context {context_key}: Best value = {self.context_best_values[context_key][-1]:.3f}')

            self.model = model

        return X_train, y_train, self.context_best_values


class MultiObjectiveBayesianOptimization:
    """Multi-Objective Bayesian Optimization with scalarization"""

    def __init__(
            self,
            objective_func,
            reference_point=None,
            inducing_points=None,
            train_steps=200,
            model_type='ExactGP',
            optimizer_type='adam',
            rho=0.001,
            mobo_id=None
    ):
        self.objective_func = objective_func
        self.input_dim = objective_func.input_dim
        self.output_dim = objective_func.output_dim
        self.mobo_id = mobo_id
        self.model_type = model_type
        self.base_beta = None
        self.beta = None
        self.rho = rho

        # Initialize reference point if not provided
        if reference_point is None:
            self.reference_point = torch.zeros(self.output_dim)
        else:
            self.reference_point = reference_point

        self.nadir_point = self.objective_func.nadir_point
        self.hv = Hypervolume(ref_point=self.nadir_point.numpy())
        self.current_hv = -1
        self.current_reference_points = None
        self.current_nadir_points = None

        # Initialize scalarization
        if SCALAR == "AT":
            self.scalarization = AugmentedTchebycheff(
                reference_point=self.reference_point,
                rho=rho
            )
        else:
            self.scalarization = HypervolumeScalarization(
                nadir_point=self.nadir_point,
                exponent=self.output_dim
            )

        # Create individual BO instances for each objective dimension
        self.bo_models = []
        self.monitor = None
        for _ in range(self.output_dim):
            # Create a wrapper single-objective function for each output dimension
            single_obj = PseudoObjectiveFunction(
                func=lambda x, dim=_: self.objective_func.evaluate(x)[:, dim],
                dim=self.input_dim,
            )

            bo = BayesianOptimization(
                objective_func=single_obj,
                inducing_points=inducing_points,
                train_steps=train_steps,
                model_type=model_type,
                optimizer_type=optimizer_type
            )
            self.bo_models.append(bo)

        # Store Pareto front approximation
        self.pareto_front = None
        self.pareto_set = None
        self.hv_history = []
        self.pareto_front_history = []
        self.pareto_set_history = []
        self.model_list = []

    def _update_and_normalize_reference_points(self, Y_train):
        self.current_reference_points = torch.min(Y_train, dim=0)[0]
        self.current_reference_points = self.current_reference_points - 0.1
        # minimal points with a threshold
        self.current_nadir_points = torch.max(Y_train, dim=0)[0]
        self.current_nadir_points = self.current_nadir_points + 0.1 * torch.abs(self.current_nadir_points)
        # maximum points with a threshold

        # Normalize the nadir points and reference points
        for ind in range(self.output_dim):
            self.current_reference_points[ind] = self.bo_models[ind].normalize_output(
                self.current_reference_points[ind])
            self.current_nadir_points[ind] = self.bo_models[ind].normalize_output(
                self.current_nadir_points[ind])

    def _update_pareto_front(self, X: torch.Tensor, Y: torch.Tensor, minimize: bool = True):
        """Update Pareto front using pymoo's non-dominated sorting."""
        # Convert to numpy for pymoo
        Y_np = Y.numpy()
        if not minimize:
            Y_np = -Y_np

        # Get non-dominated front
        front = NonDominatedSorting().do(Y_np)[0]  # Get first front only

        # Update Pareto set and front
        self.pareto_set = X[front]
        self.pareto_set_history.append(self.pareto_set)
        self.pareto_front = Y[front]
        self.pareto_front_history.append(self.pareto_front)

        # Calculate hypervolume
        self.current_hv = self.hv.do(self.pareto_front.numpy()) if len(front) > 0 else 0.0
        self.hv_history.append(self.current_hv)

    def _update_beta(self, iteration):
        self.beta = math.sqrt(self.base_beta * log(1 + 2 * iteration))

    @staticmethod
    def _sample_points(n_points, n_decision_vars, n_context_vars):
        """Generate a sample of points for monitoring predictions"""
        total_dims = n_decision_vars + n_context_vars
        return torch.rand(n_points, total_dims)

    @staticmethod
    def _generate_weight_vector(dim: int) -> torch.Tensor:
        """Generate a random weight vector from a Dirichlet distribution."""
        alpha = torch.ones(dim)  # Symmetric Dirichlet distribution
        weights = torch.distributions.Dirichlet(alpha).sample()
        return weights

    def _compute_acquisition_batch(self, predictions, log_sampled_points, beta, weights):
        """Combine multiple acquisition values using scalarization"""
        x_norm = log_sampled_points

        # Get acquisition values for each objective
        acq_values = []
        for model in predictions:
            acq_value = ucb_acquisition_group(model.model, model.likelihood, x_norm, beta)
            acq_values.append(torch.tensor(acq_value))

        # Stack and scalarize
        stacked_acq = torch.stack(acq_values, dim=-1)

        if SCALAR == "AT":
            self.scalarization = AugmentedTchebycheff(
                reference_point=self.current_reference_points,
                rho=self.rho
            )
        else:
            self.scalarization = HypervolumeScalarization(
                nadir_point=self.current_nadir_points,
                exponent=self.output_dim
            )

        scalarized = self.scalarization(stacked_acq, weights)

        return scalarized.numpy()  # Negative for minimization

    def _evaluate_objectives(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all objectives at given points.
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return self.objective_func.evaluate(x)

    def optimize(
            self,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            n_iter: int = 50,
            beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.base_beta = beta
        self.monitor = GPMonitor(dir_name="{}_{}_{}_{}_context_{}".format("MOBO",
                                                            self.input_dim,
                                                            self.output_dim,
                                                            self.model_type,
                                                            self.mobo_id))
        # Initialize Y_train for all objectives [n_initial, output_dim]
        # Y_train = self._evaluate_objectives(X_train)

        # best_scalarized = []

        if NOISE:
            Y_train_noise = Y_train + 0.01 * torch.randn_like(Y_train)

        log_sampled_points = self._sample_points(10000, self.input_dim, 0)
        Y_sampled_points = self._evaluate_objectives(log_sampled_points)

        for iteration in range(n_iter):
            self._update_beta(iteration)

            # Generate random weights
            weights = self._generate_weight_vector(dim=self.output_dim)

            # Train individual models for each objective dimension
            predictions = []
            for i, bo_model in enumerate(self.bo_models):
                if NOISE:
                    X_norm, y_norm = bo_model.normalize_data(
                        X_train.clone(),
                        Y_train_noise[:, i].clone()
                    )
                else:
                    X_norm, y_norm = bo_model.normalize_data(
                        X_train.clone(),
                        Y_train[:, i].clone()
                    )

                model = bo_model.build_model(X_norm, y_norm)
                model.train()
                bo_model.likelihood.train()

                # DEBUG:
                # print("DEBUG: length scale before training is {}.".
                #       format(model.covar_module.base_kernel.lengthscale.detach().item()))

                # Training loop (same as before)
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=0.01) if bo_model.optimizer_type == 'adam' else torch.optim.LBFGS(
                    model.parameters(), lr=0.1, max_iter=20)

                if bo_model.model_type == 'SVGP':
                    mll = gpytorch.mlls.VariationalELBO(
                        bo_model.likelihood,
                        model,
                        num_data=y_norm.size(0)
                    )
                else:
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                        bo_model.likelihood,
                        model
                    )

                if bo_model.optimizer_type == 'lbfgs':
                    prev_loss = float('inf')
                    for _ in range(20):
                        def closure():
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            loss.backward()
                            return loss

                        curr_loss = optimizer.step(closure)
                else:
                    prev_loss = float('inf')
                    for _ in range(bo_model.train_steps):
                        optimizer.zero_grad()
                        output = model(X_norm)
                        loss = -mll(output, y_norm)
                        loss.backward()
                        optimizer.step()
                        prev_loss = loss.item()
                        # print(prev_loss)


                # DEBUG:
                # print("DEBUG: length scale after training is {}.".
                #       format(model.covar_module.base_kernel.lengthscale.detach().item()))

                bo_model.model = model
                predictions.append(bo_model)

                # Log GP model parameters
                self.monitor.log_kernel_params(model, i+1, iteration)
                # Compute Norm Points
                norm_log_sampled_points = (log_sampled_points - self.bo_models[0].x_mean) / self.bo_models[0].x_std
                Y_norm_points = (Y_sampled_points[:, i] - self.bo_models[i].y_mean) / self.bo_models[i].y_std
                # Log predictions for a sample of points
                self.monitor.log_predictions(model,
                                             bo_model.likelihood,
                                             norm_log_sampled_points,
                                             Y_norm_points,
                                             iteration,
                                             i+1)

            if NOISE:
                self._update_and_normalize_reference_points(Y_train_noise)
            else:
                self._update_and_normalize_reference_points(Y_train)

            # Log acquisition function values
            norm_log_sampled_points = (log_sampled_points - self.bo_models[0].x_mean) / self.bo_models[0].x_std
            acq_values = self._compute_acquisition_batch(predictions, norm_log_sampled_points, beta, weights)
            self.monitor.log_acquisition_values(acq_values, iteration)

            if SCALAR == "AT":
                self.scalarization = AugmentedTchebycheff(
                    reference_point=self.current_reference_points,
                    rho=self.rho
                )
            else:
                self.scalarization = HypervolumeScalarization(
                    nadir_point=self.current_nadir_points,
                    exponent=self.output_dim
                )

            # Optimize acquisition function using scalarization
            next_x = optimize_scalarized_acquisition(
                models=predictions,
                scalarization_func=self.scalarization,
                weights=weights,
                input_dim=self.input_dim,
                beta=self.beta,
                x_mean=self.bo_models[0].x_mean,
                x_std=self.bo_models[0].x_std
            )

            # Evaluate new point for all objectives simultaneously
            next_y = self._evaluate_objectives(next_x.unsqueeze(0))
            if NOISE:
                next_y_noise = next_y + 0.01 * torch.randn_like(next_y)

            # Update training data
            X_train = torch.cat([X_train, next_x.unsqueeze(0)])
            Y_train = torch.cat([Y_train, next_y])
            if NOISE:
                Y_train_noise = torch.cat([Y_train_noise, next_y_noise])

            # Update Pareto front
            self._update_pareto_front(X_train, Y_train)
            self.model_list = predictions

            # Log optimization metrics
            metrics = {
                'hypervolume': self.current_hv,
                'pareto_points': self.pareto_front
            }
            self.monitor.log_optimization_metrics(metrics, iteration)

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}, Best y: {self.current_hv:.3f}')

        self.monitor.close()

        return X_train, Y_train


class EHVI(MultiObjectiveBayesianOptimization):
    """Expected Hypervolume Improvement (EHVI) Implementation

    EHVI directly optimizes the expected improvement in hypervolume
    using BoTorch's efficient implementation.
    """

    def __init__(
            self,
            objective_func,
            reference_point=None,
            inducing_points=None,
            train_steps=200,
            model_type='ExactGP',
            optimizer_type='adam',
            rho=0.001,
            mobo_id=None,
            minimize=True  # Flag to specify minimization problem
    ):
        super().__init__(
            objective_func=objective_func,
            reference_point=reference_point,
            inducing_points=inducing_points,
            train_steps=train_steps,
            model_type=model_type,
            optimizer_type=optimizer_type,
            rho=rho,
            mobo_id=mobo_id if mobo_id else "EHVI"
        )

        self.minimize = minimize
        # Store hypervolume improvement history
        self.ehvi_history = []

    def optimize(
            self,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            n_iter: int = 50,
            beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EHVI optimization process:
        1. Train separate GP models for each objective
        2. Calculate EHVI using BoTorch
        3. Select point with maximum EHVI
        4. Update models with new observation
        """
        self.base_beta = beta
        self.monitor = GPMonitor(dir_name=f"EHVI_{self.input_dim}_{self.output_dim}_{self.model_type}_{self.mobo_id}")

        # Initialize Pareto front
        self._update_pareto_front(X_train, Y_train, minimize=self.minimize)

        for iteration in range(n_iter):
            print(f"EHVI Iteration {iteration + 1}/{n_iter}")

            # Train individual models for each objective dimension
            predictions = []
            for i, bo_model in enumerate(self.bo_models):
                if NOISE:
                    Y_train_noise = Y_train + 0.01 * torch.randn_like(Y_train)
                    X_norm, y_norm = bo_model.normalize_data(
                        X_train.clone(),
                        Y_train_noise[:, i].clone()
                    )
                else:
                    X_norm, y_norm = bo_model.normalize_data(
                        X_train.clone(),
                        Y_train[:, i].clone()
                    )

                model = bo_model.build_model(X_norm, y_norm)
                model.train()
                bo_model.likelihood.train()

                # Training loop
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01) \
                    if bo_model.optimizer_type == 'adam' \
                    else torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)

                if bo_model.model_type == 'SVGP':
                    mll = gpytorch.mlls.VariationalELBO(
                        bo_model.likelihood,
                        model,
                        num_data=y_norm.size(0)
                    )
                else:
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                        bo_model.likelihood,
                        model
                    )

                # Optimization loop
                prev_loss = float('inf')
                for _ in range(bo_model.train_steps):
                    if bo_model.optimizer_type == 'lbfgs':
                        def closure():
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            loss.backward()
                            return loss

                        optimizer.step(closure)
                    else:
                        optimizer.zero_grad()
                        output = model(X_norm)
                        loss = -mll(output, y_norm)
                        loss.backward()
                        optimizer.step()
                        prev_loss = loss.item()

                bo_model.model = model
                predictions.append(bo_model)

                # Log GP model parameters
                self.monitor.log_kernel_params(model, i + 1, iteration)

            # Update reference and nadir points if needed
            self._update_and_normalize_reference_points(
                Y_train if not NOISE else (Y_train + 0.01 * torch.randn_like(Y_train)))

            # Use BoTorch EHVI to find the next point
            next_x = optimize_with_ehvi(
                models=predictions,
                pareto_front=self.pareto_front,
                nadir_point=self.nadir_point,
                input_dim=self.input_dim,
                x_mean=self.bo_models[0].x_mean,
                x_std=self.bo_models[0].x_std,
                minimize=self.minimize
            )

            # Evaluate new point for all objectives
            next_y = self._evaluate_objectives(next_x.unsqueeze(0))

            # Update training data
            X_train = torch.cat([X_train, next_x.unsqueeze(0)])
            Y_train = torch.cat([Y_train, next_y])

            # Update Pareto front
            self._update_pareto_front(X_train, Y_train, minimize=self.minimize)
            self.model_list = predictions

            # Log optimization metrics
            metrics = {
                'hypervolume': self.current_hv,
                'pareto_points': self.pareto_front,
            }
            self.monitor.log_optimization_metrics(metrics, iteration)

            if iteration % 5 == 0 or iteration == n_iter - 1:
                print(f'Iteration {iteration}/{n_iter}, HV: {self.current_hv:.3f}')

        self.monitor.close()
        return X_train, Y_train


class ParEGO(MultiObjectiveBayesianOptimization):
    """ParEGO: Pareto Efficient Global Optimization

    ParEGO uses a single surrogate model with augmented Tchebycheff scalarization and
    expected improvement as the acquisition function.
    """

    def __init__(
            self,
            objective_func,
            reference_point=None,
            inducing_points=None,
            train_steps=200,
            model_type='ExactGP',
            optimizer_type='adam',
            rho=0.05,  # ParEGO typically uses higher rho values
            mobo_id=None
    ):
        super().__init__(
            objective_func=objective_func,
            reference_point=reference_point,
            inducing_points=inducing_points,
            train_steps=train_steps,
            model_type=model_type,
            optimizer_type=optimizer_type,
            rho=rho,
            mobo_id=mobo_id if mobo_id else "ParEGO"
        )

        # Missing constructor definition
        self.optimizer_type = optimizer_type

        # Override scalarization to always use Augmented Tchebycheff
        self.scalarization = AugmentedTchebycheff(
            reference_point=self.reference_point,
            rho=rho
        )

        # Create a single BO model for scalarized objectives
        # This is a key difference from the parent class
        self.single_model = None

    def _compute_scalarized_values(self, Y, weights):
        """Compute scalarized values for each point in Y using current weights"""
        return self.scalarization(Y, weights)

    def _compute_acquisition_batch(self, model, log_sampled_points, beta, weights):
        """Use Expected Improvement instead of UCB"""
        x_norm = log_sampled_points

        # Use expected improvement acquisition function
        acq_values = expected_improvement(
            model.model,
            model.likelihood,
            x_norm,
            self.best_scalarized_value
        )

        return acq_values.numpy()

    def optimize(
            self,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            n_iter: int = 50,
            beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ParEGO optimization process:
        1. Generate random weights for Tchebycheff scalarization
        2. Scalarize the objectives to single values
        3. Train a single GP model on scalarized values
        4. Optimize EI acquisition function to find next point
        """
        self.base_beta = beta
        self.monitor = GPMonitor(dir_name=f"ParEGO_{self.input_dim}_{self.output_dim}_{self.model_type}_{self.mobo_id}")

        # Initialize best scalarized value for EI
        self.best_scalarized_value = float('inf')

        # Initialize BO model for single objective (scalarized)
        single_obj = PseudoObjectiveFunction(
            func=lambda x: torch.tensor([0.0]),  # Placeholder, will be updated
            dim=self.input_dim,
        )

        self.single_model = BayesianOptimization(
            objective_func=single_obj,
            inducing_points=None,
            train_steps=self.bo_models[0].train_steps,
            model_type=self.model_type,
            optimizer_type=self.optimizer_type
        )

        log_sampled_points = self._sample_points(10000, self.input_dim, 0)
        Y_sampled_points = self._evaluate_objectives(log_sampled_points)

        for iteration in range(n_iter):
            # Generate new random weights for this iteration (ParEGO approach)
            weights = self._generate_weight_vector(dim=self.output_dim)

            # Compute scalarized values for training data
            if NOISE:
                Y_train_noise = Y_train + 0.01 * torch.randn_like(Y_train)
                self._update_and_normalize_reference_points(Y_train_noise)
                scalarized_values = self._compute_scalarized_values(Y_train_noise, weights)
            else:
                self._update_and_normalize_reference_points(Y_train)
                scalarized_values = self._compute_scalarized_values(Y_train, weights)

            # Update best scalarized value for EI
            self.best_scalarized_value = torch.min(scalarized_values).item()

            # Train single GP model on scalarized data
            X_norm, y_norm = self.single_model.normalize_data(
                X_train.clone(),
                scalarized_values.clone()
            )

            model = self.single_model.build_model(X_norm, y_norm)
            model.train()
            self.single_model.likelihood.train()

            # Training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01) \
                if self.single_model.optimizer_type == 'adam' \
                else torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)

            if self.single_model.model_type == 'SVGP':
                mll = gpytorch.mlls.VariationalELBO(
                    self.single_model.likelihood,
                    model,
                    num_data=y_norm.size(0)
                )
            else:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    self.single_model.likelihood,
                    model
                )

            # Optimization loop for GP model
            prev_loss = float('inf')
            for _ in range(self.single_model.train_steps):
                optimizer.zero_grad()
                output = model(X_norm)
                loss = -mll(output, y_norm)
                loss.backward()
                optimizer.step()
                prev_loss = loss.item()

            self.single_model.model = model

            # Log GP model parameters
            self.monitor.log_kernel_params(model, 0, iteration)

            # Normalize sample points
            norm_log_sampled_points = (log_sampled_points - self.single_model.x_mean) / self.single_model.x_std

            # Optimize acquisition function (EI)
            acq_values = self._compute_acquisition_batch(
                self.single_model,
                norm_log_sampled_points,
                beta,
                weights
            )

            self.monitor.log_acquisition_values(acq_values, iteration)

            # Optimize acquisition function to find next point
            next_x = optimize_scalarized_acquisition_ei(
                model=self.single_model,
                best_value=self.best_scalarized_value,
                input_dim=self.input_dim,
                x_mean=self.single_model.x_mean,
                x_std=self.single_model.x_std
            )

            # Evaluate new point for all objectives
            next_y = self._evaluate_objectives(next_x.unsqueeze(0))
            if NOISE:
                next_y_noise = next_y + 0.01 * torch.randn_like(next_y)

            # Update training data
            X_train = torch.cat([X_train, next_x.unsqueeze(0)])
            Y_train = torch.cat([Y_train, next_y])

            # Update Pareto front
            self._update_pareto_front(X_train, Y_train)

            # Log optimization metrics
            metrics = {
                'hypervolume': self.current_hv,
                'pareto_points': self.pareto_front
            }
            self.monitor.log_optimization_metrics(metrics, iteration)

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}, Best HV: {self.current_hv:.3f}')

        self.monitor.close()
        return X_train, Y_train


class ContextualMultiObjectiveBayesianOptimization:
    def __init__(
            self,
            objective_func,
            reference_point: torch.Tensor = None,
            inducing_points: Optional[torch.Tensor] = None,
            train_steps: int = 200,
            model_type: str = 'ExactGP',
            optimizer_type: str = 'adam',
            rho: float = 0.001
    ):
        self.objective_func = objective_func
        self.input_dim = objective_func.input_dim
        self.output_dim = objective_func.output_dim
        self.context_dim = objective_func.context_dim
        self.dim = self.input_dim + self.context_dim
        self.output_dim = objective_func.output_dim
        self.model_type = model_type
        self.contexts = None
        self.base_beta = None
        self.beta = None
        self.rho = rho

        # Initialize reference point if not provided
        if reference_point is None:
            self.reference_point = torch.zeros(self.output_dim)
        else:
            self.reference_point = reference_point

        # Context-specific reference and nadir points
        self.current_reference_points = {}
        self.global_reference_point = None
        self.current_nadir_points = {}
        self.global_nadir_point = None

        self.nadir_point = self.objective_func.nadir_point
        self.hv = Hypervolume(ref_point=self.nadir_point.numpy())
        self.current_hv = -1

        if SCALAR == "AT":
            self.scalarization = AugmentedTchebycheff(
                reference_point=self.reference_point,
                rho=self.rho
            )
        else:
            self.scalarization = HypervolumeScalarization(
                nadir_point=self.nadir_point,
                exponent=self.output_dim
            )

        new_train_steps = max(600, self.dim * train_steps)
        self.new_train_steps = new_train_steps

        # Create individual BO models for each objective
        self.bo_models = []
        for _ in range(self.output_dim):
            # Create a wrapper single-objective function for each output dimension
            single_obj = PseudoObjectiveFunction(
                func=lambda x, dim=_: self.objective_func.evaluate(x)[:, dim],
                dim=self.dim,
                context_dim=self.context_dim
            )

            bo = ContextualBayesianOptimization(
                objective_func=single_obj,
                inducing_points=inducing_points,
                train_steps=new_train_steps,
                model_type=model_type,
                optimizer_type=optimizer_type
            )
            self.bo_models.append(bo)

        # Initialize hypervolume calculator
        # self.hv = Hypervolume(ref_point=self.reference_point.numpy())

        # Dictionary to track Pareto fronts and hypervolumes per context
        self.context_pareto_fronts = {}
        self.context_pareto_sets = {}
        self.context_hv = {}
        self.model_list = []
        self.monitors = []  # Simple list for monitors
        self.predictions = []

    def _update_context_reference_and_nadir_points(self, context, Y_context):
        """
        Update reference and nadir points for a specific context
        """
        context_key = tuple(context.numpy())

        # Initialize if not exist
        self.current_reference_points[context_key] = torch.min(Y_context, dim=0)[0]
        self.current_reference_points[context_key] = self.current_reference_points[context_key] - 0.1

        self.current_nadir_points[context_key] = torch.max(Y_context, dim=0)[0]
        self.current_nadir_points[context_key] = self.current_nadir_points[context_key] + 0.1 * torch.abs(
            self.current_nadir_points[context_key])

        # Normalize the nadir points and reference points
        for ind in range(self.output_dim):
            self.current_reference_points[context_key][ind] = self.bo_models[ind].normalize_output(
                self.current_reference_points[context_key][ind])
            self.current_nadir_points[context_key][ind] = self.bo_models[ind].normalize_output(
                self.current_nadir_points[context_key][ind])

    def _update_global_reference_and_nadir_points(self, Y_train):
        self.global_reference_point = torch.min(Y_train, dim=0)[0] - 0.1
        self.global_nadir_point = torch.max(Y_train, dim=0)[0] + 0.1 * torch.abs(
            torch.max(Y_train, dim=0)[0])

        for ind in range(self.output_dim):
            self.global_reference_point[ind] = self.bo_models[ind].normalize_output(
                self.global_reference_point[ind])
            self.global_nadir_point[ind] = self.bo_models[ind].normalize_output(
                self.global_nadir_point[ind])

    def _update_beta(self, iteration):
        self.beta = math.sqrt(self.base_beta * log(1 + 2 * iteration))

    def initialize_monitors(self, n_contexts, base_dir_name):
        """Initialize GPMonitor for each context using numeric indexing"""
        self.monitors = [GPMonitor(dir_name=f"{base_dir_name}_context_{i + 1}")
                         for i in range(n_contexts)]

    def _compute_acquisition_batch(self, predictions, log_sampled_points, beta, weights, context):
        """Combine multiple acquisition values using scalarization"""
        context_key = tuple(context.numpy())
        x_norm = log_sampled_points

        # Get acquisition values for each objective
        acq_values = []
        for model in predictions:
            acq_value = ucb_acquisition_group(model["model"], model["likelihood"], x_norm, beta)
            acq_values.append(torch.tensor(acq_value))

        # Stack and scalarize
        stacked_acq = torch.stack(acq_values, dim=-1)

        if SCALAR == "AT":
            self.scalarization = AugmentedTchebycheff(
                reference_point=self.current_reference_points[context_key],
                rho=self.rho
            )
        else:
            self.scalarization = HypervolumeScalarization(
                nadir_point=self.current_nadir_points[context_key],
                exponent=self.output_dim
            )

        scalarized = self.scalarization(stacked_acq, weights)

        return scalarized.numpy()  # Negative for minimization

    @staticmethod
    def _sample_points(n_points, n_decision_vars, n_context_vars):
        """Generate a sample of points for monitoring predictions"""
        total_dims = n_decision_vars + n_context_vars
        return torch.rand(n_points, total_dims)

    @staticmethod
    def _generate_weight_vector(dim: int) -> torch.Tensor:
        """Generate a random weight vector from a Dirichlet distribution."""
        alpha = torch.ones(dim)  # Symmetric Dirichlet distribution
        weights = torch.distributions.Dirichlet(alpha).sample()
        return weights

    def _update_pareto_front_for_context(self, X: torch.Tensor, Y: torch.Tensor, context: torch.Tensor):
        """Update Pareto front for a specific context."""
        context_key = tuple(context.numpy())

        # Convert to numpy for pymoo
        Y_np = Y.numpy()

        # Get non-dominated sorting
        front = NonDominatedSorting().do(Y_np)[0]

        if context_key not in self.context_hv:
            self.context_hv[context_key] = []
        if context_key not in self.context_pareto_fronts:
            self.context_pareto_fronts[context_key] = []
        if context_key not in self.context_pareto_sets:
            self.context_pareto_sets[context_key] = []

        # Update Pareto Front and Pareto Set
        pareto_front = Y[front]
        pareto_set = X[front]
        self.context_pareto_fronts[context_key].append(pareto_front)
        self.context_pareto_sets[context_key].append(pareto_set)

        # Calculate hypervolume
        hv = self.hv.do(pareto_front.numpy())
        self.context_hv[context_key].append(hv)

    def optimize(
            self,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            contexts: torch.Tensor,
            n_iter: int = 50,
            beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.contexts = contexts
        self.base_beta = beta

        n_contexts = contexts.shape[0]
        base_dir_name = "CMOBO_{}_{}_{}".format(self.input_dim,
                                                       self.output_dim,
                                                       self.model_type)
        self.initialize_monitors(n_contexts, base_dir_name)
        log_sampled_points = self._sample_points(10000, self.input_dim, 0)

        if NOISE:
            Y_train_noise = Y_train + 0.01 * torch.randn_like(Y_train)


        # Initialize tracking for each context
        for context in contexts:
            context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
            if torch.any(context_mask):
                Y_context = Y_train[context_mask]
                X_context = X_train[context_mask][:, :self.input_dim]
                self._update_pareto_front_for_context(X_context, Y_context, context)

        for iteration in range(n_iter):
            self._update_beta(iteration)
            # Generate random weights
            weights = self._generate_weight_vector(self.output_dim)

            # Train models for each objective
            predictions = []
            if iteration % 1 == 0:
                for i, bo_model in enumerate(self.bo_models):
                    if NOISE:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train_noise[:, i].clone()
                        )
                    else:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train[:, i].clone()
                        )

                    if iteration > 60:
                        model = bo_model.build_model(X_norm, y_norm, True)
                    else:
                        model = bo_model.build_model(X_norm, y_norm, False)
                    model.train()
                    bo_model.likelihood.train()

                    # Training loop (same as before)
                    optimizer = torch.optim.Adam(model.parameters(),
                                                 lr=0.1) if bo_model.optimizer_type == 'adam' else FullBatchLBFGS(
                        model.parameters())

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[int(self.new_train_steps * 0.5), int(self.new_train_steps * 0.75)],
                        gamma=0.1
                    )
                    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    #     optimizer,
                    #     T_max=self.new_train_steps,  # First cycle length
                    #     eta_min=1e-4
                    # )

                    # Definition of likelihood
                    if bo_model.model_type == 'SVGP':
                        mll = gpytorch.mlls.VariationalELBO(
                            bo_model.likelihood,
                            model,
                            num_data=y_norm.size(0)
                        )
                    else:
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                            bo_model.likelihood,
                            model
                        )

                    # Training Loop
                    if bo_model.optimizer_type == 'lbfgs':
                        def closure():
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            return loss

                        prev_loss = float('inf')
                        loss = closure()
                        loss.backward()
                        for dummy_range in range(60):
                            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

                    else:
                        prev_loss = float('inf')
                        for _ in range(bo_model.train_steps):
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            prev_loss = loss.item()

                            if _ % 100 == 0:
                                print("current loss is {}".format(prev_loss))

                    bo_model.model = model
                    predictions.append({"model": model, "likelihood": bo_model.likelihood})
                    for context_id, context in enumerate(contexts):
                        cur_monitor = self.monitors[context_id]
                        cur_monitor.log_kernel_params(model, i + 1, iteration)
                        temp_log_sampled_points = torch.cat([log_sampled_points,
                                                             context.unsqueeze(0).expand(log_sampled_points.shape[0], -1)],
                                                            dim=1)
                        norm_log_sampled_points = (temp_log_sampled_points - self.bo_models[0].x_mean) / self.bo_models[
                            0].x_std
                        Y_sampled_points = self.objective_func.evaluate(temp_log_sampled_points)[:, i]
                        norm_Y_sampled_points = (Y_sampled_points - self.bo_models[i].y_mean) / self.bo_models[i].y_std
                        cur_monitor.log_predictions(model,
                                                    bo_model.likelihood,
                                                    norm_log_sampled_points,
                                                    norm_Y_sampled_points,
                                                    iteration,
                                                    i + 1)

                self._update_global_reference_and_nadir_points(Y_train)

                for context_id, context in enumerate(contexts):
                    context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                    if torch.any(context_mask):
                        if NOISE:
                            Y_context = Y_train_noise[context_mask]
                        else:
                            Y_context = Y_train[context_mask]
                        # X_context = X_train[context_mask][:, :self.input_dim]
                        self._update_context_reference_and_nadir_points(context, Y_context)

            if len(predictions) > 0:
                self.predictions = predictions
            else:
                predictions = self.predictions

            for context_id, context in enumerate(contexts):
                # Log acquisition function values
                cur_monitor = self.monitors[context_id]
                temp_log_sampled_points = torch.cat([log_sampled_points,
                                                     context.unsqueeze(0).expand(log_sampled_points.shape[0], -1)],
                                                    dim=1)
                norm_log_sampled_points = (temp_log_sampled_points - self.bo_models[0].x_mean) / self.bo_models[0].x_std
                acq_values = self._compute_acquisition_batch(predictions, norm_log_sampled_points, self.beta, weights,
                                                             context)
                cur_monitor.log_acquisition_values(acq_values, iteration)

            # Optimize for each context
            next_points = []
            next_values = []
            next_values_noise = []

            for context in contexts:
                context_key = tuple(context.numpy())

                if SCALAR == "AT":
                    if IF_GLOBAL:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.global_reference_point,
                            rho=self.rho
                        )
                    else:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.current_reference_points[context_key],
                            rho=self.rho
                        )
                else:
                    if IF_GLOBAL:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.global_nadir_point,
                            exponent=self.output_dim
                        )
                    else:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.current_nadir_points[context_key],
                            exponent=self.output_dim
                        )

                next_x = optimize_scalarized_acquisition_for_context(
                    models=predictions,
                    context=context,
                    x_dim=self.input_dim,
                    scalarization_func=self.scalarization,
                    weights=weights,
                    beta=beta,
                    x_mean=self.bo_models[0].x_mean,
                    x_std=self.bo_models[0].x_std
                )

                x_c = torch.cat([next_x, context])
                # print("x_c shape is:{}".format(x_c.shape))
                next_y = self.objective_func.evaluate(x_c.clone().unsqueeze(0))
                if NOISE:
                    next_y_noise = next_y + 0.01 * torch.randn_like(next_y)

                next_points.append(x_c)
                next_values.append(next_y)
                if NOISE:
                    next_values_noise.append(next_y_noise)

                # Update Pareto front for this context
                # context_mask = torch.all(x_c[self.input_dim:] == context, dim=0)
                # if context_mask:
                #     Y_context = torch.cat([
                #         Y_train[torch.all(X_train[:, self.input_dim:] == context, dim=1)],
                #         next_y.unsqueeze(0)
                #     ])
                #     self._update_pareto_front_for_context(context, Y_context)

            # Update training data
            next_points = torch.stack(next_points)
            next_values = torch.stack(next_values)
            if NOISE:
                next_values_noise = torch.stack(next_values_noise)
                Y_train_noise = torch.cat([Y_train_noise, next_values_noise.squeeze(1)])
            X_train = torch.cat([X_train, next_points])
            Y_train = torch.cat([Y_train, next_values.squeeze(1)])


            for context_id, context in enumerate(contexts):
                context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                context_key = tuple(context.numpy())
                if torch.any(context_mask):
                    Y_context = Y_train[context_mask]
                    X_context = X_train[context_mask][:, :self.input_dim]
                    self._update_pareto_front_for_context(X_context, Y_context, context)
                metrics = {
                    'hypervolume': self.context_hv[context_key][-1],
                    'pareto_points': self.context_pareto_fronts[context_key][-1]
                }
                self.monitors[context_id].log_optimization_metrics(metrics, iteration)

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}')
                for context in contexts:
                    context_key = tuple(context.numpy())
                    print(f'Context {context_key}:')
                    print(f'  Hypervolume: {self.context_hv[context_key][-1]:.3f}')
                    print(f'  Pareto front size: {len(self.context_pareto_fronts[context_key])}')

        return X_train, Y_train


class VAEEnhancedCMOBO(ContextualMultiObjectiveBayesianOptimization):
    """
    VAE-enhanced Contextual Multi-Objective Bayesian Optimization.
    Extends the base CMOBO class with VAE capabilities for improved exploration.
    """

    def __init__(
            self,
            objective_func,
            reference_point: torch.Tensor = None,
            inducing_points: Optional[torch.Tensor] = None,
            train_steps: int = 200,
            model_type: str = 'ExactGP',
            optimizer_type: str = 'adam',
            rho: float = 0.001,
            # VAE-specific parameters
            # vae_training_frequency: int = 5,
            vae_training_frequency: int = 2,
            vae_min_data_points: int = 8,
            vae_latent_dim: Optional[int] = None,
            vae_epochs: int = 50,
            vae_batch_size: int = 32,
            use_noise: bool = False,
            scalar_type: str = "HV",
            use_global_reference: bool = True,
            problem_name: str = None
    ):
        # Initialize the parent class
        super().__init__(
            objective_func=objective_func,
            reference_point=reference_point,
            inducing_points=inducing_points,
            train_steps=train_steps,
            model_type=model_type,
            optimizer_type=optimizer_type,
            rho=rho
        )

        # VAE-specific parameters
        self.vae_training_frequency = vae_training_frequency
        self.vae_min_data_points = vae_min_data_points
        self.vae_latent_dim = vae_latent_dim or max(2, self.output_dim - 1)
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.vae_model = None
        self.problem_name = problem_name

        # Settings - override global constants with instance variables for cleaner design
        self.USE_NOISE = use_noise
        self.SCALAR = scalar_type
        self.IF_GLOBAL = use_global_reference

        # New structure for VAE training data (ranks 1 and 2)
        self.vae_training_sets = {}
        self.vae_training_fronts = {}
        self.vae_training_contexts = {}

        # Store all the available training data so far
        self.X_train = None
        self.Y_train = None

    def _update_pareto_front_for_context(self, X: torch.Tensor, Y: torch.Tensor, context: torch.Tensor):
        """
        Override the parent method to additionally collect rank-1 and rank-2 solutions for VAE training.
        """
        context_key = tuple(context.numpy())

        # Create scalarization function based on this weight
        if self.SCALAR == "AT":
            scalarization = AugmentedTchebycheff(
                reference_point=self.global_reference_point,
                rho=self.rho
            )
        else:
            scalarization = HypervolumeScalarization(
                nadir_point=self.global_nadir_point,
                exponent=self.output_dim
            )

        # Convert to numpy for pymoo
        Y_np = Y.numpy()

        # Get non-dominated sorting with multiple fronts
        fronts = NonDominatedSorting().do(Y_np)

        # Initialize regular tracking structures (same as parent class)
        if context_key not in self.context_hv:
            self.context_hv[context_key] = []
        if context_key not in self.context_pareto_fronts:
            self.context_pareto_fronts[context_key] = []
        if context_key not in self.context_pareto_sets:
            self.context_pareto_sets[context_key] = []

        # Initialize VAE training data structures
        if context_key not in self.vae_training_sets:
            self.vae_training_sets[context_key] = []
        if context_key not in self.vae_training_fronts:
            self.vae_training_fronts[context_key] = []
        if context_key not in self.vae_training_contexts:
            self.vae_training_contexts[context_key] = []

        # Update regular Pareto front tracking (rank-1 only) - same as parent class
        pareto_front = Y[fronts[0]]
        pareto_set = X[fronts[0]]
        self.context_pareto_fronts[context_key].append(pareto_front)
        self.context_pareto_sets[context_key].append(pareto_set)

        # Calculate hypervolume using rank-1 solutions only
        hv = self.hv.do(pareto_front.numpy())
        self.context_hv[context_key].append(hv)

        # Collect solutions for VAE training (rank-1 and rank-2)
        vae_sets = []
        vae_fronts = []

        # Include rank-1 solutions
        vae_sets.append(pareto_set)
        vae_fronts.append(pareto_front)

        # Add rank-2 solutions if available
        if len(fronts) > 1 and len(fronts[1]) > 0:
            rank2_front = Y[fronts[1]]
            rank2_set = X[fronts[1]]
            vae_sets.append(rank2_set)
            vae_fronts.append(rank2_front)

        # Combine all solutions
        combined_set = torch.cat(vae_sets) if len(vae_sets) > 0 else torch.tensor([])
        combined_front = torch.cat(vae_fronts) if len(vae_fronts) > 0 else torch.tensor([])

        # Only proceed if we have data to work with
        if len(combined_set) > 0:
            # Compute weight vectors for each solution
            reference_point = self.global_reference_point
            # top_p = 0.4
            top_p = 0.1
            context_mask = torch.all(self.X_train[:, self.input_dim:] == context, dim=1)
            combined_contexts = []
            augmented_vae_sets = []
            augmented_vae_fronts = []
            augmented_contexts = []

            all_X_context = self.X_train[context_mask][:, :self.input_dim]
            all_Y_context = self.Y_train[context_mask]

            for y_value in combined_front:
                weight = self.compute_weight_from_solution(y_value, reference_point, context_key)

                scalarized_values = scalarization(all_Y_context, weight)
                # Find the indices of the top p% solutions according to this weight vector
                num_to_select = max(1, int(len(scalarized_values) * top_p))
                _, top_indices = torch.topk(scalarized_values, num_to_select, largest=False)
                # Combine context and weight for VAE conditioning
                combined_context = torch.cat([context[1:], weight])

                combined_contexts.append(combined_context)

                # Efficiently select solutions and replicate contexts in one shot
                selected_X = all_X_context[top_indices]
                selected_Y = all_Y_context[top_indices]

                # Replicate the context for each selected solution
                replicated_context = combined_context.unsqueeze(0).expand(num_to_select, -1)

                # Add to our lists
                augmented_vae_sets.append(selected_X)
                augmented_vae_fronts.append(selected_Y)
                augmented_contexts.append(replicated_context)

            if len(augmented_vae_sets) > 0:
                augmented_vae_sets = torch.cat(augmented_vae_sets, dim=0)
                augmented_vae_fronts = torch.cat(augmented_vae_fronts, dim=0)
                augmented_contexts = torch.cat(augmented_contexts, dim=0)

                # Store for VAE training
                self.vae_training_sets[context_key].append(augmented_vae_sets)
                self.vae_training_fronts[context_key].append(augmented_vae_fronts)
                self.vae_training_contexts[context_key].append(augmented_contexts)

    def compute_weight_from_solution(self, y, reference_point, context_key):
        """
        Compute weight vector from objective values using the scalarization method.

        For Tchebycheff scalarization: w_i = 1/|f_i - r_i| (normalized)
        For Hypervolume scalarization: w_i = (n_i - f_i) (normalized)

        Args:
            y: Solution's objective values
            reference_point: Reference point for this context
            context_key: Key for the context (used to get nadir point)

        Returns:
            Computed weight vector
        """
        # Check which scalarization method is being used
        if self.SCALAR == "AT":
            # Augmented Tchebycheff scalarization uses reference point
            # For Pareto-optimal solution y, the weights are inversely proportional to |y_i - r_i|
            diff = y - reference_point

            # Avoid division by zero (where y_i = r_i)
            diff = torch.clamp(diff, min=1e-6)

            # Compute weights: w_i = 1/|y_i - r_i|
            weights = 1.0 / diff

        else:
            # Hypervolume scalarization uses nadir point
            nadir_point = self.global_nadir_point

            # For Pareto-optimal solution y, the weights are proportional to (n_i - y_i)
            diff = nadir_point - y

            # Ensure weights are non-negative (nadir should be worse than y, but handle edge cases)
            diff = torch.clamp(diff, min=1e-6)

            # Compute weights: w_i = (n_i - y_i)
            weights = diff

        # Normalize weights to sum to 1
        weights_sum = torch.sum(weights)
        if weights_sum > 1e-10:
            weights = weights / weights_sum
        else:
            # Fallback to uniform weights
            weights = torch.ones_like(y) / len(y)

        return weights

    def initialize_or_update_vae(self, iteration, full_training=False):
        """
        Initialize a new VAE model or update the existing one.

        Args:
            iteration: Current iteration number (for naming)
            full_training: Whether to do full training or incremental update
        """
        # Collect training data from all contexts
        all_X = []
        all_contexts = []

        # Create a TensorBoard writer
        log_dir = os.path.join(f"./log_tier/{self.base_dir_name}_0.1", f"beta_VAE_logs_{iteration}")
        writer = SummaryWriter(log_dir)

        for context_key in self.vae_training_sets.keys():
            if len(self.vae_training_sets[context_key]) > 0:
                # Get the latest data
                latest_set = self.vae_training_sets[context_key][-1]
                latest_contexts = self.vae_training_contexts[context_key][-1]

                all_X.append(latest_set)
                all_contexts.append(latest_contexts)

        if len(all_X) == 0:
            print("No training data available for VAE")
            return

        # Convert lists to tensors
        X_train = torch.cat(all_X)
        contexts_train = torch.cat(all_contexts)

        # Create a custom callback for the VAE trainer
        class TensorBoardCallback:
            def __init__(self, writer, prefix=""):
                self.writer = writer
                self.prefix = prefix
                self.epoch = 0

            def after_epoch(self, epoch, logs):
                self.epoch = epoch
                # Log training metrics
                for key, value in logs.items():
                    self.writer.add_scalar(f"{self.prefix}/{key}", value, epoch)

            def after_batch(self, batch_idx, logs):
                # Log batch-level metrics
                for key, value in logs.items():
                    self.writer.add_scalar(f"{self.prefix}/batch_{key}", value,
                                           self.epoch * batch_idx)

            def log_gradients(self, grad_stats):
                # Log gradient statistics
                for key, value in grad_stats.items():
                    self.writer.add_scalar(f"{self.prefix}/grad_{key}", value, self.epoch)

        # Instantiate callback
        tensorboard_callback = TensorBoardCallback(writer, prefix="VAE_training")

        if len(X_train) < self.vae_min_data_points:
            print(f"Not enough data points for VAE training ({len(X_train)} < {self.vae_min_data_points})")
            return

        # if self.vae_model is None:
            # Initialize new VAE model
        if self.vae_model is None:
            self.vae_model = ParetoVAETrainer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                latent_dim=self.vae_latent_dim,
                context_dim=contexts_train.shape[1],  # Context + weights
                conditional=True,
                epochs=self.vae_epochs,
                batch_size=min(self.vae_batch_size, len(X_train)),
                trainer_id=f"CMOBO_VAE_{iteration}"
            )
            full_training = True  # Always do full training for new model
        else:
            full_training = False

        # Extend the ParetoVAETrainer train method to use our callback
        def train_with_callback(vae_model, X, contexts, callback=None):
            """Wraps the original train method to add callback functionality"""
            data_loader = vae_model.prepare_data(X=X, contexts=contexts)

            for epoch in range(vae_model.epochs):
                epoch_loss = 0
                epoch_mse = 0
                epoch_kld = 0

                for iteration_inner, batch in enumerate(data_loader):
                    # Process batch (simplified, actual implementation in ParetoVAETrainer)
                    if vae_model.conditional:
                        x, c = batch
                        x, c = x.to(vae_model.device), c.to(vae_model.device)
                        recon_x, mean, log_var, z = vae_model.model(x, c)
                    else:
                        x = batch[0].to(vae_model.device)
                        c = None
                        recon_x, mean, log_var, z = vae_model.model(x)

                    # Calculate loss
                    loss, mse, kld = vae_model.loss_fn(recon_x, x, mean, log_var)

                    # Backward pass
                    vae_model.optimizer.zero_grad()
                    loss.backward()

                    # Monitor gradients
                    if callback and iteration_inner % max(1, len(data_loader) // 5) == 0:
                        grad_total, grad_max, grad_min = vae_model.monitor_gradients(epoch, iteration_inner)
                        callback.log_gradients({
                            'total_norm': grad_total,
                            'max_norm': grad_max,
                            'min_norm': grad_min
                        })

                    vae_model.optimizer.step()
                    vae_model.scheduler.step()

                    # Track batch results
                    if callback:
                        callback.after_batch(iteration_inner, {
                            'loss': loss.item(),
                            'mse': mse.item(),
                            'kld': kld.item()
                        })

                    # Track epoch results
                    epoch_loss += loss.item()
                    epoch_mse += mse.item()
                    epoch_kld += kld.item()

                # End of epoch - log metrics
                avg_loss = epoch_loss / len(data_loader)
                vae_model.logs['loss'].append(avg_loss)
                vae_model.logs['mse'].append(epoch_mse / len(data_loader))
                vae_model.logs['kld'].append(epoch_kld / len(data_loader))

                if callback:
                    callback.after_epoch(epoch, {
                        'loss': avg_loss,
                        'mse': epoch_mse / len(data_loader),
                        'kld': epoch_kld / len(data_loader)
                    })

                print(f"Epoch {epoch + 1}/{vae_model.epochs} completed, Avg Loss: {avg_loss:.4f}")

            return vae_model.logs

        if full_training:
            # Full training
            # self.vae_model.train(X=X_train, contexts=contexts_train)
            print(f"VAE fully trained at iteration {iteration} with {len(X_train)} points")
            train_with_callback(self.vae_model, X_train, contexts_train, tensorboard_callback)
            # print(f"VAE fully trained at iteration {iteration} with {len(X_train)} points")

        else:
            # Incremental training
            original_epochs = self.vae_model.epochs
            self.vae_model.epochs = max(10, int(original_epochs * 0.3))  # Use fewer epochs for incremental updates
            # self.vae_model.train(X=X_train, contexts=contexts_train)
            train_with_callback(self.vae_model, X_train, contexts_train, tensorboard_callback)
            self.vae_model.epochs = original_epochs  # Restore original setting
            print(f"VAE incrementally updated at iteration {iteration} with {len(X_train)} points")

    def generate_cvae_candidates(self, context, weight_vector, num_samples=500):
        """
        Generate candidate solutions by sampling from the latent space.
        Optimized for batch evaluation with acquisition functions.

        Args:
            context: Context vector
            weight_vector: Current weight vector
            num_samples: Number of samples to generate

        Returns:
            Tensor of candidate solutions and their full inputs with context
        """
        if self.vae_model is None:
            return None, None

        # Combine context and weight vector
        combined_context = torch.cat([context[1:], weight_vector])

        # Sample latent vectors directly
        latent_dim = self.vae_model.latent_dim
        z_samples = torch.randn(num_samples, latent_dim).to(self.vae_model.device)

        # Create batch of identical contexts
        context_batch = combined_context.unsqueeze(0).expand(num_samples, -1)

        with torch.no_grad():
            # Generate solutions by decoding the latent vectors
            candidates = self.vae_model.model.inference(z_samples, context_batch)

        # Prepare full inputs including context for acquisition function evaluation
        full_candidates = []
        for candidate in candidates:
            full_candidate = torch.cat([candidate, context])
            full_candidates.append(full_candidate)

        full_candidates = torch.stack(full_candidates)

        return candidates.detach(), full_candidates.detach()

    def optimize(
            self,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            contexts: torch.Tensor,
            n_iter: int = 50,
            beta: float = 1.0,
            run: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override the optimize method to incorporate VAE capabilities.
        """
        self.contexts = contexts
        self.base_beta = beta
        self.X_train = X_train.clone()
        self.Y_train = Y_train.clone()

        n_contexts = contexts.shape[0]
        self.base_dir_name = f"VAE_CMOBO_{self.problem_name}_{self.input_dim}_{self.output_dim}_{self.model_type}_{self.vae_training_frequency}_{run}_0.1"
        self.initialize_monitors(n_contexts, self.base_dir_name)
        log_sampled_points = self._sample_points(30000, self.input_dim, 0)

        if self.USE_NOISE:
            Y_train_noise = Y_train + 0.01 * torch.randn_like(Y_train)
        else:
            Y_train_noise = None  # Explicitly set to None if not using noise

        # the update_pareto_front function requires this function call
        # so that the global ref point and nadir point can be obtained
        self._update_global_reference_and_nadir_points(Y_train)
        # Initialize tracking for each context
        for context in contexts:
            context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
            if torch.any(context_mask):
                Y_context = Y_train[context_mask]
                X_context = X_train[context_mask][:, :self.input_dim]
                self._update_pareto_front_for_context(X_context, Y_context, context)

        for iteration in range(n_iter):
            self._update_beta(iteration)
            # Generate random weights
            weights = self._generate_weight_vector(self.output_dim)

            # Train models for each objective
            predictions = []
            if iteration % 1 == 0:
                for i, bo_model in enumerate(self.bo_models):
                    if self.USE_NOISE:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train_noise[:, i].clone()
                        )
                    else:
                        X_norm, y_norm = bo_model.normalize_data(
                            X_train.clone(),
                            Y_train[:, i].clone()
                        )

                    if iteration > 60:
                        model = bo_model.build_model(X_norm, y_norm, True)
                    else:
                        model = bo_model.build_model(X_norm, y_norm, False)
                    model.train()
                    bo_model.likelihood.train()

                    # Training loop (same as before)
                    optimizer = torch.optim.Adam(model.parameters(),
                                                 lr=0.1) if bo_model.optimizer_type == 'adam' else FullBatchLBFGS(
                        model.parameters())

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[int(self.new_train_steps * 0.5), int(self.new_train_steps * 0.75)],
                        gamma=0.1
                    )

                    # Definition of likelihood
                    if bo_model.model_type == 'SVGP':
                        mll = gpytorch.mlls.VariationalELBO(
                            bo_model.likelihood,
                            model,
                            num_data=y_norm.size(0)
                        )
                    else:
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                            bo_model.likelihood,
                            model
                        )

                    # Training Loop
                    if bo_model.optimizer_type == 'lbfgs':
                        def closure():
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            return loss

                        prev_loss = float('inf')
                        loss = closure()
                        loss.backward()
                        for dummy_range in range(60):
                            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

                    else:
                        prev_loss = float('inf')
                        for _ in range(bo_model.train_steps):
                            optimizer.zero_grad()
                            output = model(X_norm)
                            loss = -mll(output, y_norm)
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            prev_loss = loss.item()

                            if _ % 100 == 0:
                                print(f"Current loss is {prev_loss}")

                    bo_model.model = model
                    predictions.append({"model": model, "likelihood": bo_model.likelihood})
                    for context_id, context in enumerate(contexts):
                        cur_monitor = self.monitors[context_id]
                        cur_monitor.log_kernel_params(model, i + 1, iteration)
                        temp_log_sampled_points = torch.cat([log_sampled_points,
                                                             context.unsqueeze(0).expand(log_sampled_points.shape[0],
                                                                                         -1)],
                                                            dim=1)
                        norm_log_sampled_points = (temp_log_sampled_points - self.bo_models[0].x_mean) / self.bo_models[
                            0].x_std
                        Y_sampled_points = self.objective_func.evaluate(temp_log_sampled_points)[:, i]
                        norm_Y_sampled_points = (Y_sampled_points - self.bo_models[i].y_mean) / self.bo_models[i].y_std
                        cur_monitor.log_predictions(model,
                                                    bo_model.likelihood,
                                                    norm_log_sampled_points,
                                                    norm_Y_sampled_points,
                                                    iteration,
                                                    i + 1)

                self._update_global_reference_and_nadir_points(Y_train)

                for context_id, context in enumerate(contexts):
                    context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                    if torch.any(context_mask):
                        if self.USE_NOISE:
                            Y_context = Y_train_noise[context_mask]
                        else:
                            Y_context = Y_train[context_mask]
                        self._update_context_reference_and_nadir_points(context, Y_context)

            if len(predictions) > 0:
                self.predictions = predictions
            else:
                predictions = self.predictions

            for context_id, context in enumerate(contexts):
                # Log acquisition function values
                cur_monitor = self.monitors[context_id]
                temp_log_sampled_points = torch.cat([log_sampled_points,
                                                     context.unsqueeze(0).expand(log_sampled_points.shape[0], -1)],
                                                    dim=1)
                norm_log_sampled_points = (temp_log_sampled_points - self.bo_models[0].x_mean) / self.bo_models[0].x_std
                acq_values = self._compute_acquisition_batch(predictions, norm_log_sampled_points, self.beta, weights,
                                                             context)
                cur_monitor.log_acquisition_values(acq_values, iteration)

            # Train or update VAE model if it's time
            if iteration % self.vae_training_frequency == 0:
                # sum(len(front[-1]) if len(front) > 0 else 0
                #     for front in self.vae_training_sets.values()) >= self.vae_min_data_points):`
                # Do full training periodically or incremental otherwise
                # full_training = (self.vae_model is None or
                #                  iteration % (self.vae_training_frequency * 2) == 0)
                full_training = True
                self.initialize_or_update_vae(iteration, full_training)

            # Optimize for each context
            next_points = []
            next_values = []
            next_values_noise = []

            for context in contexts:
                context_key = tuple(context.numpy())

                # Set up scalarization function
                if self.SCALAR == "AT":
                    if self.IF_GLOBAL:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.global_reference_point,
                            rho=self.rho
                        )
                    else:
                        self.scalarization = AugmentedTchebycheff(
                            reference_point=self.current_reference_points[context_key],
                            rho=self.rho
                        )
                else:
                    if self.IF_GLOBAL:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.global_nadir_point,
                            exponent=self.output_dim
                        )
                    else:
                        self.scalarization = HypervolumeScalarization(
                            nadir_point=self.current_nadir_points[context_key],
                            exponent=self.output_dim
                        )

                # Decide whether to use VAE-generated candidates or traditional acquisition
                use_vae = (self.vae_model is not None and
                           iteration >= self.vae_training_frequency and
                           iteration % self.vae_training_frequency == 0)  # Use VAE every other iteration

                if use_vae:
                    # Generate candidates using VAE with latent perturbation
                    cvae_candidates, full_candidates = self.generate_cvae_candidates(
                        context=context,
                        weight_vector=weights,
                        num_samples=30000
                    )

                    if cvae_candidates is not None and len(cvae_candidates) > 0:
                        # For each candidate, evaluate with acquisition function
                        acq_values = []

                        # Normalize for GP prediction in batch
                        norm_candidates = (full_candidates - self.bo_models[0].x_mean) / self.bo_models[0].x_std

                        # Evaluate acquisition function in batch
                        acq_values = self._compute_acquisition_batch(
                            predictions,
                            norm_candidates,
                            self.beta,
                            weights,
                            context
                        )

                        # Find best candidate
                        best_idx = torch.argmin(torch.tensor(acq_values))
                        next_x = cvae_candidates[best_idx]

                    else:
                        # Fall back to traditional acquisition if VAE fails
                        next_x = optimize_scalarized_acquisition_for_context(
                            models=predictions,
                            context=context,
                            x_dim=self.input_dim,
                            scalarization_func=self.scalarization,
                            weights=weights,
                            beta=beta,
                            x_mean=self.bo_models[0].x_mean,
                            x_std=self.bo_models[0].x_std
                        )
                else:
                    # Use traditional acquisition optimization
                    next_x = optimize_scalarized_acquisition_for_context(
                        models=predictions,
                        context=context,
                        x_dim=self.input_dim,
                        scalarization_func=self.scalarization,
                        weights=weights,
                        beta=beta,
                        x_mean=self.bo_models[0].x_mean,
                        x_std=self.bo_models[0].x_std
                    )

                # Evaluate selected point
                x_c = torch.cat([next_x, context])
                next_y = self.objective_func.evaluate(x_c.clone().unsqueeze(0))

                if self.USE_NOISE:
                    next_y_noise = next_y + 0.01 * torch.randn_like(next_y)
                    next_values_noise.append(next_y_noise)

                next_points.append(x_c)
                next_values.append(next_y)

            # Update training data
            next_points = torch.stack(next_points)
            next_values = torch.stack(next_values)

            if self.USE_NOISE:
                next_values_noise = torch.stack(next_values_noise)
                Y_train_noise = torch.cat([Y_train_noise, next_values_noise.squeeze(1)])

            X_train = torch.cat([X_train, next_points])
            Y_train = torch.cat([Y_train, next_values.detach().squeeze(1)])
            self.X_train = X_train.clone()
            self.Y_train = Y_train.clone()

            # Update Pareto fronts for all contexts
            for context_id, context in enumerate(contexts):
                context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                context_key = tuple(context.numpy())
                if torch.any(context_mask):
                    Y_context = Y_train[context_mask]
                    X_context = X_train[context_mask][:, :self.input_dim]
                    self._update_pareto_front_for_context(X_context, Y_context, context)

                # Log metrics
                metrics = {
                    'hypervolume': self.context_hv[context_key][-1],
                    'pareto_points': len(self.context_pareto_fronts[context_key][-1])
                }
                self.monitors[context_id].log_optimization_metrics(metrics, iteration)

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}')
                for context in contexts:
                    context_key = tuple(context.numpy())
                    print(f'Context {context_key}:')
                    print(f'  Hypervolume: {self.context_hv[context_key][-1]:.3f}')
                    print(f'  Pareto front size: {len(self.context_pareto_fronts[context_key][-1])}')
                    # print(
                    #     f'  VAE training data size: {len(self.vae_training_sets[context_key][-1]) if context_key in self.vae_training_sets and len(self.vae_training_sets[context_key]) > 0 else 0}')

                # Add VAE-specific logging
                # if self.vae_model is not None:
                #     print(f'VAE model status:')
                #     print(f'  Latent dimension: {self.vae_model.latent_dim}')
                #     print(
                #         f'  Last trained: iteration {(iteration // self.vae_training_frequency) * self.vae_training_frequency}')

        return X_train, Y_train
