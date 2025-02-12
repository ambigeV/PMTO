import torch
import gpytorch
from .models import SVGPModel, ExactGPModel
from .acquisition import optimize_acquisition, optimize_scalarized_acquisition, \
    optimize_acquisition_for_context, optimize_scalarized_acquisition_for_context
from typing import Callable, Optional, Tuple, List, Dict
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class AugmentedTchebycheff:
    """Augmented Tchebycheff scalarization"""

    def __init__(self, reference_point: torch.Tensor, rho: float = 0.05):
        self.reference_point = reference_point
        self.rho = rho

    def __call__(self, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_diff = weights * torch.abs(y - self.reference_point)
        max_term = torch.max(weighted_diff, dim=-1)[0]
        sum_term = self.rho * torch.sum(weighted_diff, dim=-1)
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

    def build_model(self, X_train, y_train):
        if self.model_type == 'SVGP':
            model = SVGPModel(self.inducing_points, input_dim=self.dim)
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

            for _ in range(self.train_steps):
                def closure():
                    optimizer.zero_grad()
                    output = model(X_train_norm)
                    loss = -mll(output, y_train_norm)
                    loss.backward()
                    return loss

                optimizer.step(closure if self.optimizer_type == 'lbfgs' else None)

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

    def build_model(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Build GP model based on specified type."""
        if self.model_type == 'SVGP':
            model = SVGPModel(self.inducing_points, input_dim=self.dim)
        else:
            model = ExactGPModel(X_train, y_train, self.likelihood)
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
            for _ in range(self.train_steps):
                def closure():
                    optimizer.zero_grad()
                    output = model(X_train_norm)
                    loss = -mll(output, y_train_norm)
                    loss.backward()
                    return loss

                optimizer.step(closure if self.optimizer_type == 'lbfgs' else None)

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

            if iteration % 5 == 0:
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
            train_steps=500,
            model_type='ExactGP',
            optimizer_type='adam',
            rho=0.05,
    ):
        self.objective_func = objective_func
        self.input_dim = objective_func.input_dim
        self.output_dim = objective_func.output_dim

        # Initialize reference point if not provided
        if reference_point is None:
            self.reference_point = torch.zeros(self.output_dim)
        else:
            self.reference_point = reference_point

        self.nadir_point = self.objective_func.nadir_point
        self.hv = Hypervolume(ref_point=self.nadir_point.numpy())
        self.current_hv = -1

        # Initialize scalarization
        self.scalarization = AugmentedTchebycheff(
            reference_point=self.reference_point,
            rho=rho
        )

        # Create individual BO instances for each objective dimension
        self.bo_models = []
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

    @staticmethod
    def _generate_weight_vector(dim: int) -> torch.Tensor:
        """Generate a random weight vector from a Dirichlet distribution."""
        alpha = torch.ones(dim)  # Symmetric Dirichlet distribution
        weights = torch.distributions.Dirichlet(alpha).sample()
        return weights

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

        # Initialize Y_train for all objectives [n_initial, output_dim]
        # Y_train = self._evaluate_objectives(X_train)

        # best_scalarized = []

        for iteration in range(n_iter):

            # Generate random weights
            weights = self._generate_weight_vector(dim=self.output_dim)

            # Train individual models for each objective dimension
            predictions = []
            for i, bo_model in enumerate(self.bo_models):
                X_norm, y_norm = bo_model.normalize_data(
                    X_train.clone(),
                    Y_train[:, i].clone()
                )

                model = bo_model.build_model(X_norm, y_norm)
                model.train()
                bo_model.likelihood.train()

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

                for _ in range(bo_model.train_steps):
                    def closure():
                        optimizer.zero_grad()
                        output = model(X_norm)
                        loss = -mll(output, y_norm)
                        loss.backward()
                        return loss

                    optimizer.step(
                        closure if bo_model.optimizer_type == 'lbfgs' else None
                    )

                bo_model.model = model
                predictions.append(bo_model)

            # Optimize acquisition function using scalarization
            next_x = optimize_scalarized_acquisition(
                models=predictions,
                scalarization_func=self.scalarization,
                weights=weights,
                input_dim=self.input_dim,
                beta=beta,
            )

            # Evaluate new point for all objectives simultaneously
            next_y = self._evaluate_objectives(next_x.unsqueeze(0))

            # Update training data
            X_train = torch.cat([X_train, next_x.unsqueeze(0)])
            Y_train = torch.cat([Y_train, next_y])

            # Update Pareto front
            self._update_pareto_front(X_train, Y_train)
            self.model_list = predictions

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}, Best y: {self.current_hv:.3f}')

        return X_train, Y_train


class ContextualMultiObjectiveBayesianOptimization:
    def __init__(
            self,
            objective_func,
            reference_point: torch.Tensor = None,
            inducing_points: Optional[torch.Tensor] = None,
            train_steps: int = 500,
            model_type: str = 'ExactGP',
            optimizer_type: str = 'adam',
            rho: float = 0.05
    ):
        self.objective_func = objective_func
        self.input_dim = objective_func.input_dim
        self.output_dim = objective_func.output_dim
        self.context_dim = objective_func.context_dim
        self.dim = self.input_dim + self.context_dim
        self.output_dim = objective_func.output_dim
        self.contexts = None

        # Initialize reference point if not provided
        if reference_point is None:
            self.reference_point = torch.zeros(self.output_dim)
        else:
            self.reference_point = reference_point

        self.scalarization = AugmentedTchebycheff(
            reference_point=self.reference_point,
            rho=rho
        )

        self.nadir_point = self.objective_func.nadir_point
        self.hv = Hypervolume(ref_point=self.nadir_point.numpy())
        self.current_hv = -1

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
                train_steps=train_steps,
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
        # Initialize tracking for each context
        for context in contexts:
            context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
            if torch.any(context_mask):
                Y_context = Y_train[context_mask]
                X_context = X_train[context_mask][:, :self.input_dim]
                self._update_pareto_front_for_context(X_context, Y_context, context)

        for iteration in range(n_iter):
            # Generate random weights
            weights = self._generate_weight_vector(self.output_dim)

            # Train models for each objective
            predictions = []
            for i, bo_model in enumerate(self.bo_models):
                X_norm, y_norm = bo_model.normalize_data(
                    X_train.clone(),
                    Y_train[:, i].clone()
                )

                model = bo_model.build_model(X_norm, y_norm)
                model.train()
                bo_model.likelihood.train()

                # Training loop (same as before)
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=0.01) if bo_model.optimizer_type == 'adam' else torch.optim.LBFGS(
                    model.parameters(), lr=0.1, max_iter=20)

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
                for _ in range(bo_model.train_steps):
                    def closure():
                        optimizer.zero_grad()
                        output = model(X_norm)
                        loss = -mll(output, y_norm)
                        loss.backward()
                        return loss

                    optimizer.step(
                        closure if bo_model.optimizer_type == 'lbfgs' else None
                    )

                predictions.append({"model": model, "likelihood": bo_model.likelihood})

            # Optimize for each context
            next_points = []
            next_values = []

            for context in contexts:
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

                next_points.append(x_c)
                next_values.append(next_y)

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

            X_train = torch.cat([X_train, next_points])
            Y_train = torch.cat([Y_train, next_values.squeeze(1)])

            for context in contexts:
                context_mask = torch.all(X_train[:, self.input_dim:] == context, dim=1)
                if torch.any(context_mask):
                    Y_context = Y_train[context_mask]
                    X_context = X_train[context_mask][:, :self.input_dim]
                    self._update_pareto_front_for_context(X_context, Y_context, context)

            if iteration % 5 == 0:
                print(f'Iteration {iteration}/{n_iter}')
                for context in contexts:
                    context_key = tuple(context.numpy())
                    print(f'Context {context_key}:')
                    print(f'  Hypervolume: {self.context_hv[context_key][-1]:.3f}')
                    print(f'  Pareto front size: {len(self.context_pareto_fronts[context_key])}')

        return X_train, Y_train

