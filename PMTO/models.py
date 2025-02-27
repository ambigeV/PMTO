import gpytorch
import torch


# SVGP Model Definition
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, input_dim):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)  # Automatic Relevance Determination (ARD)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Exact GP Model Definition
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Option 1: Use constraint
        lengthscale_constraint = gpytorch.constraints.Interval(0.1, 2.0)

        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=
        #                                                                             lengthscale_constraint))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # Option 2: Set priors
        self.covar_module.base_kernel.register_prior(
            'lengthscale_prior',
            gpytorch.priors.LogNormalPrior(0, 1),  # mean=0, variance=1 in log-space
            'lengthscale'
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Ard GP Model Definition
class ArdGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ArdGPModel, self).__init__(train_x, train_y, likelihood)
        lengthscale_constraint = gpytorch.constraints.Interval(0.1, 2.0)
        self.no_of_data, self.no_of_dim = train_x.shape
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.no_of_dim,
                                                                                    lengthscale_constraint=lengthscale_constraint))
        self.covar_module.base_kernel.register_prior(
            'lengthscale_prior',
            gpytorch.priors.LogNormalPrior(
                loc=torch.zeros(self.no_of_dim),
                scale=torch.ones(self.no_of_dim)
            ),
            'lengthscale'
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CompositeKernel(gpytorch.kernels.Kernel):
    def __init__(self, n_decision_vars, n_context_vars):
        super().__init__()
        self.n_decision = n_decision_vars
        self.n_context = n_context_vars
        self.k_decision = gpytorch.kernels.RBFKernel()
        self.k_context = gpytorch.kernels.RBFKernel(ard_num_dims=self.n_context)

    def get_lengthscales(self):
        """
        Returns the lengthscales of both decision and context kernels.

        Returns:
            tuple: (decision_lengthscale, context_lengthscales)
            - decision_lengthscale: single value (isotropic RBF)
            - context_lengthscales: array of values (ARD RBF)
        """
        # Get raw parameters
        decision_lengthscale = self.k_decision.lengthscale.detach().item()
        context_lengthscales = self.k_context.lengthscale.detach()

        # Convert float to tensor and reshape
        dec_tensor = torch.tensor([[decision_lengthscale]], dtype=torch.float32)  # make it a tensor

        # Concatenate along dimension 1
        all_lengthscales = torch.cat([dec_tensor, context_lengthscales], dim=1)

        return all_lengthscales

    def forward(self, x1, x2, diag=False, **params):
        # Split input into decision and context variables
        # print(x1.shape)
        # print(x2.shape)

        x1_decision = x1[:, :self.n_decision]
        x1_context = x1[:, self.n_decision:]
        x2_decision = x2[:, :self.n_decision]
        x2_context = x2[:, self.n_decision:]

        # print("Split shapes:")
        # print(f"x1_decision: {x1_decision.shape}")
        # print(f"x1_context: {x1_context.shape}")
        # print(f"x2_decision: {x2_decision.shape}")
        # print(f"x2_context: {x2_context.shape}")

        # Compute kernels separately
        k_dec = self.k_decision.forward(x1_decision, x2_decision, diag=diag)
        # print(k_dec.shape)
        k_ctx = self.k_context.forward(x1_context, x2_context, diag=diag)
        # print(k_ctx.shape)

        return k_dec * k_ctx

class CustomGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_decision_vars, n_context_vars):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(CompositeKernel(n_decision_vars, n_context_vars))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
