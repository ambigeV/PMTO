import gpytorch


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
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Ard GP Model Definition
class ArdGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ArdGPModel, self).__init__(train_x, train_y, likelihood)
        self.no_of_data, self.no_of_dim = train_x.shape
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.no_of_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CompositeKernel(gpytorch.kernels.Kernel):
    def __init__(self, n_decision_vars, n_context_vars):
        super().__init__()
        self.k_decision = gpytorch.kernels.RBFKernel()
        self.k_context = gpytorch.kernels.RBFKernel()
        self.n_decision = n_decision_vars
        self.n_context = n_context_vars

    def forward(self, x1, x2, diag=False, **params):
        # Split input into decision and context variables
        x1_decision = x1[:, :self.n_decision]
        x1_context = x1[:, self.n_decision:]
        x2_decision = x2[:, :self.n_decision]
        x2_context = x2[:, self.n_decision:]

        # Compute kernels separately
        k_dec = self.k_decision.forward(x1_decision, x2_decision, diag=diag)
        k_ctx = self.k_context.forward(x1_context, x2_context, diag=diag)

        return k_dec * k_ctx

class CustomGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_decision_vars, n_context_vars):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = CompositeKernel(n_decision_vars, n_context_vars)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
