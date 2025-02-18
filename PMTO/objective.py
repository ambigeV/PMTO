import torch
import numpy as np


# Objective Function Class with Scaling
class ObjectiveFunction:
    def __init__(self, func_name='rastrigin', dim=10, bounds=(-5, 5)):
        self.func_name = func_name.lower()
        self.dim = dim
        self.bounds = bounds  # Tuple (min, max) for scaling

    def scale(self, x):
        min_bound, max_bound = self.bounds
        return min_bound + (max_bound - min_bound) * x

    def evaluate(self, x):
        x_scaled = self.scale(x)
        if self.func_name == 'ackley':
            return self.ackley(x_scaled)
        elif self.func_name == 'rastrigin':
            return self.rastrigin(x_scaled)
        elif self.func_name == 'rosenbrock':
            return self.rosenbrock(x_scaled)
        else:
            raise ValueError("Unsupported function. Choose from 'ackley', 'rastrigin', 'rosenbrock'.")

    def ackley(self, x):
        if_minimum = False
        nega = -1 if if_minimum else 1

        a = 20
        b = 0.2
        c = 2 * np.pi
        d = self.dim
        sum1 = torch.sum(x ** 2)
        sum2 = torch.sum(torch.cos(c * x))
        return nega * (-a * torch.exp(-b * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + a + np.e)

    def rastrigin(self, x):
        if_minimum = False
        nega = -1 if if_minimum else 1

        A = 10
        return nega * (A * self.dim + torch.sum(x ** 2 - A * torch.cos(2 * np.pi * x)))

    def rosenbrock(self, x):
        if_minimum = False
        nega = -1 if if_minimum else 1

        x_i = x[:-1]
        x_next = x[1:]
        return nega * torch.sum(100 * (x_next - x_i ** 2) ** 2 + (1 - x_i) ** 2)


class MultiObjectiveFunction:
    def __init__(self, func_name='dtlz1', n_objectives=2, n_variables=None, bounds=(0, 1)):
        """
        Initialize multi-objective test problem.

        Args:
            func_name: Name of the test problem ('dtlz1', 'dtlz2', 'dtlz3')
            n_objectives: Number of objectives (M)
            n_variables: Number of variables (n). If None, set to n_objectives + k - 1
            bounds: Tuple (min, max) for scaling
        """
        self.func_name = func_name.lower()
        self.n_objectives = n_objectives

        # k represents the number of position parameters
        default_k = {
            'dtlz1': 10,
            'dtlz2': 10,
            'dtlz3': 10
        }[self.func_name]

        if n_variables is None:
            # Use default k value to determine n
            self.n_variables = n_objectives + default_k - 1
        else:
            # Calculate actual k based on provided n and m
            self.n_variables = n_variables

        self.nadir_point = {
            'dtlz1': (120 + 100 * (self.n_variables - 2)) * torch.ones(self.n_objectives),
            'dtlz2': 1.25 ** (self.n_variables - 1) * torch.ones(self.n_objectives),
            'dtlz3': 100 * (self.n_variables + self.n_variables * 1.25) * torch.ones(self.n_objectives),
        }[self.func_name]

        # k is now calculated based on n and m
        self.k = self.n_variables - n_objectives + 1

        self.bounds = bounds
        self.input_dim = self.n_variables
        self.output_dim = self.n_objectives

    def scale(self, x):
        """Scale input from [0,1] to bounds"""
        min_bound, max_bound = self.bounds
        return min_bound + (max_bound - min_bound) * x

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the multi-objective function.

        Args:
            x: Input tensor of shape [batch_size, n_variables]

        Returns:
            Tensor of shape [batch_size, n_objectives]
        """
        if self.func_name == 'dtlz1':
            return self.dtlz1(x)
        elif self.func_name == 'dtlz2':
            return self.dtlz2(x)
        elif self.func_name == 'dtlz3':
            return self.dtlz3(x)
        else:
            raise ValueError("Unsupported function. Choose from 'dtlz1', 'dtlz2', 'dtlz3'.")

    def g_dtlz1(self, x_m):
        """Helper function for DTLZ1"""
        return 100 * (x_m.shape[1] + torch.sum(
            (x_m - 0.5) ** 2 - torch.cos(20 * np.pi * (x_m - 0.5)),
            dim=1
        ))

    def g_dtlz2(self, x_m):
        """Helper function for DTLZ2 and DTLZ3"""
        return torch.sum((x_m - 0.5) ** 2, dim=1)

    def dtlz1(self, x):
        """
        DTLZ1 test problem.

        Properties:
        - Linear Pareto front
        - Multi-modal landscape with 11^(n-M+1) local Pareto-optimal fronts
        """
        x = self.scale(x)

        # Split x into position parameters (x_p) and distance parameters (x_m)
        x_p = x[:, :self.n_objectives - 1]  # First M-1 variables
        x_m = x[:, self.n_objectives - 1:]  # Remaining variables

        g = self.g_dtlz1(x_m)

        f = torch.zeros((x.shape[0], self.n_objectives))

        # Calculate objectives
        for i in range(self.n_objectives):
            f[:, i] = 0.5 * (1 + g)

            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * x_p[:, j]

            if i > 0:
                f[:, i] = f[:, i] * (1 - x_p[:, self.n_objectives - 1 - i])

        return f

    def dtlz2(self, x):
        """
        DTLZ2 test problem.

        Properties:
        - Spherical Pareto front
        - Tests the ability to scale with number of objectives
        """
        x = self.scale(x)

        x_p = x[:, :self.n_objectives - 1]
        x_m = x[:, self.n_objectives - 1:]

        g = self.g_dtlz2(x_m)

        f = torch.zeros((x.shape[0], self.n_objectives))

        # Calculate objectives
        for i in range(self.n_objectives):
            f[:, i] = 1 + g

            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.cos(x_p[:, j] * np.pi / 2)

            if i > 0:
                f[:, i] = f[:, i] * torch.sin(x_p[:, self.n_objectives - 1 - i] * np.pi / 2)

        return f

    def dtlz3(self, x):
        """
        DTLZ3 test problem.

        Properties:
        - Spherical Pareto front
        - Multi-modal landscape with 3^(n-M+1) local Pareto-optimal fronts
        """
        x = self.scale(x)

        x_p = x[:, :self.n_objectives - 1]
        x_m = x[:, self.n_objectives - 1:]

        # Modified g function for DTLZ3
        g = self.g_dtlz1(x_m)

        f = torch.zeros((x.shape[0], self.n_objectives))

        # Calculate objectives (same as DTLZ2 but with different g)
        for i in range(self.n_objectives):
            f[:, i] = 1 + g

            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.cos(x_p[:, j] * np.pi / 2)

            if i > 0:
                f[:, i] = f[:, i] * torch.sin(x_p[:, self.n_objectives - 1 - i] * np.pi / 2)

        return f


class ContextualMultiObjectiveFunction:
    def __init__(
            self,
            func_name='dtlz1',
            n_objectives=2,
            n_variables=None,
            bounds=(0, 1)
    ):
        self.func_name = func_name.lower()
        self.n_objectives = n_objectives

        default_k = {
            'dtlz1': 5,
            'dtlz2': 5,
            'dtlz3': 5
        }[self.func_name]

        if n_variables is None:
            self.n_variables = n_objectives + default_k - 1
        else:
            self.n_variables = n_variables

        self.k = self.n_variables - n_objectives + 1
        self.x_dim = self.n_variables
        # self.context_dim = self.k
        self.context_dim = 2  # Fixed 2D context

        self.bounds = bounds
        self.input_dim = self.n_variables
        self.output_dim = self.n_objectives

        self.nadir_point = {
            'dtlz1': (160 + 100 * (self.n_variables - 2)) * torch.ones(self.n_objectives),
            'dtlz2': 1.25 ** (self.n_variables - 1) * torch.ones(self.n_objectives) + 0.5,
            'dtlz3': 100 * (self.n_variables + self.n_variables * 1.25) * torch.ones(self.n_objectives),
        }[self.func_name]

    def scale_x(self, x):
        """Scale decision variables"""
        min_bound, max_bound = self.bounds
        return min_bound + (max_bound - min_bound) * x

    def get_context_shift(self, c):
        """
        Get shift value from first context dimension.
        Maps c[0] from [0,1] to [-0.2, 0.2]
        """
        return 0.4 * c[:, 0] - 0.2  # First context dimension controls shift

    def get_context_power(self, c):
        """
        Get power scaling from second context dimension.
        Maps c[1] from [0,1] to [0.8, 1.0]
        """
        # return 0.8 + 0.2 * c[:, 1]  # Second context dimension controls power
        return 0 * c[:, 1] + 1.0

    def g_dtlz1(self, x_m, c):
        """
        Modified DTLZ1 g function with context-dependent shift.
        Context is scaled to [-0.2, 0.2] for shifting.
        """
        c_shift = self.get_context_shift(c)
        x_shifted = x_m - c_shift.unsqueeze(-1)

        return 100 * (x_m.shape[1] + torch.sum(
            (x_shifted - 0.5) ** 2 - torch.cos(20 * np.pi * (x_shifted - 0.5)),
            dim=1
        ))

    def g_dtlz2(self, x_m, c):
        """
        Modified DTLZ2 g function with context-dependent shift.
        Context is scaled to [-0.2, 0.2] for shifting.
        """
        c_shift = self.get_context_shift(c)
        x_shifted = x_m - c_shift.unsqueeze(-1)
        return torch.sum((x_shifted - 0.5) ** 2, dim=1)

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the contextual multi-objective function.

        Args:
            inputs: Input tensor of shape [batch_size, n_variables + k]
                   First n_variables columns are decision variables
                   Last k columns are context variables (assumed in [0,1])
        """
        x = inputs[:, :self.n_variables]
        c = inputs[:, self.n_variables:]

        x_scaled = self.scale_x(x)

        if self.func_name == 'dtlz1':
            return self._contextual_dtlz1(x_scaled, c)
        elif self.func_name == 'dtlz2':
            return self._contextual_dtlz2(x_scaled, c)
        elif self.func_name == 'dtlz3':
            return self._contextual_dtlz3(x_scaled, c)
        else:
            raise ValueError("Unsupported function.")

    def _contextual_dtlz1(self, x, c):
        """
        Contextual DTLZ1 with:
        1. Context shift in [-0.2, 0.2] for g function
        2. Power scaling in [0.8, 1] for decision variables
        """
        x_p = x[:, :self.n_objectives - 1]
        x_m = x[:, self.n_objectives - 1:]

        g = self.g_dtlz1(x_m, c)
        power = self.get_context_power(c)

        f = torch.zeros((x.shape[0], self.n_objectives))

        print("Shapes are g:{}, f:{}, power:{}.".format(g.shape, f.shape, power.shape))

        for i in range(self.n_objectives):
            f[:, i] = 0.5 * (1 + g)

            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.pow(x_p[:, j], power)

            if i > 0:
                f[:, i] = f[:, i] * torch.pow(1 - x_p[:, self.n_objectives - 1 - i], power)

        return f

    def _contextual_dtlz2(self, x, c):
        """
        Contextual DTLZ2 with:
        1. Context shift in [-0.2, 0.2] for g function
        2. Power scaling in [0.8, 1] for decision variables
        """
        x_p = x[:, :self.n_objectives - 1]
        x_m = x[:, self.n_objectives - 1:]

        g = self.g_dtlz2(x_m, c)
        power = self.get_context_power(c)

        f = torch.zeros((x.shape[0], self.n_objectives))

        for i in range(self.n_objectives):
            f[:, i] = 1 + g

            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.cos(torch.pow(x_p[:, j], power) * np.pi / 2)

            if i > 0:
                f[:, i] = f[:, i] * torch.sin(
                    torch.pow(x_p[:, self.n_objectives - 1 - i], power) * np.pi / 2
                )

        return f

    def _contextual_dtlz3(self, x, c):
        """
        Contextual DTLZ3 with:
        1. Context shift in [-0.2, 0.2] for g function (using DTLZ1's g)
        2. Power scaling in [0.8, 1] for decision variables
        """
        x_p = x[:, :self.n_objectives - 1]
        x_m = x[:, self.n_objectives - 1:]

        g = self.g_dtlz1(x_m, c)
        power = self.get_context_power(c)

        f = torch.zeros((x.shape[0], self.n_objectives))

        for i in range(self.n_objectives):
            f[:, i] = 1 + g

            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.cos(torch.pow(x_p[:, j], power) * np.pi / 2)

            if i > 0:
                f[:, i] = f[:, i] * torch.sin(
                    torch.pow(x_p[:, self.n_objectives - 1 - i], power) * np.pi / 2
                )

        return f
