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
            bounds=(0, 1),
            context_dim=None  # Gamut: Made configurable
    ):
        """
        :param func_name:       Name -> Evaluation
        :param n_objectives:    The number of objectives
        :param n_variables:     The number of decision variables
        :param bounds:          Assume the x lying in [0, 1]^N
        :param context_dim:     Number of context dimensions (turbine: added parameter)
        """
        self.func_name = func_name.lower()
        self.n_objectives = n_objectives

        # turbine: Problem-specific configuration
        if self.func_name == 'turbine':
            self.n_objectives = 2  # Turbine DW: mass proxy + negative power (both minimized)
            self.n_variables = 3  # Turbine DW: radius, height, pitch (design variables)
            self.context_dim = 2  # Turbine DW: wind speed + dummy variable for DTLZ framework consistency
        else:
            # Original DTLZ configuration
            default_k = {
                'dtlz1': 5,
                'dtlz2': 5,
                'dtlz3': 5
            }[self.func_name]

            # Determine n_variables in DLTZ-1/DTLZ-2/DLTZ-3
            # Fix the context_dim
            if n_variables is None:
                self.n_variables = n_objectives + default_k - 1
            else:
                self.n_variables = n_variables

            self.k = self.n_variables - n_objectives + 1
            self.context_dim = 2  # Fixed 2D context for DTLZ

        self.x_dim = self.n_variables
        self.bounds = bounds
        self.input_dim = self.n_variables
        self.output_dim = self.n_objectives

        # turbine: Problem-specific nadir points
        if self.func_name == 'turbine':
            # Turbine DW: Nadir point determination - NEEDS EMPIRICAL VERIFICATION
            # Turbine DW: WARNING: (1.0, 1.0) assumes worst case occurs at normalization bounds
            # Turbine DW: This should be verified by evaluating across all wind speeds and design space
            # Turbine DW: True nadir = max(f1) across all contexts, max(f2) across all contexts
            self.nadir_point = torch.tensor([1.0, 1.0])  # Turbine DW: TENTATIVE - requires validation
        else:
            # Configuring the nadir point via an ad-hoc way?
            self.nadir_point = {
                'dtlz1': (160 + 100 * (self.n_variables - 2)) * torch.ones(self.n_objectives),
                'dtlz2': torch.ones(self.n_objectives) * 2.0,
                'dtlz3': 90 * (self.n_variables + self.n_variables) * torch.ones(self.n_objectives),
            }[self.func_name]

    def scale_x(self, x):
        """Scale decision variables"""
        min_bound, max_bound = self.bounds
        return min_bound + (max_bound - min_bound) * x

    # turbine: Added turbine-specific scaling functions
    def scale_turbine_variables(self, x):
        """
        Scale turbine decision variables from [0,1] to physical units

        Turbine DW: Decision variable ranges explained:
        Turbine DW: - Radius [40,100]m: Practical range for small to medium wind turbines
        Turbine DW:   * 40m = ~80m diameter (small turbine)
        Turbine DW:   * 100m = ~200m diameter (large turbine)
        Turbine DW: - Height [6,15]m: Hub height, typically 0.8-1.5x rotor radius
        Turbine DW: - Pitch [5,20]°: Blade pitch angle for power control and regulation
        Turbine DW:   * 5° = fine pitch for high winds
        Turbine DW:   * 20° = coarse pitch for low winds/startup
        """
        scaled = torch.zeros_like(x)
        scaled[:, 0] = 40 + x[:, 0] * (100 - 40)  # Turbine DW: radius [40,100] m
        scaled[:, 1] = 6 + x[:, 1] * (15 - 6)  # Turbine DW: height [6,15] m
        scaled[:, 2] = (5 + x[:, 2] * (20 - 5)) * (np.pi / 180)  # Turbine DW: pitch [5,20] deg -> rad
        return scaled

    def scale_turbine_context(self, c):
        """
        Scale context from [0,1] to wind speed [4,6] m/s

        Turbine DW: Context scaling explained:
        Turbine DW: - Using c[:, 1] (second dimension) for wind speed
        Turbine DW: - c[:, 0] (first dimension) is dummy/unused for turbine
        Turbine DW: - Wind speed [4,6] m/s represents typical operational range:
        Turbine DW:   * 4 m/s = cut-in wind speed (turbine starts generating power)
        Turbine DW:   * 6 m/s = rated wind speed region (optimal power generation)
        """
        # Turbine DW: Extract wind speed from second context dimension
        wind_speed = 4 + c[:, 1] * 0.5 * (6 - 4)  # Turbine DW: Wind speed from [4,6] m/s
        return wind_speed

    def get_context_shift(self, c):
        """
        Get shift value from first context dimension.
        Maps c[0] from [0,1] to [-0.2, 0.2]
        Hint: Now we simplify the first dim to 0 for all the problems
        """
        # return 0.4 * c[:, 0] - 0.2  # First context dimension controls shift
        return 0 * c[:, 0]

    def get_context_power(self, c):
        """
        Get power scaling from second context dimension.
        Maps c[1] from [0,1] to [0.8, 1.0]
        Hint: Originally we set it up as 0 to 1 but the problems instantiated are dissimilar too much
        """
        return 0.8 + 0.2 * c[:, 1]  # Second context dimension controls power
        # return 0 * c[:, 1] + 1.0

    def g_dtlz1(self, x_m, c):
        """
        Modified DTLZ1 g function with context-dependent shift.
        Context is scaled to [-0.2, 0.2] for shifting.
        """
        c_shift = self.get_context_shift(c)
        c_power = self.get_context_power(c)
        x_shifted = torch.pow(x_m - c_shift.unsqueeze(-1), c_power.unsqueeze(-1))

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
        c_power = self.get_context_power(c)
        x_shifted = torch.pow(x_m - c_shift.unsqueeze(-1), c_power.unsqueeze(-1))

        return torch.sum((x_shifted - 0.5) ** 2, dim=1)

    # turbine: Added turbine-specific performance coefficient calculation
    def _performance_coefficient(self, radius, wind_speed):
        """
        Empirical performance coefficient model

        Turbine DW: Uses tip-speed ratio (TSR) and 6th degree polynomial fit
        Turbine DW: from experimental data to determine power coefficient Cp
        """
        # Turbine DW: Tip speed ratio calculation
        rpm = 8.0  # Turbine DW: Fixed RPM for this analysis
        omega = 2 * np.pi * rpm / 60  # Turbine DW: Convert to rad/s
        tsr = radius * omega / wind_speed  # Turbine DW: Tip speed ratio = (blade tip speed) / (wind speed)

        # Turbine DW: 6th degree polynomial fit coefficients from MATLAB polyfit
        # Turbine DW: These coefficients were derived from experimental turbine data
        p = torch.tensor([-1.05348512795590e-07, 7.08209729939267e-06, -0.000140277525378244,
                          0.000307565692335347, 0.0118771972725566, -0.0352202780490948, 0.0160943595028349])

        # Turbine DW: Evaluate polynomial: Cp = p[0]*TSR^6 + p[1]*TSR^5 + ... + p[6]*TSR^0
        Cp = torch.zeros_like(tsr)
        for i, coef in enumerate(p):
            Cp += coef * (tsr ** (6 - i))

        return Cp

    # turbine: Added turbine power calculation
    def _turbine_power(self, radius, wind_speed):
        """
        Calculate normalized negative power output

        Turbine DW: LEGITIMATE power equation from original code:
        Turbine DW: Reference: "P_out = Cp * 1/2 * fluidDensity * sweptArea * self.v**3"
        Turbine DW: Reference: "f2 = -P_out; # want to maximize power, so we minimize the negative"

        Turbine DW: Standard wind power equation: P = Cp × 0.5 × ρ × A × v³
        Turbine DW: where Cp = performance coefficient, ρ = air density, A = swept area, v = wind speed
        """
        # Turbine DW: Performance coefficient from empirical model
        Cp = self._performance_coefficient(radius, wind_speed)

        # Turbine DW: Power calculation from original code
        # Turbine DW: Reference: "fluidDensity = 1; # can really disregard, assumed constant"
        # Turbine DW: Reference: "sweptArea = np.pi * radius**2;"
        fluid_density = 1.0  # Turbine DW: From original: "fluidDensity = 1"
        swept_area = np.pi * radius ** 2  # Turbine DW: From original: "sweptArea = np.pi * radius**2"
        power = Cp * 0.5 * fluid_density * swept_area * wind_speed ** 3
        negative_power = -power  # Turbine DW: From original: "f2 = -P_out"

        # Turbine DW: Normalization bounds from original code comments
        # Turbine DW: Reference: "min_power = -1096672.537507; # for experiment with wind in 4-6m/s"
        # Turbine DW: Reference: "max_power = 28929.459366;"
        min_power, max_power = -1096672.537507, 28929.459366
        f2 = (negative_power - min_power) / (max_power - min_power)

        return f2

    # turbine: Added complete turbine evaluation function
    def _contextual_turbine(self, x, c):
        """
        Wind turbine evaluation with mass and power objectives
        x: [batch_size, 3] - radius, height, pitch (scaled to physical units)
        c: [batch_size, 2] - context variables, where c[:, 1] is wind speed
        """
        radius = x[:, 0]
        height = x[:, 1]
        pitch = x[:, 2]
        wind_speed = c  # c is already the scaled wind speed from scale_turbine_context

        # Mass objective (f1): SIMPLIFIED PROXY MODEL - NOT ACTUAL MASS
        # Reference: Original turbine.py line "f1 = radius * height * 1/pitch"
        # WARNING: This is NOT a physically accurate mass calculation
        # Real turbine mass should include tower, rotor, nacelle components
        # This appears to be a design complexity metric rather than actual mass
        mass = radius * height * (1.0 / pitch)

        # Normalization bounds from original code comments
        # Reference: "min_mass = 229.1831; # 0,0,1" and "max_mass = 2.1486e+04; #1,1,0"
        min_mass, max_mass = 229.1831, 2.1486e+04
        f1 = (mass - min_mass) / (max_mass - min_mass)

        # Power objective (f2): LEGITIMATE wind power calculation
        # Reference: Original turbine.py "P_out = Cp * 1/2 * fluidDensity * sweptArea * self.v**3"
        # Reference: "f2 = -P_out; # want to maximize power, so we minimize the negative"
        # This follows standard wind turbine power equation: P = Cp × 0.5 × ρ × A × v³
        f2 = self._turbine_power(radius, wind_speed)

        return torch.stack([f1, f2], dim=1)

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the contextual multi-objective function.

        Args:
            inputs: Input tensor of shape [batch_size, n_variables + context_dim]
                   First n_variables columns are decision variables
                   Last context_dim columns are context variables (assumed in [0,1])
        """
        x = inputs[:, :self.n_variables]
        c = inputs[:, self.n_variables:]

        # turbine: Added turbine evaluation branch
        if self.func_name == 'turbine':
            x_scaled = self.scale_turbine_variables(x)
            wind_speed = self.scale_turbine_context(c)  # Returns 1D wind speed tensor
            return self._contextual_turbine(x_scaled, wind_speed)
        else:
            # Original DTLZ evaluation
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

        for i in range(self.n_objectives):
            f[:, i] = 0.5 * (1 + g)

            for j in range(self.n_objectives - 1 - i):
                f[:, i] = f[:, i] * torch.pow(x_p[:, j], power)

            if i > 0:
                f[:, i] = f[:, i] * (1 - torch.pow(x_p[:, self.n_objectives - 1 - i], power))

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


import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


# Assuming the ContextualMultiObjectiveFunction class is available
# from your_module import ContextualMultiObjectiveFunction

def visualize_turbine_landscape():
    """
    Visualize the landscape of turbine multi-objective function
    Step 1: Initialize solutions in a large pool
    Step 2: Evaluate the solutions
    Step 3: Visualize the solutions
    """

    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Wind Turbine Multi-Objective Optimization Landscape', fontsize=16)

    # Test parameters
    n_samples = 1000  # Turbine DW: Large pool of solutions for comprehensive visualization

    print("Visualizing Turbine Problem Landscape...")

    # =====================================================
    # Step 1: Initialize turbine problem
    # =====================================================
    turbine_problem = ContextualMultiObjectiveFunction(func_name='turbine')

    # =====================================================
    # Step 2: Generate and evaluate solutions
    # =====================================================

    # Turbine DW: Input format [radius, height, pitch, dummy_context, wind_speed]
    # All values in [0,1] will be scaled appropriately by the class
    X_turbine = torch.rand(n_samples, turbine_problem.n_variables + turbine_problem.context_dim)

    # Step 3: Evaluate turbine solutions
    Y_turbine = turbine_problem.evaluate(X_turbine)

    print(f"Turbine Problem Analysis:")
    print(f"Input shape: {X_turbine.shape}")
    print(f"Output shape: {Y_turbine.shape}")
    print(f"Mass objective range: [{Y_turbine[:, 0].min():.3f}, {Y_turbine[:, 0].max():.3f}]")
    print(f"Power objective range: [{Y_turbine[:, 1].min():.3f}, {Y_turbine[:, 1].max():.3f}]")

    # Extract design variables and context for analysis
    radius_norm = X_turbine[:, 0].numpy()  # Turbine DW: Normalized radius [0,1]
    height_norm = X_turbine[:, 1].numpy()  # Turbine DW: Normalized height [0,1]
    pitch_norm = X_turbine[:, 2].numpy()  # Turbine DW: Normalized pitch [0,1]
    wind_speed_norm = X_turbine[:, 4].numpy()  # Turbine DW: Normalized wind speed [0,1]

    mass_obj = Y_turbine[:, 0].numpy()  # Turbine DW: Mass objective (minimize)
    power_obj = Y_turbine[:, 1].numpy()  # Turbine DW: Power objective (minimize negative power)

    # =====================================================
    # Plot 1: Main Objective Space (Mass vs Power)
    # =====================================================
    scatter1 = axes[0, 0].scatter(mass_obj, power_obj, c=wind_speed_norm,
                                  cmap='viridis', alpha=0.7, s=30)
    axes[0, 0].set_title('Objective Space: Mass vs Power\n(colored by wind speed)', fontsize=12)
    axes[0, 0].set_xlabel('Mass Objective (f1) - Minimize')
    axes[0, 0].set_ylabel('Power Objective (f2) - Minimize')
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Wind Speed (normalized [0,1])')
    axes[0, 0].grid(True, alpha=0.3)

    # =====================================================
    # Plot 2: Design Space - Radius vs Height
    # =====================================================
    scatter2 = axes[0, 1].scatter(radius_norm, height_norm, c=mass_obj,
                                  cmap='plasma', alpha=0.7, s=30)
    axes[0, 1].set_title('Design Space: Radius vs Height\n(colored by mass objective)', fontsize=12)
    axes[0, 1].set_xlabel('Radius (normalized [0,1])')
    axes[0, 1].set_ylabel('Height (normalized [0,1])')
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('Mass Objective')
    axes[0, 1].grid(True, alpha=0.3)

    # =====================================================
    # Plot 3: Wind Speed vs Power Relationship
    # =====================================================
    scatter3 = axes[0, 2].scatter(wind_speed_norm, power_obj, c=radius_norm,
                                  cmap='coolwarm', alpha=0.7, s=30)
    axes[0, 2].set_title('Context Dependency: Wind Speed vs Power\n(colored by radius)', fontsize=12)
    axes[0, 2].set_xlabel('Wind Speed (normalized [0,1])')
    axes[0, 2].set_ylabel('Power Objective (f2)')
    cbar3 = plt.colorbar(scatter3, ax=axes[0, 2])
    cbar3.set_label('Radius (normalized)')
    axes[0, 2].grid(True, alpha=0.3)

    # =====================================================
    # Plot 4: Pitch Angle Effects
    # =====================================================
    scatter4 = axes[1, 0].scatter(pitch_norm, mass_obj, c=power_obj,
                                  cmap='RdYlBu', alpha=0.7, s=30)
    axes[1, 0].set_title('Pitch vs Mass\n(colored by power objective)', fontsize=12)
    axes[1, 0].set_xlabel('Pitch Angle (normalized [0,1])')
    axes[1, 0].set_ylabel('Mass Objective (f1)')
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 0])
    cbar4.set_label('Power Objective')
    axes[1, 0].grid(True, alpha=0.3)

    # =====================================================
    # Plot 5: 3D Design Space Projection
    # =====================================================
    # Turbine DW: Show relationship between all three design variables
    scatter5 = axes[1, 1].scatter(radius_norm, pitch_norm, c=height_norm,
                                  cmap='spring', alpha=0.7, s=30)
    axes[1, 1].set_title('Design Variables: Radius vs Pitch\n(colored by height)', fontsize=12)
    axes[1, 1].set_xlabel('Radius (normalized [0,1])')
    axes[1, 1].set_ylabel('Pitch Angle (normalized [0,1])')
    cbar5 = plt.colorbar(scatter5, ax=axes[1, 1])
    cbar5.set_label('Height (normalized)')
    axes[1, 1].grid(True, alpha=0.3)

    # =====================================================
    # Plot 6: Pareto Front Analysis
    # =====================================================
    # Turbine DW: Identify approximate Pareto front for different wind speeds

    # Separate solutions by wind speed bins
    wind_bins = np.linspace(0, 1, 4)  # 3 wind speed ranges
    colors = ['red', 'blue', 'green']

    for i in range(len(wind_bins) - 1):
        mask = (wind_speed_norm >= wind_bins[i]) & (wind_speed_norm < wind_bins[i + 1])
        if np.sum(mask) > 0:
            axes[1, 2].scatter(mass_obj[mask], power_obj[mask],
                               c=colors[i], alpha=0.6, s=20,
                               label=f'Wind {wind_bins[i]:.1f}-{wind_bins[i + 1]:.1f}')

    axes[1, 2].set_title('Pareto Front by Wind Speed Ranges', fontsize=12)
    axes[1, 2].set_xlabel('Mass Objective (f1)')
    axes[1, 2].set_ylabel('Power Objective (f2)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # Print Summary Statistics
    # =====================================================
    print("\n" + "=" * 50)
    print("TURBINE LANDSCAPE SUMMARY STATISTICS")
    print("=" * 50)

    # Convert normalized values to physical units for interpretation
    radius_physical = 40 + radius_norm * (100 - 40)  # [40, 100] m
    height_physical = 6 + height_norm * (15 - 6)  # [6, 15] m
    pitch_physical = 5 + pitch_norm * (20 - 5)  # [5, 20] degrees
    wind_speed_physical = 4 + wind_speed_norm * (6 - 4)  # [4, 6] m/s

    print(f"Design Variable Ranges (Physical Units):")
    print(f"  Radius: {radius_physical.min():.1f} - {radius_physical.max():.1f} m")
    print(f"  Height: {height_physical.min():.1f} - {height_physical.max():.1f} m")
    print(f"  Pitch:  {pitch_physical.min():.1f} - {pitch_physical.max():.1f} degrees")
    print(f"  Wind Speed: {wind_speed_physical.min():.1f} - {wind_speed_physical.max():.1f} m/s")

    print(f"\nObjective Statistics:")
    print(f"  Mass Objective (f1): {mass_obj.min():.3f} - {mass_obj.max():.3f}")
    print(f"  Power Objective (f2): {power_obj.min():.3f} - {power_obj.max():.3f}")

    # Find best and worst solutions
    best_mass_idx = np.argmin(mass_obj)
    best_power_idx = np.argmin(power_obj)

    print(f"\nBest Mass Solution:")
    print(f"  Mass: {mass_obj[best_mass_idx]:.3f}, Power: {power_obj[best_mass_idx]:.3f}")
    print(f"  Radius: {radius_physical[best_mass_idx]:.1f}m, Height: {height_physical[best_mass_idx]:.1f}m")
    print(f"  Pitch: {pitch_physical[best_mass_idx]:.1f}°, Wind: {wind_speed_physical[best_mass_idx]:.1f}m/s")

    print(f"\nBest Power Solution:")
    print(f"  Mass: {mass_obj[best_power_idx]:.3f}, Power: {power_obj[best_power_idx]:.3f}")
    print(f"  Radius: {radius_physical[best_power_idx]:.1f}m, Height: {height_physical[best_power_idx]:.1f}m")
    print(f"  Pitch: {pitch_physical[best_power_idx]:.1f}°, Wind: {wind_speed_physical[best_power_idx]:.1f}m/s")

    return fig, axes, X_turbine, Y_turbine


# Example usage
if __name__ == "__main__":
    # Turbine DW: Run the turbine landscape visualization
    fig, axes, inputs, outputs = visualize_turbine_landscape()

    print(f"\nVisualization complete!")
    print(f"Generated {inputs.shape[0]} solutions for analysis")
    print(f"Turbine problem has {inputs.shape[1] - 2} design variables and 2 context dimensions")
    print(f"Outputs: {outputs.shape[1]} objectives (mass, power)")