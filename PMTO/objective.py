import torch
import magpylib as magpy
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
        # Magnetic sifter configuration
        if self.func_name == 'magnetic_sifter':
            self.n_objectives = 3  # Sifter: f1 (unwanted capture), f2 (separation), f3 (thickness)
            self.n_variables = 3  # Sifter: gap, mag_len, thickness (design variables)
            self.context_dim = 3  # Sifter: dummy + ms1 + ms2 (1 dummy + 2 magnetic moments)

            # Sifter: Fixed physical parameters from original code
            self.grid_res = 0.0005
            self.Q = 15  # mL/hr flow rate
            self.cell_r = 0.0075  # Cell radius in mm

        elif self.func_name == 'gridshell':
            self.n_objectives = 2  # Gridshell: morning power + evening power (both minimized)

            # Gridshell: Grid dimensions - configurable
            self.numX = 5  # Default 5x5 grid, can be made configurable
            self.numY = 5
            self.n_variables = (self.numX - 2) * (self.numY - 2)  # Only interior points are variables
            self.context_dim = 2  # Gridshell: 1 dummy + 1 building orientation

            # Gridshell: Fixed solar parameters (as torch tensors)
            self.morning_sun = torch.tensor([-0.5, -0.5, -0.6])  # Morning sun direction
            self.evening_sun = torch.tensor([-0.5, 0.5, -0.6])   # Evening sun direction
            self.regularizer_weight = 0.2  # Balance between power and smoothness

        elif self.func_name == 'turbine':
            self.n_objectives = 2  # Turbine DW: mass proxy + negative power (both minimized)
            self.n_variables = 3  # Turbine DW: radius, height, pitch (design variables)
            self.context_dim = 2  # Turbine DW: wind speed + dummy variable for DTLZ framework consistency

        elif self.func_name == 'bicopter':
            self.n_objectives = 2  # Bicopter DW: distance to goal + energy consumption (both minimized)
            self.n_variables = 12  # Bicopter DW: 6 time steps × 2 actuators = 12 control variables
            self.context_dim = 3  # Bicopter DW: dummy + length + density (3 context dims total)
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
        # Sifter: Nadir point - will be set empirically later as requested
        if self.func_name == 'magnetic_sifter':
            self.nadir_point = torch.tensor([1.0, 1.0, 1.0])  # Placeholder - to be determined empirically
        elif self.func_name == 'gridshell':
            # Gridshell: Nadir point - to be determined empirically
            self.nadir_point = torch.tensor([1.0, 1.0])
        elif self.func_name == 'turbine':
            # Turbine DW: Nadir point determination - NEEDS EMPIRICAL VERIFICATION
            # Turbine DW: WARNING: (1.0, 1.0) assumes worst case occurs at normalization bounds
            # Turbine DW: This should be verified by evaluating across all wind speeds and design space
            # Turbine DW: True nadir = max(f1) across all contexts, max(f2) across all contexts
            self.nadir_point = torch.tensor([1.0, 1.0])  # Turbine DW: TENTATIVE - requires validation
        elif self.func_name == 'bicopter':
            # Bicopter DW: Nadir point left undefined as requested
            # Bicopter DW: Will need empirical determination later
            self.nadir_point = torch.tensor([1.0, 1.0])  # Bicopter DW: To be determined empirically
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

    # bicopter: Added bicopter-specific scaling functions
    def scale_bicopter_variables(self, x):
        """
        Scale bicopter control variables from [0,1] to thrust commands

        Bicopter DW: Control variable ranges explained:
        Bicopter DW: - 12 variables = 6 time steps × 2 actuators (reduced from 32 to 12)
        Bicopter DW: - Each control input scaled from [0,1] to [-5,5] thrust units
        Bicopter DW: - Negative values allow reverse thrust for more complex maneuvers
        """
        # Bicopter DW: Scale control inputs from [0,1] to [-5,5] thrust units
        scaled = -5 + x * 10  # Bicopter DW: Map [0,1] to [-5,5]
        return scaled

    def scale_bicopter_context(self, c):
        """
        Scale context from [0,1] to physical bicopter parameters

        Bicopter DW: Context scaling explained:
        Bicopter DW: - c[:, 0] (first dimension) is DUMMY/UNUSED for framework consistency
        Bicopter DW: - c[:, 1] (second dimension) controls LENGTH independently
        Bicopter DW: - c[:, 2] (third dimension) controls DENSITY independently
        Bicopter DW: - Length range [0.5, 2.0] meters
        Bicopter DW: - Density range [0.5, 2.0] kg/m
        Bicopter DW: - This allows full 2D exploration of length-density space with dummy consistency
        """
        # Bicopter DW: Independent scaling of length and density (ignore dummy c[:, 0])
        length = 1.0 + c[:, 1] * 0.5  # Bicopter DW: Length from [0.5, 2.0] m using c[:, 1]
        density = 1.0 + c[:, 2] * 0.5  # Bicopter DW: Density from [0.5, 2.0] kg/m using c[:, 2]

        return length, density

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

    # bicopter: Added bicopter dynamics simulation
    def _bicopter_dynamics(self, u, length, density):
        """
        Simulate bicopter dynamics and return normalized objectives

        Bicopter DW: Physics simulation with proper normalization
        Bicopter DW: u: [batch_size, 12] scaled control inputs in [-5,5]
        Bicopter DW: length: [batch_size] length values in [0.5, 2.0] m
        Bicopter DW: density: [batch_size] density values in [0.5, 2.0] kg/m

        Returns:
            f1_norm: Distance to goal objective in [0,1] (0=perfect, 1=worst)
            f2_norm: Energy objective in [0,1] (0=minimal energy, 1=maximum energy)
        """
        batch_size = u.shape[0]
        n_steps = 6  # Bicopter DW: 6 time steps (reduced from 16)
        g = -9.81  # Bicopter DW: Gravity acceleration
        dt = 0.25  # Bicopter DW: Time step

        # Bicopter DW: Physical parameters from context
        mass = length * density  # Bicopter DW: Mass = length × density
        inertia = (1 / 12) * mass * (2 * length) ** 2  # Bicopter DW: Rotational inertia

        # Bicopter DW: Initial and goal states
        q_init = torch.zeros(batch_size, 6)  # Bicopter DW: [x, y, θ, vx, vy, vθ]
        goal_state = torch.zeros(batch_size, 6)
        goal_state[:, 0] = 1.0  # Bicopter DW: Target x-position = 1.0

        # Bicopter DW: Initialize state and objectives
        current_state = q_init.clone()
        total_energy = torch.zeros(batch_size)

        # Bicopter DW: Reshape control inputs: [batch_size, 12] -> [batch_size, 6, 2]
        u_reshaped = u.view(batch_size, n_steps, 2)

        # Bicopter DW: Simulate dynamics over time steps
        for step in range(n_steps):
            # Bicopter DW: Extract current state
            x, y, theta = current_state[:, 0], current_state[:, 1], current_state[:, 2]
            vx, vy, vtheta = current_state[:, 3], current_state[:, 4], current_state[:, 5]

            # Bicopter DW: Control inputs for this time step
            u1, u2 = u_reshaped[:, step, 0], u_reshaped[:, step, 1]

            # Bicopter DW: Energy calculation (quadratic in control effort)
            step_energy = 0.5 * (u1 ** 2 + u2 ** 2) * dt
            total_energy += step_energy

            # Bicopter DW: Forces and torques
            total_thrust = u1 + u2
            differential_thrust = u1 - u2

            # Bicopter DW: Accelerations in body frame
            ax = -total_thrust * torch.sin(theta) / mass
            ay = total_thrust * torch.cos(theta) / mass + g
            alpha = length * differential_thrust / inertia

            # Bicopter DW: Update velocities
            vx_new = vx + ax * dt
            vy_new = vy + ay * dt
            vtheta_new = vtheta + alpha * dt

            # Bicopter DW: Update positions
            x_new = x + vx_new * dt
            y_new = y + vy_new * dt
            theta_new = theta + vtheta_new * dt

            # Bicopter DW: Update state vector
            current_state = torch.stack([x_new, y_new, theta_new, vx_new, vy_new, vtheta_new], dim=1)

        # Bicopter DW: Calculate final distance to goal
        distance_error = torch.norm(current_state - goal_state, dim=1)

        # Bicopter DW: Normalize objectives to [0,1] based on reasonable bounds
        # Bicopter DW: These bounds are estimated based on physics and typical performance
        max_distance = 5.0  # Bicopter DW: Maximum reasonable distance error
        max_energy = 100.0  # Bicopter DW: Maximum reasonable energy consumption

        # f1_norm = torch.clamp(distance_error / max_distance, 0, 1)
        # f2_norm = torch.clamp(total_energy / max_energy, 0, 1)
        f1_norm = distance_error / max_distance / 7.0
        f2_norm = total_energy / max_energy * 4.0

        return f1_norm, f2_norm

    # bicopter: Added complete bicopter evaluation function
    def _contextual_bicopter(self, x, c):
        """
        Bicopter evaluation with distance and energy objectives

        Bicopter DW: x: [batch_size, 12] - scaled control inputs
        Bicopter DW: c: [batch_size, 2] - context variables encoding length and density
        """
        # Bicopter DW: Scale control inputs to physical units
        u_scaled = self.scale_bicopter_variables(x)

        # Bicopter DW: Extract physical parameters from context
        length, density = self.scale_bicopter_context(c)

        # Bicopter DW: Simulate bicopter dynamics
        f1, f2 = self._bicopter_dynamics(u_scaled, length, density)

        return torch.stack([f1, f2], dim=1)

    # sifter: Scale Variables
    def scale_sifter_variables(self, x):
        """
        Scale magnetic sifter decision variables from [0,1] to physical units

        Sifter: Decision variable ranges from original ExTrEMO code:
        Sifter: - gap: [0.02, 0.10] mm (x[0]*0.08 + 0.02)
        Sifter: - mag_len: [0.01, 0.10] mm (x[1]*0.09 + 0.01)
        Sifter: - thickness: [0.002, 0.015] mm (x[2]*0.013 + 0.002)
        """
        scaled = torch.zeros_like(x)
        scaled[:, 0] = 0.02 + x[:, 0] * 0.06  # Sifter: gap [0.02, 0.10] mm
        scaled[:, 1] = 0.01 + x[:, 1] * 0.06  # Sifter: mag_len [0.01, 0.10] mm
        scaled[:, 2] = 0.002 + x[:, 2] * 0.013  # Sifter: thickness [0.002, 0.015] mm
        return scaled

    # sifter: Scale Contexts
    def scale_sifter_context(self, c):
        """
        Scale context from [0,1] to magnetic moment ranges

        Sifter: Context scaling from Table II in ExTrEMO paper:
        Sifter: - c[:, 0] (first dimension) is DUMMY/UNUSED for framework consistency
        Sifter: - c[:, 1] (second dimension) controls ms1 (cell type 1 magnetic moment)
        Sifter: - c[:, 2] (third dimension) controls ms2 (cell type 2 magnetic moment)

        Sifter: Shrunk ranges for highly similar cells (as requested):
        Sifter: - ms1 range: [1.5, 2.0] × 10^-13 (shrunk from [0.160, 3.130])
        Sifter: - ms2 range: [0.5, 0.8] × 10^-13 (shrunk from [0.055, 0.656])
        """
        # Sifter: Scale magnetic moments to shrunk ranges for similar cells
        ms1 = (1.5 + c[:, 1] * 0.5) * 1e-13  # Sifter: ms1 from [1.5, 2.0] × 10^-13
        ms2 = (0.5 + c[:, 2] * 0.3) * 1e-13  # Sifter: ms2 from [0.5, 0.8] × 10^-13

        return ms1, ms2

    # sifter: sub-helper functions
    def _calc_F_mag(self, gap, mag_len, t, cell_r):
        """
        Calculate magnetic field and gradients using magpylib (original implementation)

        Sifter: This is the exact implementation from your provided code
        Sifter: gap, mag_len, t are single float values (not tensors)
        """
        mag_max = 1.5  # Magnetization in mT
        y_max = 100  # Distance in y for saturation, units in mm

        # Sifter: Create 10 magnetic cuboid sources as in original
        src1 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(2 * mag_len, y_max, t),
                                   position=(-1 * (4.5 * gap + 6.0 * mag_len), 0, 0))
        src2 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(mag_len, y_max, t),
                                   position=(-1 * (3.5 * gap + 4.5 * mag_len), 0, 0))
        src3 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(mag_len, y_max, t),
                                   position=(-1 * (2.5 * gap + 3.5 * mag_len), 0, 0))
        src4 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(2 * mag_len, y_max, t),
                                   position=(-1 * (1.5 * gap + 2.0 * mag_len), 0, 0))
        src5 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(mag_len, y_max, t),
                                   position=(-1 * (0.5 * gap + 0.5 * mag_len), 0, 0))
        src6 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(mag_len, y_max, t),
                                   position=(1 * (0.5 * gap + 0.5 * mag_len), 0, 0))
        src7 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(2 * mag_len, y_max, t),
                                   position=(1 * (1.5 * gap + 2.0 * mag_len), 0, 0))
        src8 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(mag_len, y_max, t),
                                   position=(1 * (2.5 * gap + 3.5 * mag_len), 0, 0))
        src9 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(mag_len, y_max, t),
                                   position=(1 * (3.5 * gap + 4.5 * mag_len), 0, 0))
        src10 = magpy.magnet.Cuboid(magnetization=(mag_max, 0, 0), dimension=(2 * mag_len, y_max, t),
                                    position=(1 * (4.5 * gap + 6.0 * mag_len), 0, 0))

        c = magpy.Collection(src1, src2, src3, src4, src5, src6, src7, src8, src9, src10)

        # Sifter: Create observation grid as in original
        max_dim = max([gap, mag_len, t])
        ts = np.arange(-2 * max_dim, 2 * max_dim, self.grid_res)
        observer = np.array([[(x, 0., z) for x in ts] for z in ts])

        # Sifter: Calculate magnetic field using magpylib
        B = c.getB(observer)
        B_mag = (B[:, :, 0] ** 2 + B[:, :, 1] ** 2 + B[:, :, 2] ** 2) ** 0.5
        grad_B = np.gradient(B_mag, self.grid_res)  # grad_B[0] is dB_dz; grad_B[1] is dB_dx

        return observer, grad_B[0], B_mag

    def _calc_F_drag(self, gap, mag_len, Q, cell_r):
        """
        Calculate drag force (original implementation)
        """
        u_ave = (Q * 1e-6 / (3600 * 3913 * 4 * 4 * 1e-10)) * (0.6 * 0.6) / (((3 * gap) / (3 * gap + 4 * mag_len)) ** 2)
        F_drag = 6 * np.pi * 0.001 * cell_r * u_ave * 1e-3
        return F_drag

    def _calc_dBdz(self, gap, t, cell_r, grid_res, dBdz, observer, ms, F_drag):
        """
        Calculate capture efficiency using original implementation

        Sifter: This is the exact calc_dBdz function from your provided code
        """
        theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        x_part, y_part = cell_r * np.sin(theta), cell_r * np.cos(theta)
        x_part_rd, y_part_rd = np.fix(cell_r * np.sin(theta) * 1e3) / 1e3, np.fix(cell_r * np.cos(theta) * 1e3) / 1e3

        cell_all_loc_x = np.arange(0., (gap / 2 - cell_r), grid_res * 2)
        cell_all_loc_z = np.arange(-t / 2 - 0.01, -0.003, grid_res)

        CE_frac = 0.0

        for b in cell_all_loc_z:
            cell_z = b + y_part_rd
            cur_max_dBdz = 0.0

            for a in cell_all_loc_x:
                cell_x = a + x_part_rd

                dBdz_sum = 0.0
                for i in range(len(cell_x)):
                    cur_x = cell_x[i]
                    cur_z = cell_z[i]

                    xx_search = np.where(np.abs(observer[0, :, 0] - cur_x) < (grid_res) * 0.75)
                    zz_search = np.where(np.abs(observer[:, 0, 2] - cur_z) < (grid_res) * 0.75)

                    if len(xx_search[0]) > 0 and len(zz_search[0]) > 0:
                        dBdz_sum += dBdz[zz_search[0][0], xx_search[0][0]]

                dBdz_ave = dBdz_sum / len(cell_x)

                if (dBdz_ave * ms * 30 > F_drag):
                    cur_max_dBdz += 1

            CE_frac_cur = cur_max_dBdz / (len(cell_all_loc_x)) if len(cell_all_loc_x) > 0 else 0.0

            if (CE_frac_cur > CE_frac):
                CE_frac = CE_frac_cur

        return CE_frac

    # sifter: helper function
    def _calc_CE(self, gap, mag_len, t, grid_res, ms, Q, cell_r):
        """
        Calculate capture efficiency using original implementation

        Sifter: This is the exact calc_CE function from your provided code
        """
        F_drag = self._calc_F_drag(gap, mag_len, Q, cell_r)
        observer, dBdz, B_mag = self._calc_F_mag(gap, mag_len, t, cell_r)
        CE = self._calc_dBdz(gap, t, cell_r, grid_res, dBdz, observer, ms, F_drag)
        return CE

    # sifter: Contextual Evaluation
    def _contextual_magnetic_sifter(self, x, c):
        """
        Magnetic sifter evaluation with full magpylib simulation

        Sifter: x: [batch_size, 3] - scaled design variables (gap, mag_len, thickness)
        Sifter: c: (ms1, ms2) - scaled magnetic moments for both cell types
        """
        batch_size = x.shape[0]
        ms1, ms2 = c

        # Sifter: Convert tensors to numpy for magpylib compatibility
        x_np = x.detach().cpu().numpy()
        ms1_np = ms1.detach().cpu().numpy()
        ms2_np = ms2.detach().cpu().numpy()

        # Sifter: Initialize result arrays
        f1_results = np.zeros(batch_size)
        f2_results = np.zeros(batch_size)
        f3_results = np.zeros(batch_size)

        # Sifter: Process each sample in the batch individually
        for i in range(batch_size):
            gap = float(x_np[i, 0])
            mag_len = float(x_np[i, 1])
            thickness = float(x_np[i, 2])
            ms1_val = float(ms1_np[i])
            ms2_val = float(ms2_np[i])

            # Sifter: Calculate capture efficiencies using full simulation
            f1_raw = self._calc_CE(gap, mag_len, thickness, self.grid_res, ms1_val, self.Q, self.cell_r)
            f2_raw = self._calc_CE(gap, mag_len, thickness, self.grid_res, ms2_val, self.Q, self.cell_r)

            # Sifter: Construct objectives following original structure
            # Sifter: Original: f = np.array([-f1, -(f1 - f2), f3 - 1])
            # Sifter: Modified for minimization as requested:
            f1_results[i] = f1_raw * 0.95  # Minimize unwanted cell capture (was -f1)
            f2_results[i] = (f2_raw - f1_raw + 1.0) * 0.95  # Minimize negative separation (was -(f1-f2))
            f3_results[i] = thickness / 0.015  # Normalize thickness to [0,1] for minimization

        # Sifter: Convert results back to PyTorch tensors
        f1_tensor = torch.tensor(f1_results, dtype=x.dtype, device=x.device)
        f2_tensor = torch.tensor(f2_results, dtype=x.dtype, device=x.device)
        f3_tensor = torch.tensor(f3_results, dtype=x.dtype, device=x.device)

        return torch.stack([f1_tensor, f2_tensor, f3_tensor], dim=1)

    # GridShell:
    def scale_gridshell_variables(self, x):
        """
        Scale gridshell decision variables from [0,1] to height values

        Gridshell: Decision variable ranges explained:
        Gridshell: - Each interior grid point z-coordinate: [0, 5] units (height/elevation)
        Gridshell: - Only interior points (numX-2)×(numY-2) are optimizable
        Gridshell: - Boundary points are fixed at z=0
        """
        # Gridshell: Scale all interior point heights from [0,1] to [0,5]
        scaled = x * 5.0  # Map [0,1] to [0,5] height units
        return scaled

    def scale_gridshell_context(self, c):
        """
        Scale context from [0,1] to building orientation angle [0, 2π]

        Gridshell: Context scaling explained:
        Gridshell: - c[:, 0] (first dimension) is DUMMY/UNUSED for framework consistency
        Gridshell: - c[:, 1] (second dimension) controls building orientation
        Gridshell: - Range [0, 2π] radians (full 360° rotation)
        Gridshell: - Controls how building is oriented relative to sun positions

        Args:
            c: [batch_size, 2] context variables in [0,1]

        Returns:
            house_theta: [batch_size] building orientation angles in [0,2π]
        """
        house_theta = c[:, 1] * 0.5 * torch.pi  # Building orientation angle from c[:, 1]
        return house_theta

    def _rotate_about_z(self, theta, vec):
        """
        Rotate 3D vector about Z-axis by angle theta

        Args:
            theta: [batch_size] rotation angles in radians
            vec: [batch_size, 3] or [3] vectors to rotate

        Returns:
            rotated_vec: [batch_size, 3] rotated vectors
        """
        batch_size = theta.shape[0]
        device = theta.device
        dtype = theta.dtype

        # Create rotation matrices for each angle
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Build rotation matrix: [cos -sin 0; sin cos 0; 0 0 1]
        rot_matrices = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        rot_matrices[:, 0, 0] = cos_theta
        rot_matrices[:, 0, 1] = -sin_theta
        rot_matrices[:, 1, 0] = sin_theta
        rot_matrices[:, 1, 1] = cos_theta
        rot_matrices[:, 2, 2] = 1.0

        # Handle broadcasting for vector input
        if vec.dim() == 1:  # Single vector [3]
            vec_expanded = vec.unsqueeze(0).expand(batch_size, -1).to(device)
        else:  # Already [batch_size, 3]
            vec_expanded = vec.to(device)

        # Apply rotation: vec_out = R @ vec
        rotated_vec = torch.bmm(rot_matrices, vec_expanded.unsqueeze(-1)).squeeze(-1)

        return rotated_vec

    def _reconstruct_grid(self, x_interior, numX, numY):
        """
        Reconstruct full grid from interior point variables

        Args:
            x_interior: [batch_size, (numX-2)*(numY-2)] interior point heights
            numX, numY: Grid dimensions

        Returns:
            pts_grid: [batch_size, numX, numY] full grid with fixed boundaries
        """
        batch_size = x_interior.shape[0]
        device = x_interior.device
        dtype = x_interior.dtype

        # Initialize full grid with zeros (boundary conditions)
        pts_grid = torch.zeros(batch_size, numX, numY, device=device, dtype=dtype)

        # Reshape interior variables to 2D grid layout
        interior_2d = x_interior.view(batch_size, numX - 2, numY - 2)

        # Place interior variables in the center of the grid
        pts_grid[:, 1:-1, 1:-1] = interior_2d

        return pts_grid

    def _gridshell_smoothness(self, pts_grid):
        """
        Calculate smoothness regularization term

        Args:
            pts_grid: [batch_size, numX, numY] grid of z-coordinates

        Returns:
            smoothness: [batch_size] smoothness penalty values
        """
        batch_size, numX, numY = pts_grid.shape

        # Calculate first derivatives (differences)
        diff_y = (pts_grid[:, 1:, :] - pts_grid[:, :-1, :]) * numY  # [batch_size, numX-1, numY]
        diff_x = (pts_grid[:, :, 1:] - pts_grid[:, :, :-1]) * numX  # [batch_size, numX, numY-1]

        # Calculate second derivatives (differences of differences)
        diff_y2 = (diff_y[:, 1:, :] - diff_y[:, :-1, :]) * numY  # [batch_size, numX-2, numY]
        diff_x2 = (diff_x[:, :, 1:] - diff_x[:, :, :-1]) * numX  # [batch_size, numX, numY-2]

        # Sum squared second derivatives
        smoothness_y = torch.sum(diff_y2 ** 2, dim=(1, 2)) / numY  # [batch_size]
        smoothness_x = torch.sum(diff_x2 ** 2, dim=(1, 2)) / numX  # [batch_size]

        # Total smoothness penalty (normalized as in MATLAB)
        smoothness = (smoothness_y + smoothness_x) / 1000000.0

        return smoothness

    def _gridshell_power_output(self, pts_grid, sun_dir, house_theta):
        """
        Calculate solar power output using vectorized surface normal computation

        Args:
            pts_grid: [batch_size, numX, numY] grid of z-coordinates
            sun_dir: [3] sun direction vector (morning or evening)
            house_theta: [batch_size] building orientation angles

        Returns:
            power: [batch_size] normalized power output values
        """
        batch_size, numX, numY = pts_grid.shape
        device = pts_grid.device

        # Ensure sun_dir is on correct device
        sun_dir = sun_dir.to(device)

        # Normalize sun direction
        sun_dir_norm = sun_dir / torch.norm(sun_dir)

        # Rotate sun direction by -house_theta to account for building orientation
        sun_dir_rotated = self._rotate_about_z(-house_theta, sun_dir_norm)  # [batch_size, 3]

        # Calculate surface differences for triangulation
        # vec1: horizontal differences (j+1) - (j)
        vec1 = pts_grid[:, :, 1:] - pts_grid[:, :, :-1]  # [batch_size, numX, numY-1]
        # vec2: vertical differences (i+1) - (i)
        vec2 = pts_grid[:, 1:, :] - pts_grid[:, :-1, :]  # [batch_size, numX-1, numY]

        # === TRIANGLE A CALCULATIONS (upper-left triangles) ===
        # Take overlapping region for triangle A: [batch_size, numX-1, numY-1]
        vec1_A = vec1[:, :-1, :]  # Crop last row: [batch_size, numX-1, numY-1]
        vec2_A = vec2[:, :, :-1]  # Crop last col: [batch_size, numX-1, numY-1]

        # Build 3D vectors for cross product: [0, 1, vec1] × [1, 0, vec2]
        zeros = torch.zeros_like(vec1_A)
        ones = torch.ones_like(vec1_A)

        # First vector: [0, 1, vec1_A]
        v1_A = torch.stack([zeros, ones, vec1_A], dim=-1)  # [batch_size, numX-1, numY-1, 3]
        # Second vector: [1, 0, vec2_A]
        v2_A = torch.stack([ones, zeros, vec2_A], dim=-1)  # [batch_size, numX-1, numY-1, 3]

        # Vectorized cross product and normalization
        normals_A = torch.cross(v1_A, v2_A, dim=-1)  # [batch_size, numX-1, numY-1, 3]
        norms_A = torch.norm(normals_A, dim=-1, keepdim=True)  # [batch_size, numX-1, numY-1, 1]
        normals_A_unit = normals_A / (norms_A + 1e-8)  # Avoid division by zero

        # Calculate incidence angles
        sun_expanded = sun_dir_rotated.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, 3]
        dots_A = torch.sum(normals_A_unit * sun_expanded, dim=-1)  # [batch_size, numX-1, numY-1]
        incidence_A = 1 - dots_A ** 2  # [batch_size, numX-1, numY-1]

        # === TRIANGLE B CALCULATIONS (lower-right triangles) ===
        # Match MATLAB indexing exactly: vec1(2:end, :) and vec2(:, 2:end)
        vec1_B = -vec1[:, 1:, :]  # Crop first row: [batch_size, numX-1, numY-1]
        vec2_B = -vec2[:, :, 1:]  # Crop first col: [batch_size, numX-1, numY-1]

        # Build 3D vectors: [0, -1, -vec1] × [-1, 0, -vec2]
        v1_B = torch.stack([zeros, -ones, vec1_B], dim=-1)  # [batch_size, numX-1, numY-1, 3]
        v2_B = torch.stack([-ones, zeros, vec2_B], dim=-1)  # [batch_size, numX-1, numY-1, 3]

        # Vectorized cross product and normalization
        normals_B = torch.cross(v1_B, v2_B, dim=-1)  # [batch_size, numX-1, numY-1, 3]
        norms_B = torch.norm(normals_B, dim=-1, keepdim=True)
        normals_B_unit = normals_B / (norms_B + 1e-8)

        # Calculate incidence angles
        dots_B = torch.sum(normals_B_unit * sun_expanded, dim=-1)
        incidence_B = 1 - dots_B ** 2

        # === TOTAL POWER CALCULATION ===
        # Sum all triangle contributions and normalize
        total_incidence = torch.sum(incidence_A, dim=(1, 2)) + torch.sum(incidence_B, dim=(1, 2))
        num_triangles = 2 * (numX - 1) * (numY - 1)  # Each quad has 2 triangles
        power = total_incidence / num_triangles  # [batch_size]

        return power

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

        # Sifter: Added magnetic sifter evaluation branch
        if self.func_name == 'magnetic_sifter':
            x_scaled = self.scale_sifter_variables(x)
            ms1, ms2 = self.scale_sifter_context(c)
            return self._contextual_magnetic_sifter(x_scaled, (ms1, ms2))

        # GridShell
        elif self.func_name == 'gridshell':
            x_scaled = self.scale_gridshell_variables(x)
            house_theta = self.scale_gridshell_context(c)
            return self._contextual_gridshell(x_scaled, house_theta)

        # turbine: Added turbine evaluation branch
        elif self.func_name == 'turbine':
            x_scaled = self.scale_turbine_variables(x)
            wind_speed = self.scale_turbine_context(c)  # Returns 1D wind speed tensor
            return self._contextual_turbine(x_scaled, wind_speed)

        # bicopter: Added bicopter evaluation branch
        elif self.func_name == 'bicopter':
            return self._contextual_bicopter(x, c)

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

    def _contextual_gridshell(self, x, c):
        """
        Gridshell evaluation with morning and evening power objectives

        Args:
            x: [batch_size, (numX-2)*(numY-2)] scaled interior grid heights
            c: [batch_size] scaled building orientation angles

        Returns:
            objectives: [batch_size, 2] - [morning_power, evening_power]
        """
        batch_size = x.shape[0]
        numX, numY = self.numX, self.numY

        # Reconstruct full grid from interior variables
        pts_grid = self._reconstruct_grid(x, numX, numY)  # [batch_size, numX, numY]

        # Calculate smoothness penalty
        smoothness = self._gridshell_smoothness(pts_grid)  # [batch_size]

        # Calculate morning power output
        morning_power = self._gridshell_power_output(
            pts_grid, self.morning_sun, c)  # [batch_size]

        # Calculate evening power output
        evening_power = self._gridshell_power_output(
            pts_grid, self.evening_sun, c)  # [batch_size]

        # Combine power and smoothness with regularization
        # For minimization objectives: minimize negative power + smoothness penalty
        f1 = -morning_power + self.regularizer_weight * smoothness  # Minimize negative morning power
        min_output_1, max_output_1 = -0.85, -0.40
        f1 = (f1 - min_output_1) / (max_output_1 - min_output_1)
        f2 = -evening_power + self.regularizer_weight * smoothness  # Minimize negative evening power
        min_output_2, max_output_2 = -0.85, -0.40
        f2 = (f2 - min_output_2) / (max_output_2 - min_output_2)

        return torch.stack([f1, f2], dim=1)


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


def visualize_bicopter_landscape():
    """
    Visualize the landscape of bicopter multi-objective function
    Step 1: Initialize solutions in a large pool
    Step 2: Evaluate the solutions
    Step 3: Visualize the solutions and analyze context sensitivity
    """

    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bicopter Multi-Objective Optimization Landscape', fontsize=16)

    # Test parameters
    n_samples = 10000  # Bicopter DW: Large pool of solutions for comprehensive visualization

    print("Visualizing Bicopter Problem Landscape...")

    # =====================================================
    # Step 1: Initialize bicopter problem
    # =====================================================
    bicopter_problem = ContextualMultiObjectiveFunction(func_name='bicopter')

    # =====================================================
    # Step 2: Generate and evaluate solutions
    # =====================================================

    # Bicopter DW: Input format [12 control vars, dummy_context, length, density]
    # All values in [0,1] will be scaled appropriately by the class
    X_bicopter = torch.rand(n_samples, bicopter_problem.n_variables + bicopter_problem.context_dim)

    # Step 3: Evaluate bicopter solutions
    Y_bicopter = bicopter_problem.evaluate(X_bicopter)

    print(f"Bicopter Problem Analysis:")
    print(f"Input shape: {X_bicopter.shape}")
    print(f"Output shape: {Y_bicopter.shape}")
    print(f"Distance objective range: [{Y_bicopter[:, 0].min():.3f}, {Y_bicopter[:, 0].max():.3f}]")
    print(f"Energy objective range: [{Y_bicopter[:, 1].min():.3f}, {Y_bicopter[:, 1].max():.3f}]")

    # Extract control variables and context for analysis
    control_mean = X_bicopter[:, :12].mean(dim=1).numpy()  # Bicopter DW: Average control intensity
    control_std = X_bicopter[:, :12].std(dim=1).numpy()  # Bicopter DW: Control variation
    length_norm = X_bicopter[:, 13].numpy()  # Bicopter DW: Normalized length [0,1] (skip dummy at index 12)
    density_norm = X_bicopter[:, 14].numpy()  # Bicopter DW: Normalized density [0,1]

    distance_obj = Y_bicopter[:, 0].numpy()  # Bicopter DW: Distance to goal objective (minimize)
    energy_obj = Y_bicopter[:, 1].numpy()  # Bicopter DW: Energy consumption objective (minimize)

    # =====================================================
    # Plot 1: Main Objective Space (Distance vs Energy)
    # =====================================================
    # Bicopter DW: Color by length to show context dependency
    scatter1 = axes[0, 0].scatter(distance_obj, energy_obj, c=length_norm,
                                  cmap='viridis', alpha=0.7, s=30)
    axes[0, 0].set_title('Objective Space: Distance vs Energy\n(colored by length)', fontsize=12)
    axes[0, 0].set_xlabel('Distance to Goal (f1) - Minimize')
    axes[0, 0].set_ylabel('Energy Consumption (f2) - Minimize')
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Length (normalized [0,1])')
    axes[0, 0].grid(True, alpha=0.3)

    # =====================================================
    # Plot 2: Context Space - Length vs Density
    # =====================================================
    scatter2 = axes[0, 1].scatter(length_norm, density_norm, c=distance_obj,
                                  cmap='plasma', alpha=0.7, s=30)
    axes[0, 1].set_title('Context Space: Length vs Density\n(colored by distance objective)', fontsize=12)
    axes[0, 1].set_xlabel('Length (normalized [0,1])')
    axes[0, 1].set_ylabel('Density (normalized [0,1])')
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('Distance Objective')
    axes[0, 1].grid(True, alpha=0.3)

    # =====================================================
    # Plot 3: Length Context Sensitivity
    # =====================================================
    scatter3 = axes[0, 2].scatter(length_norm, distance_obj, c=energy_obj,
                                  cmap='coolwarm', alpha=0.7, s=30)
    axes[0, 2].set_title('Length Sensitivity: Length vs Distance\n(colored by energy)', fontsize=12)
    axes[0, 2].set_xlabel('Length (normalized [0,1])')
    axes[0, 2].set_ylabel('Distance Objective (f1)')
    cbar3 = plt.colorbar(scatter3, ax=axes[0, 2])
    cbar3.set_label('Energy Objective')
    axes[0, 2].grid(True, alpha=0.3)

    # =====================================================
    # Plot 4: Density Context Sensitivity
    # =====================================================
    scatter4 = axes[1, 0].scatter(density_norm, energy_obj, c=distance_obj,
                                  cmap='RdYlBu', alpha=0.7, s=30)
    axes[1, 0].set_title('Density Sensitivity: Density vs Energy\n(colored by distance)', fontsize=12)
    axes[1, 0].set_xlabel('Density (normalized [0,1])')
    axes[1, 0].set_ylabel('Energy Objective (f2)')
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 0])
    cbar4.set_label('Distance Objective')
    axes[1, 0].grid(True, alpha=0.3)

    # =====================================================
    # Plot 5: Control Strategy Analysis
    # =====================================================
    # Bicopter DW: Show relationship between control characteristics and performance
    scatter5 = axes[1, 1].scatter(control_mean, control_std, c=energy_obj,
                                  cmap='spring', alpha=0.7, s=30)
    axes[1, 1].set_title('Control Strategy: Mean vs Variation\n(colored by energy)', fontsize=12)
    axes[1, 1].set_xlabel('Average Control Intensity')
    axes[1, 1].set_ylabel('Control Variation (Std Dev)')
    cbar5 = plt.colorbar(scatter5, ax=axes[1, 1])
    cbar5.set_label('Energy Objective')
    axes[1, 1].grid(True, alpha=0.3)

    # =====================================================
    # Plot 6: Pareto Front Analysis by Context
    # =====================================================
    # Bicopter DW: Identify approximate Pareto fronts for different length categories

    # Separate solutions by length bins
    length_bins = np.linspace(0, 1, 4)  # 3 length ranges
    colors = ['red', 'blue', 'green']

    for i in range(len(length_bins) - 1):
        mask = (length_norm >= length_bins[i]) & (length_norm < length_bins[i + 1])
        if np.sum(mask) > 0:
            axes[1, 2].scatter(distance_obj[mask], energy_obj[mask],
                               c=colors[i], alpha=0.6, s=20,
                               label=f'Length {length_bins[i]:.1f}-{length_bins[i + 1]:.1f}')

    axes[1, 2].set_title('Pareto Front by Length Ranges', fontsize=12)
    axes[1, 2].set_xlabel('Distance Objective (f1)')
    axes[1, 2].set_ylabel('Energy Objective (f2)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # Create Additional 3D Visualization
    # =====================================================
    fig_3d = plt.figure(figsize=(15, 5))

    # 3D Plot 1: Context-Objective Relationship
    ax1 = fig_3d.add_subplot(131, projection='3d')
    scatter_3d1 = ax1.scatter(length_norm, density_norm, distance_obj,
                              c=energy_obj, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Length (normalized)')
    ax1.set_ylabel('Density (normalized)')
    ax1.set_zlabel('Distance Objective')
    ax1.set_title('3D: Context vs Distance\n(colored by energy)')
    fig_3d.colorbar(scatter_3d1, ax=ax1, shrink=0.5)

    # 3D Plot 2: Mass vs Context Relationship
    # Calculate mass from length and density for visualization
    length_phys = 0.5 + length_norm * 1.5  # Convert to [0.5, 2.0] m
    density_phys = 0.5 + density_norm * 1.5  # Convert to [0.5, 2.0] kg/m
    mass_calc = length_phys * density_phys

    ax2 = fig_3d.add_subplot(132, projection='3d')
    scatter_3d2 = ax2.scatter(length_norm, density_norm, mass_calc,
                              c=distance_obj, cmap='plasma', alpha=0.7)
    ax2.set_xlabel('Length (normalized)')
    ax2.set_ylabel('Density (normalized)')
    ax2.set_zlabel('Calculated Mass (kg)')
    ax2.set_title('3D: Context vs Mass\n(colored by distance)')
    fig_3d.colorbar(scatter_3d2, ax=ax2, shrink=0.5)

    # 3D Plot 3: Control-Objective Relationship
    ax3 = fig_3d.add_subplot(133, projection='3d')
    scatter_3d3 = ax3.scatter(control_mean, control_std, energy_obj,
                              c=distance_obj, cmap='coolwarm', alpha=0.7)
    ax3.set_xlabel('Control Mean')
    ax3.set_ylabel('Control Std')
    ax3.set_zlabel('Energy Objective')
    ax3.set_title('3D: Control vs Energy\n(colored by distance)')
    fig_3d.colorbar(scatter_3d3, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # Print Summary Statistics
    # =====================================================
    print("\n" + "=" * 50)
    print("BICOPTER LANDSCAPE SUMMARY STATISTICS")
    print("=" * 50)

    # Convert normalized values to physical units for interpretation
    length_physical = 0.5 + length_norm * 1.5  # [0.5, 2.0] m
    density_physical = 0.5 + density_norm * 1.5  # [0.5, 2.0] kg/m
    mass_physical = length_physical * density_physical  # Calculated mass

    print(f"Context Variable Ranges (Physical Units):")
    print(f"  Length: {length_physical.min():.2f} - {length_physical.max():.2f} m")
    print(f"  Density: {density_physical.min():.2f} - {density_physical.max():.2f} kg/m")
    print(f"  Calculated Mass: {mass_physical.min():.2f} - {mass_physical.max():.2f} kg")

    print(f"\nControl Statistics:")
    print(f"  Control Mean: {control_mean.min():.3f} - {control_mean.max():.3f}")
    print(f"  Control Std: {control_std.min():.3f} - {control_std.max():.3f}")

    print(f"\nObjective Statistics:")
    print(f"  Distance Objective (f1): {distance_obj.min():.3f} - {distance_obj.max():.3f}")
    print(f"  Energy Objective (f2): {energy_obj.min():.3f} - {energy_obj.max():.3f}")

    # Find best and worst solutions
    best_distance_idx = np.argmin(distance_obj)
    best_energy_idx = np.argmin(energy_obj)

    print(f"\nBest Distance Solution:")
    print(f"  Distance: {distance_obj[best_distance_idx]:.3f}, Energy: {energy_obj[best_distance_idx]:.3f}")
    print(
        f"  Length: {length_physical[best_distance_idx]:.2f}m, Density: {density_physical[best_distance_idx]:.2f}kg/m")
    print(f"  Mass: {mass_physical[best_distance_idx]:.2f}kg, Control Mean: {control_mean[best_distance_idx]:.3f}")

    print(f"\nBest Energy Solution:")
    print(f"  Distance: {distance_obj[best_energy_idx]:.3f}, Energy: {energy_obj[best_energy_idx]:.3f}")
    print(f"  Length: {length_physical[best_energy_idx]:.2f}m, Density: {density_physical[best_energy_idx]:.2f}kg/m")
    print(f"  Mass: {mass_physical[best_energy_idx]:.2f}kg, Control Mean: {control_mean[best_energy_idx]:.3f}")

    # =====================================================
    # Context Sensitivity Analysis
    # =====================================================
    print("\n" + "=" * 50)
    print("CONTEXT SENSITIVITY ANALYSIS")
    print("=" * 50)

    # Analyze correlation between context and objectives
    from scipy.stats import pearsonr

    length_distance_corr, _ = pearsonr(length_norm, distance_obj)
    length_energy_corr, _ = pearsonr(length_norm, energy_obj)
    density_distance_corr, _ = pearsonr(density_norm, distance_obj)
    density_energy_corr, _ = pearsonr(density_norm, energy_obj)
    mass_distance_corr, _ = pearsonr(mass_physical, distance_obj)
    mass_energy_corr, _ = pearsonr(mass_physical, energy_obj)

    print(f"Correlation Analysis:")
    print(f"  Length vs Distance: {length_distance_corr:.3f}")
    print(f"  Length vs Energy: {length_energy_corr:.3f}")
    print(f"  Density vs Distance: {density_distance_corr:.3f}")
    print(f"  Density vs Energy: {density_energy_corr:.3f}")
    print(f"  Mass vs Distance: {mass_distance_corr:.3f}")
    print(f"  Mass vs Energy: {mass_energy_corr:.3f}")

    # Identify context regions for best performance
    # Low distance (good tracking)
    low_distance_mask = distance_obj < np.percentile(distance_obj, 25)
    print(f"\nBest Distance Performance Regions:")
    print(
        f"  Preferred Length: {length_norm[low_distance_mask].mean():.3f} ± {length_norm[low_distance_mask].std():.3f}")
    print(
        f"  Preferred Density: {density_norm[low_distance_mask].mean():.3f} ± {density_norm[low_distance_mask].std():.3f}")

    # Low energy (efficient)
    low_energy_mask = energy_obj < np.percentile(energy_obj, 25)
    print(f"\nBest Energy Performance Regions:")
    print(f"  Preferred Length: {length_norm[low_energy_mask].mean():.3f} ± {length_norm[low_energy_mask].std():.3f}")
    print(
        f"  Preferred Density: {density_norm[low_energy_mask].mean():.3f} ± {density_norm[low_energy_mask].std():.3f}")

    return fig, axes, fig_3d, X_bicopter, Y_bicopter


# # Bicopter DW: Example usage and testing
# if __name__ == "__main__":
#     print("Starting Bicopter Landscape Visualization...")
#     try:
#         # Note: Make sure ContextualMultiObjectiveFunction with bicopter is imported
#         fig_main, axes_main, fig_3d, X_data, Y_data = visualize_bicopter_landscape()
#         print("\nVisualization completed successfully!")
#         print(f"Generated data shapes: X={X_data.shape}, Y={Y_data.shape}")
#     except Exception as e:
#         print(f"Error during visualization: {e}")
#         print("Make sure the ContextualMultiObjectiveFunction class with bicopter support is available.")

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr


def visualize_magnetic_sifter_landscape_simplified():
    """
    Simplified visualization of magnetic sifter multi-objective function
    Focus on objective space colored by context variables for clear analysis
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Set up the figure with cleaner layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Magnetic Sifter Multi-Objective Optimization: Objective Space Analysis',
                 fontsize=16, fontweight='bold')

    # Test parameters
    n_samples = 100  # 100 random solutions as requested

    print("Generating 100 random solutions for Magnetic Sifter Problem...")

    # =====================================================
    # Step 1: Initialize magnetic sifter problem
    # =====================================================
    sifter_problem = ContextualMultiObjectiveFunction(func_name='magnetic_sifter')

    print(f"Problem Configuration:")
    print(f"  Objectives: {sifter_problem.n_objectives} (f1: unwanted capture, f2: separation, f3: thickness)")
    print(f"  Variables: {sifter_problem.n_variables} (gap, mag_len, thickness)")
    print(f"  Context dim: {sifter_problem.context_dim} (dummy, ms1, ms2)")

    # =====================================================
    # Step 2: Generate random solutions
    # =====================================================
    # Random sampling: [3 design vars + 3 context vars] all in [0,1]
    X_sifter = torch.rand(n_samples, sifter_problem.n_variables + sifter_problem.context_dim)

    print("Evaluating solutions...")
    # Evaluate all solutions
    Y_sifter = sifter_problem.evaluate(X_sifter)

    print(f"Evaluation complete!")
    print(f"Input shape: {X_sifter.shape}")
    print(f"Output shape: {Y_sifter.shape}")

    # Extract variables for plotting
    ms1_norm = X_sifter[:, 4].numpy()  # ms1 context (skip dummy at index 3)
    ms2_norm = X_sifter[:, 5].numpy()  # ms2 context

    f1 = Y_sifter[:, 0].numpy()  # Unwanted capture (minimize)
    f2 = Y_sifter[:, 1].numpy()  # Separation (minimize)
    f3 = Y_sifter[:, 2].numpy()  # Thickness (minimize)

    print(f"Objective Ranges:")
    print(f"  f1 (unwanted): [{f1.min():.4f}, {f1.max():.4f}]")
    print(f"  f2 (separation): [{f2.min():.4f}, {f2.max():.4f}]")
    print(f"  f3 (thickness): [{f3.min():.4f}, {f3.max():.4f}]")

    # =====================================================
    # Plot 1: f1 vs f2 colored by ms1
    # =====================================================
    scatter1 = axes[0, 0].scatter(f1, f2, c=ms1_norm, cmap='viridis',
                                  alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    axes[0, 0].set_title('Objective Space: f1 vs f2\nColored by ms1 (Cell Type 1 Magnetic Moment)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('f1: Unwanted Capture (minimize)', fontweight='bold')
    axes[0, 0].set_ylabel('f2: Separation (minimize)', fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('ms1 (normalized)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # =====================================================
    # Plot 2: f1 vs f2 colored by ms2
    # =====================================================
    scatter2 = axes[0, 1].scatter(f1, f2, c=ms2_norm, cmap='plasma',
                                  alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    axes[0, 1].set_title('Objective Space: f1 vs f2\nColored by ms2 (Cell Type 2 Magnetic Moment)',
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('f1: Unwanted Capture (minimize)', fontweight='bold')
    axes[0, 1].set_ylabel('f2: Separation (minimize)', fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('ms2 (normalized)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # =====================================================
    # Plot 3: f1 vs f3 colored by ms1
    # =====================================================
    scatter3 = axes[1, 0].scatter(f1, f3, c=ms1_norm, cmap='coolwarm',
                                  alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    axes[1, 0].set_title('Objective Space: f1 vs f3\nColored by ms1',
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('f1: Unwanted Capture (minimize)', fontweight='bold')
    axes[1, 0].set_ylabel('f3: Thickness (minimize)', fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=axes[1, 0])
    cbar3.set_label('ms1 (normalized)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # =====================================================
    # Plot 4: f2 vs f3 colored by ms2
    # =====================================================
    scatter4 = axes[1, 1].scatter(f2, f3, c=ms2_norm, cmap='spring',
                                  alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('Objective Space: f2 vs f3\nColored by ms2',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('f2: Separation (minimize)', fontweight='bold')
    axes[1, 1].set_ylabel('f3: Thickness (minimize)', fontweight='bold')
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 1])
    cbar4.set_label('ms2 (normalized)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # 3D Objective Space Visualization
    # =====================================================
    fig_3d = plt.figure(figsize=(16, 6))

    # 3D Plot 1: All objectives colored by ms1
    ax1 = fig_3d.add_subplot(121, projection='3d')
    scatter_3d1 = ax1.scatter(f1, f2, f3, c=ms1_norm, cmap='viridis',
                              alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('f1: Unwanted Capture', fontweight='bold')
    ax1.set_ylabel('f2: Separation', fontweight='bold')
    ax1.set_zlabel('f3: Thickness', fontweight='bold')
    ax1.set_title('3D Objective Space\nColored by ms1', fontsize=12, fontweight='bold')
    cbar_3d1 = fig_3d.colorbar(scatter_3d1, ax=ax1, shrink=0.5)
    cbar_3d1.set_label('ms1 (normalized)', fontweight='bold')

    # 3D Plot 2: All objectives colored by ms2
    ax2 = fig_3d.add_subplot(122, projection='3d')
    scatter_3d2 = ax2.scatter(f1, f2, f3, c=ms2_norm, cmap='plasma',
                              alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('f1: Unwanted Capture', fontweight='bold')
    ax2.set_ylabel('f2: Separation', fontweight='bold')
    ax2.set_zlabel('f3: Thickness', fontweight='bold')
    ax2.set_title('3D Objective Space\nColored by ms2', fontsize=12, fontweight='bold')
    cbar_3d2 = fig_3d.colorbar(scatter_3d2, ax=ax2, shrink=0.5)
    cbar_3d2.set_label('ms2 (normalized)', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # =====================================================
    # Analysis Summary
    # =====================================================
    print("\n" + "=" * 60)
    print("CONTEXT SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Convert to physical units for better interpretation
    ms1_physical = (1.5 + ms1_norm * 0.5) * 1e-13  # [1.5, 2.0] × 10^-13
    ms2_physical = (0.5 + ms2_norm * 0.3) * 1e-13  # [0.5, 0.8] × 10^-13

    print(f"Context Variable Ranges (Physical Units):")
    print(f"  ms1: {ms1_physical.min():.2e} - {ms1_physical.max():.2e}")
    print(f"  ms2: {ms2_physical.min():.2e} - {ms2_physical.max():.2e}")

    # Correlation analysis
    from scipy.stats import pearsonr

    ms1_f1_corr, _ = pearsonr(ms1_norm, f1)
    ms1_f2_corr, _ = pearsonr(ms1_norm, f2)
    ms1_f3_corr, _ = pearsonr(ms1_norm, f3)
    ms2_f1_corr, _ = pearsonr(ms2_norm, f1)
    ms2_f2_corr, _ = pearsonr(ms2_norm, f2)
    ms2_f3_corr, _ = pearsonr(ms2_norm, f3)

    print(f"\nCorrelation Analysis:")
    print(f"  ms1 vs f1 (unwanted): {ms1_f1_corr:+.3f}")
    print(f"  ms1 vs f2 (separation): {ms1_f2_corr:+.3f}")
    print(f"  ms1 vs f3 (thickness): {ms1_f3_corr:+.3f}")
    print(f"  ms2 vs f1 (unwanted): {ms2_f1_corr:+.3f}")
    print(f"  ms2 vs f2 (separation): {ms2_f2_corr:+.3f}")
    print(f"  ms2 vs f3 (thickness): {ms2_f3_corr:+.3f}")

    # Find best solutions
    best_f1_idx = np.argmin(f1)
    best_f2_idx = np.argmin(f2)
    best_f3_idx = np.argmin(f3)

    print(f"\nBest Solutions:")
    print(f"  Best Unwanted Capture (f1): {f1[best_f1_idx]:.4f}")
    print(f"    ms1: {ms1_physical[best_f1_idx]:.2e}, ms2: {ms2_physical[best_f1_idx]:.2e}")
    print(f"  Best Separation (f2): {f2[best_f2_idx]:.4f}")
    print(f"    ms1: {ms1_physical[best_f2_idx]:.2e}, ms2: {ms2_physical[best_f2_idx]:.2e}")
    print(f"  Best Thickness (f3): {f3[best_f3_idx]:.4f}")
    print(f"    ms1: {ms1_physical[best_f3_idx]:.2e}, ms2: {ms2_physical[best_f3_idx]:.2e}")

    # Context preferences for good performance
    print(f"\nContext Preferences (best 25% solutions):")

    best_f1_mask = f1 <= np.percentile(f1, 25)
    best_f2_mask = f2 <= np.percentile(f2, 25)

    print(f"  For low unwanted capture:")
    print(f"    Preferred ms1: {ms1_norm[best_f1_mask].mean():.3f} ± {ms1_norm[best_f1_mask].std():.3f}")
    print(f"    Preferred ms2: {ms2_norm[best_f1_mask].mean():.3f} ± {ms2_norm[best_f1_mask].std():.3f}")

    print(f"  For good separation:")
    print(f"    Preferred ms1: {ms1_norm[best_f2_mask].mean():.3f} ± {ms1_norm[best_f2_mask].std():.3f}")
    print(f"    Preferred ms2: {ms2_norm[best_f2_mask].mean():.3f} ± {ms2_norm[best_f2_mask].std():.3f}")

    return fig, fig_3d, X_sifter, Y_sifter


import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr


def visualize_gridshell_landscape():
    """
    Visualize the landscape of gridshell multi-objective function
    Step 1: Initialize solutions in a large pool
    Step 2: Evaluate the solutions
    Step 3: Visualize the solutions and analyze context sensitivity
    """

    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Gridshell Solar Building Optimization Landscape', fontsize=16)

    # Test parameters
    n_samples = 10000  # Gridshell: Large pool of solutions for comprehensive visualization

    print("Visualizing Gridshell Problem Landscape...")

    # =====================================================
    # Step 1: Initialize gridshell problem
    # =====================================================
    gridshell_problem = ContextualMultiObjectiveFunction(func_name='gridshell')

    # =====================================================
    # Step 2: Generate and evaluate solutions
    # =====================================================

    # Gridshell: Input format [9 interior heights, dummy_context, building_orientation]
    # All values in [0,1] will be scaled appropriately by the class
    X_gridshell = torch.rand(n_samples, gridshell_problem.n_variables + gridshell_problem.context_dim)

    # Step 3: Evaluate gridshell solutions
    Y_gridshell = gridshell_problem.evaluate(X_gridshell)

    print(f"Gridshell Problem Analysis:")
    print(f"Input shape: {X_gridshell.shape}")
    print(f"Output shape: {Y_gridshell.shape}")
    print(f"Morning power objective range: [{Y_gridshell[:, 0].min():.6f}, {Y_gridshell[:, 0].max():.6f}]")
    print(f"Evening power objective range: [{Y_gridshell[:, 1].min():.6f}, {Y_gridshell[:, 1].max():.6f}]")

    # Extract design variables and context for analysis
    heights_mean = X_gridshell[:, :9].mean(dim=1).numpy()  # Gridshell: Average interior height
    heights_std = X_gridshell[:, :9].std(dim=1).numpy()  # Gridshell: Height variation (roughness)
    heights_max = X_gridshell[:, :9].max(dim=1)[0].numpy()  # Gridshell: Maximum height
    heights_min = X_gridshell[:, :9].min(dim=1)[0].numpy()  # Gridshell: Minimum height
    orientation_norm = X_gridshell[:, 10].numpy()  # Gridshell: Normalized orientation [0,1] (skip dummy at index 9)

    morning_obj = Y_gridshell[:, 0].numpy()  # Gridshell: Morning power objective (minimize negative power)
    evening_obj = Y_gridshell[:, 1].numpy()  # Gridshell: Evening power objective (minimize negative power)

    # =====================================================
    # Plot 1: Main Objective Space (Morning vs Evening Power)
    # =====================================================
    # Gridshell: Color by building orientation to show context dependency
    scatter1 = axes[0, 0].scatter(morning_obj, evening_obj, c=orientation_norm,
                                  cmap='viridis', alpha=0.7, s=30)
    axes[0, 0].set_title('Objective Space: Morning vs Evening Power\n(colored by building orientation)', fontsize=12)
    axes[0, 0].set_xlabel('Morning Power Objective (f1) - Minimize')
    axes[0, 0].set_ylabel('Evening Power Objective (f2) - Minimize')
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Building Orientation (normalized [0,1])')
    axes[0, 0].grid(True, alpha=0.3)

    # =====================================================
    # Plot 2: Surface Characteristics - Height Mean vs Variation
    # =====================================================
    scatter2 = axes[0, 1].scatter(heights_mean, heights_std, c=morning_obj,
                                  cmap='plasma', alpha=0.7, s=30)
    axes[0, 1].set_title('Surface Characteristics: Mean vs Variation\n(colored by morning power)', fontsize=12)
    axes[0, 1].set_xlabel('Average Interior Height')
    axes[0, 1].set_ylabel('Height Variation (Std Dev)')
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('Morning Power Objective')
    axes[0, 1].grid(True, alpha=0.3)

    # =====================================================
    # Plot 3: Orientation Sensitivity
    # =====================================================
    scatter3 = axes[0, 2].scatter(orientation_norm, morning_obj, c=evening_obj,
                                  cmap='coolwarm', alpha=0.7, s=30)
    axes[0, 2].set_title('Orientation Sensitivity: Angle vs Morning Power\n(colored by evening power)', fontsize=12)
    axes[0, 2].set_xlabel('Building Orientation (normalized [0,1])')
    axes[0, 2].set_ylabel('Morning Power Objective (f1)')
    cbar3 = plt.colorbar(scatter3, ax=axes[0, 2])
    cbar3.set_label('Evening Power Objective')
    axes[0, 2].grid(True, alpha=0.3)

    # =====================================================
    # Plot 4: Height Range vs Performance
    # =====================================================
    height_range = heights_max - heights_min
    scatter4 = axes[1, 0].scatter(height_range, evening_obj, c=orientation_norm,
                                  cmap='RdYlBu', alpha=0.7, s=30)
    axes[1, 0].set_title('Height Range vs Evening Power\n(colored by orientation)', fontsize=12)
    axes[1, 0].set_xlabel('Height Range (max - min)')
    axes[1, 0].set_ylabel('Evening Power Objective (f2)')
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 0])
    cbar4.set_label('Building Orientation')
    axes[1, 0].grid(True, alpha=0.3)

    # =====================================================
    # Plot 5: Surface Smoothness Analysis
    # =====================================================
    # Gridshell: Show relationship between surface characteristics and performance
    scatter5 = axes[1, 1].scatter(heights_mean, height_range, c=(morning_obj + evening_obj),
                                  cmap='spring', alpha=0.7, s=30)
    axes[1, 1].set_title('Surface Design Space: Mean vs Range\n(colored by total power)', fontsize=12)
    axes[1, 1].set_xlabel('Average Interior Height')
    axes[1, 1].set_ylabel('Height Range')
    cbar5 = plt.colorbar(scatter5, ax=axes[1, 1])
    cbar5.set_label('Total Power Objective (f1+f2)')
    axes[1, 1].grid(True, alpha=0.3)

    # =====================================================
    # Plot 6: Pareto Front Analysis by Orientation
    # =====================================================
    # Gridshell: Identify approximate Pareto fronts for different orientation categories

    # Separate solutions by orientation bins
    orientation_bins = np.linspace(0, 1, 4)  # 3 orientation ranges
    colors = ['red', 'blue', 'green']

    for i in range(len(orientation_bins) - 1):
        mask = (orientation_norm >= orientation_bins[i]) & (orientation_norm < orientation_bins[i + 1])
        if np.sum(mask) > 0:
            axes[1, 2].scatter(morning_obj[mask], evening_obj[mask],
                               c=colors[i], alpha=0.6, s=20,
                               label=f'Orientation {orientation_bins[i]:.1f}-{orientation_bins[i + 1]:.1f}')

    axes[1, 2].set_title('Pareto Front by Orientation Ranges', fontsize=12)
    axes[1, 2].set_xlabel('Morning Power Objective (f1)')
    axes[1, 2].set_ylabel('Evening Power Objective (f2)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # Create Additional 3D Visualization
    # =====================================================
    fig_3d = plt.figure(figsize=(15, 5))

    # 3D Plot 1: Surface-Objective Relationship
    ax1 = fig_3d.add_subplot(131, projection='3d')
    scatter_3d1 = ax1.scatter(heights_mean, heights_std, morning_obj,
                              c=evening_obj, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Average Height')
    ax1.set_ylabel('Height Variation')
    ax1.set_zlabel('Morning Power Objective')
    ax1.set_title('3D: Surface vs Morning Power\n(colored by evening power)')
    fig_3d.colorbar(scatter_3d1, ax=ax1, shrink=0.5)

    # 3D Plot 2: Orientation vs Surface Relationship
    # Convert orientation to physical angle for visualization
    orientation_phys = orientation_norm * 2 * np.pi  # Convert to [0, 2π] radians
    orientation_degrees = orientation_phys * 180 / np.pi  # Convert to degrees

    ax2 = fig_3d.add_subplot(132, projection='3d')
    scatter_3d2 = ax2.scatter(orientation_degrees, heights_mean, height_range,
                              c=morning_obj, cmap='plasma', alpha=0.7)
    ax2.set_xlabel('Building Orientation (degrees)')
    ax2.set_ylabel('Average Height')
    ax2.set_zlabel('Height Range')
    ax2.set_title('3D: Orientation vs Surface\n(colored by morning power)')
    fig_3d.colorbar(scatter_3d2, ax=ax2, shrink=0.5)

    # 3D Plot 3: Complete Design Space
    ax3 = fig_3d.add_subplot(133, projection='3d')
    scatter_3d3 = ax3.scatter(morning_obj, evening_obj, orientation_degrees,
                              c=heights_mean, cmap='coolwarm', alpha=0.7)
    ax3.set_xlabel('Morning Power Objective')
    ax3.set_ylabel('Evening Power Objective')
    ax3.set_zlabel('Building Orientation (degrees)')
    ax3.set_title('3D: Objectives vs Orientation\n(colored by avg height)')
    fig_3d.colorbar(scatter_3d3, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # Print Summary Statistics
    # =====================================================
    print("\n" + "=" * 50)
    print("GRIDSHELL LANDSCAPE SUMMARY STATISTICS")
    print("=" * 50)

    # Convert normalized values to physical units for interpretation
    heights_physical = heights_mean * 5.0  # Convert to [0, 5] units (physical height)
    orientation_physical = orientation_norm * 360.0  # Convert to [0, 360] degrees

    print(f"Design Variable Ranges (Physical Units):")
    print(f"  Average Interior Height: {heights_physical.min():.2f} - {heights_physical.max():.2f} units")
    print(f"  Height Variation (Std): {heights_std.min():.3f} - {heights_std.max():.3f}")
    print(f"  Height Range: {height_range.min():.3f} - {height_range.max():.3f}")

    print(f"Context Variable Ranges (Physical Units):")
    print(f"  Building Orientation: {orientation_physical.min():.1f} - {orientation_physical.max():.1f} degrees")

    print(f"\nObjective Statistics:")
    print(f"  Morning Power Objective (f1): {morning_obj.min():.6f} - {morning_obj.max():.6f}")
    print(f"  Evening Power Objective (f2): {evening_obj.min():.6f} - {evening_obj.max():.6f}")
    print(
        f"  Objective Span: f1 = {morning_obj.max() - morning_obj.min():.6f}, f2 = {evening_obj.max() - evening_obj.min():.6f}")

    # Find best and worst solutions
    best_morning_idx = np.argmin(morning_obj)
    best_evening_idx = np.argmin(evening_obj)
    best_balanced_idx = np.argmin(morning_obj + evening_obj)

    print(f"\nBest Morning Power Solution:")
    print(f"  Morning: {morning_obj[best_morning_idx]:.6f}, Evening: {evening_obj[best_morning_idx]:.6f}")
    print(f"  Avg Height: {heights_physical[best_morning_idx]:.2f}, Height Std: {heights_std[best_morning_idx]:.3f}")
    print(f"  Orientation: {orientation_physical[best_morning_idx]:.1f}°")

    print(f"\nBest Evening Power Solution:")
    print(f"  Morning: {morning_obj[best_evening_idx]:.6f}, Evening: {evening_obj[best_evening_idx]:.6f}")
    print(f"  Avg Height: {heights_physical[best_evening_idx]:.2f}, Height Std: {heights_std[best_evening_idx]:.3f}")
    print(f"  Orientation: {orientation_physical[best_evening_idx]:.1f}°")

    print(f"\nBest Balanced Solution:")
    print(f"  Morning: {morning_obj[best_balanced_idx]:.6f}, Evening: {evening_obj[best_balanced_idx]:.6f}")
    print(f"  Avg Height: {heights_physical[best_balanced_idx]:.2f}, Height Std: {heights_std[best_balanced_idx]:.3f}")
    print(f"  Orientation: {orientation_physical[best_balanced_idx]:.1f}°")

    # =====================================================
    # Context Sensitivity Analysis
    # =====================================================
    print("\n" + "=" * 50)
    print("CONTEXT SENSITIVITY ANALYSIS")
    print("=" * 50)

    # Analyze correlation between context/design and objectives
    orientation_morning_corr, _ = pearsonr(orientation_norm, morning_obj)
    orientation_evening_corr, _ = pearsonr(orientation_norm, evening_obj)
    height_mean_morning_corr, _ = pearsonr(heights_mean, morning_obj)
    height_mean_evening_corr, _ = pearsonr(heights_mean, evening_obj)
    height_std_morning_corr, _ = pearsonr(heights_std, morning_obj)
    height_std_evening_corr, _ = pearsonr(heights_std, evening_obj)
    height_range_morning_corr, _ = pearsonr(height_range, morning_obj)
    height_range_evening_corr, _ = pearsonr(height_range, evening_obj)

    print(f"Correlation Analysis:")
    print(f"  Orientation vs Morning Power: {orientation_morning_corr:.3f}")
    print(f"  Orientation vs Evening Power: {orientation_evening_corr:.3f}")
    print(f"  Average Height vs Morning Power: {height_mean_morning_corr:.3f}")
    print(f"  Average Height vs Evening Power: {height_mean_evening_corr:.3f}")
    print(f"  Height Variation vs Morning Power: {height_std_morning_corr:.3f}")
    print(f"  Height Variation vs Evening Power: {height_std_evening_corr:.3f}")
    print(f"  Height Range vs Morning Power: {height_range_morning_corr:.3f}")
    print(f"  Height Range vs Evening Power: {height_range_evening_corr:.3f}")

    # Identify design regions for best performance
    # Low morning power (good morning performance)
    low_morning_mask = morning_obj < np.percentile(morning_obj, 25)
    print(f"\nBest Morning Power Performance Regions:")
    print(
        f"  Preferred Avg Height: {heights_mean[low_morning_mask].mean():.3f} ± {heights_mean[low_morning_mask].std():.3f}")
    print(
        f"  Preferred Height Std: {heights_std[low_morning_mask].mean():.3f} ± {heights_std[low_morning_mask].std():.3f}")
    print(
        f"  Preferred Orientation: {orientation_norm[low_morning_mask].mean():.3f} ± {orientation_norm[low_morning_mask].std():.3f}")

    # Low evening power (good evening performance)
    low_evening_mask = evening_obj < np.percentile(evening_obj, 25)
    print(f"\nBest Evening Power Performance Regions:")
    print(
        f"  Preferred Avg Height: {heights_mean[low_evening_mask].mean():.3f} ± {heights_mean[low_evening_mask].std():.3f}")
    print(
        f"  Preferred Height Std: {heights_std[low_evening_mask].mean():.3f} ± {heights_std[low_evening_mask].std():.3f}")
    print(
        f"  Preferred Orientation: {orientation_norm[low_evening_mask].mean():.3f} ± {orientation_norm[low_evening_mask].std():.3f}")

    # Analyze trade-offs
    objective_correlation, _ = pearsonr(morning_obj, evening_obj)
    print(f"\nObjective Trade-off Analysis:")
    print(f"  Morning vs Evening Power Correlation: {objective_correlation:.3f}")
    if objective_correlation > 0.5:
        print("  → Strong positive correlation: Objectives are conflicting")
    elif objective_correlation < -0.5:
        print("  → Strong negative correlation: Objectives are aligned")
    else:
        print("  → Weak correlation: Complex trade-off structure")

    # Empirical Nadir Point Determination
    nadir_point = [morning_obj.max(), evening_obj.max()]
    print(f"\nEmpirical Nadir Point:")
    print(f"  Nadir: [{nadir_point[0]:.6f}, {nadir_point[1]:.6f}]")
    print("  (Worst-case values for minimization objectives)")

    return fig, axes, fig_3d, X_gridshell, Y_gridshell


# Example usage
if __name__ == "__main__":
    # Import your gridshell class here
    # from your_gridshell_module import ContextualMultiObjectiveFunction

    # Run the comprehensive landscape visualization
    fig, axes, fig_3d, X_samples, Y_objectives = visualize_gridshell_landscape()

    print("\n✅ Gridshell landscape analysis completed!")
    print("Generated comprehensive visualizations showing:")
    print("- Objective space structure (morning vs evening power)")
    print("- Surface design characteristics (height patterns)")
    print("- Building orientation sensitivity")
    print("- Pareto front analysis by context")
    print("- 3D design space relationships")
    print("- Statistical correlation analysis")