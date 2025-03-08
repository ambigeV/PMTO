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