import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict


class VAE(torch.nn.Module):
    """
    Variational Autoencoder for Pareto set/front modeling.
    Can be conditional on context variables.
    """

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, context_size=0):
        super().__init__()

        if conditional:
            assert context_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.conditional = conditional

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, context_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, context_size)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        """
        Generate new Pareto set solutions from latent vectors.

        Args:
            z: Latent vectors
            c: Context vectors (if conditional)

        Returns:
            Generated Pareto set solutions
        """
        recon_x = self.decoder(z, c)
        return recon_x


class Encoder(torch.nn.Module):
    """
    Encoder network for VAE.
    Maps input (X) to latent distribution parameters.
    """

    def __init__(self, layer_sizes, latent_size, conditional, context_size):
        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += context_size

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())

        self.linear_means = torch.nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = torch.nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional and c is not None:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(torch.nn.Module):
    """
    Decoder network for VAE.
    Maps latent vectors to reconstructed Pareto set solutions.
    """

    def __init__(self, layer_sizes, latent_size, conditional, context_size):
        super().__init__()

        self.MLP = torch.nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + context_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            else:
                pass
                # self.MLP.add_module(name="sigmoid", module=torch.nn.Sigmoid())
            # No activation in final layer - will be applied in forward method

    def forward(self, z, c=None):
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        # # Clamping output to [0,1] range
        x = torch.clamp(x, 0, 1)

        return x


class ParetoVAETrainer:
    """
    Trainer class for Pareto set/front VAE modeling.
    """

    def __init__(self,
                 input_dim,  # Dimension of decision variables (X)
                 output_dim=None,  # Dimension of objective values (Y) - unused in this version
                 latent_dim=2,  # Dimension of latent space
                 context_dim=0,  # Dimension of context variables
                 conditional=False,  # Whether to use conditional VAE
                 learning_rate=0.1,
                 batch_size=64,
                 epochs=100,
                 device=None,
                 trainer_id=None,
                 save_dir='./results',
                 # NEW: Aggressive training parameters
                 aggressive_training=True,
                 max_aggressive_epochs=5):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.conditional = conditional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir

        # NEW: Aggressive training settings
        self.aggressive_training = aggressive_training
        self.max_aggressive_epochs = max_aggressive_epochs
        self.current_aggressive_epoch = 0
        self.aggressive_phase = True

        # Determine device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create directory for saving results
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create timestamp for this run
        self.timestamp = trainer_id
        os.makedirs(os.path.join(save_dir, self.timestamp), exist_ok=True)

        # Create model architecture
        # Default architecture with reasonable layer sizes
        encoder_sizes = [input_dim, max(input_dim, 2 * latent_dim, input_dim)]
        decoder_sizes = [max(input_dim, 2 * latent_dim, input_dim), input_dim]

        self.model = VAE(
            encoder_layer_sizes=encoder_sizes,
            latent_size=latent_dim,
            decoder_layer_sizes=decoder_sizes,
            conditional=conditional,
            context_size=context_dim
        ).to(self.device)

        # MODIFIED: Create separate optimizers for encoder and decoder
        # MODIFIED: Create separate optimizers for encoder and decoder
        self.encoder_optimizer = torch.optim.Adam(
            self.model.encoder.parameters(),  # This includes MLP, linear_means, and linear_log_var
            lr=learning_rate
        )
        self.decoder_optimizer = torch.optim.Adam(
            self.model.decoder.parameters(),
            lr=learning_rate
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Schedulers for all optimizers
        self.encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.encoder_optimizer,
            milestones=[int(self.epochs * 0.5), int(self.epochs * 0.75)],
            gamma=0.1
        )
        self.decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.decoder_optimizer,
            milestones=[int(self.epochs * 0.5), int(self.epochs * 0.75)],
            gamma=0.1
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.epochs * 0.5), int(self.epochs * 0.75)],
            gamma=0.1
        )

        # Training logs
        self.logs = defaultdict(list)
        self._initialize_logs()

    def _initialize_logs(self):
        """Initialize all log containers to prevent KeyError"""
        # Standard training logs
        log_keys = [
            'loss', 'mse', 'kld', 'mutual_info',
            'grad_norm', 'max_grad', 'min_grad',
            # Aggressive training logs
            'aggressive_loss', 'aggressive_encoder_loss', 'aggressive_decoder_loss',
            'aggressive_mse', 'aggressive_kld', 'aggressive_epochs',
            # FIXED: Missing gradient logs
            'aggressive_encoder_grad_norm', 'aggressive_encoder_grad_max', 'aggressive_encoder_grad_min',
            'aggressive_decoder_grad_norm', 'aggressive_decoder_grad_max', 'aggressive_decoder_grad_min',
            # Standard training gradient logs (for consistency)
            'standard_grad_norm', 'standard_grad_max', 'standard_grad_min'
        ]

        for key in log_keys:
            self.logs[key] = []

    # New: To compute the lagging inference network (Encoder) to avoid
    # posterior collapse?
    def compute_mutual_information(self, data_loader):
        """
        Compute mutual information I_q = E_x[KL(q(z|x)||p(z))] - KL(q(z)||p(z))
        """
        self.model.eval()

        kl_term = 0
        all_means = []
        all_log_vars = []
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                if self.conditional:
                    x, c = batch
                    x, c = x.to(self.device), c.to(self.device)
                    _, mean, log_var, z = self.model(x, c)
                else:
                    x = batch[0].to(self.device)
                    _, mean, log_var, z = self.model(x)

                # E_x[KL(q(z|x)||p(z))] - this is the KL divergence in our loss
                batch_kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
                kl_term += torch.sum(batch_kl)
                total_samples += x.size(0)

                # Collect mean and log_var for aggregated posterior
                all_means.append(mean)
                all_log_vars.append(log_var)

        # Average KL divergence: E_x[KL(q(z|x)||p(z))]
        kl_term = kl_term / total_samples

        # Compute KL(q(z)||p(z)) where q(z) is the aggregated posterior
        # q(z) ≈ (1/N) Σ q(z|x_i) with mean = (1/N) Σ μ_i, var = (1/N) Σ (σ_i^2 + μ_i^2) - μ_agg^2
        all_means = torch.cat(all_means, dim=0)  # [N, latent_dim]
        all_log_vars = torch.cat(all_log_vars, dim=0)  # [N, latent_dim]
        all_vars = torch.exp(all_log_vars)  # Convert log_var to var

        # Aggregated posterior statistics
        mu_agg = torch.mean(all_means, dim=0)  # [latent_dim]
        var_agg = torch.mean(all_vars + all_means.pow(2), dim=0) - mu_agg.pow(2)  # [latent_dim]

        # KL(q(z)||p(z)) where p(z) = N(0,I)
        # KL(N(μ,Σ)||N(0,I)) = 0.5 * (tr(Σ) + μ^T μ - k - log|Σ|)
        # For diagonal covariance: KL = 0.5 * Σ(σ_i^2 + μ_i^2 - 1 - log(σ_i^2))
        aggregated_kl = 0.5 * torch.sum(var_agg + mu_agg.pow(2) - 1 - torch.log(var_agg + 1e-8))

        mutual_info = kl_term - aggregated_kl
        self.model.train()
        return mutual_info.item()

    def encoder_loss_fn(self, x, c=None):
        """Compute encoder-only loss (reconstruction + KL)"""
        if self.conditional and c is not None:
            recon_x, mean, log_var, z = self.model(x, c)
        else:
            recon_x, mean, log_var, z = self.model(x)

        # MSE reconstruction loss
        MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        beta = 1.0

        total_loss = (MSE + beta * KLD) / x.size(0)
        return total_loss, MSE / x.size(0), KLD / x.size(0)

    def decoder_loss_fn(self, x, c=None):
        """Compute decoder-only loss (reconstruction)"""
        if self.conditional and c is not None:
            recon_x, mean, log_var, z = self.model(x, c)
        else:
            recon_x, mean, log_var, z = self.model(x)

        # Only reconstruction loss for decoder
        MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        return MSE / x.size(0)

    def aggressive_encoder_training(self, data_loader, num_updates=5, epoch=0, callback=None):
        """Perform aggressive encoder training"""
        self.model.train()

        # Initialize metrics tracking
        total_encoder_loss = 0
        total_mse = 0
        total_kld = 0
        total_batches = 0

        for update in range(num_updates):
            for batch_idx, batch in enumerate(data_loader):
                if self.conditional:
                    x, c = batch
                    x, c = x.to(self.device), c.to(self.device)
                else:
                    x = batch[0].to(self.device)
                    c = None

                # IMPORTANT: Zero gradients for ALL optimizers
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                # Only update encoder parameters
                encoder_loss, mse, kld = self.encoder_loss_fn(x, c)
                encoder_loss.backward()

                # Monitor gradients during aggressive training following callback pattern
                if callback and batch_idx % max(1, len(data_loader) // 5) == 0:
                    grad_total, grad_max, grad_min = self.monitor_gradients(epoch, batch_idx, mode='aggressive')
                    callback.log_gradients({
                        'encoder_total_norm': grad_total,
                        'encoder_max_norm': grad_max,
                        'encoder_min_norm': grad_min
                    })

                # Only step encoder optimizer
                self.encoder_optimizer.step()
                self.encoder_scheduler.step()

                # DUSTINDUSTIN: Track batch results following callback pattern
                if callback:
                    callback.after_batch(total_batches, {
                        'encoder_loss': encoder_loss.item(),
                        'mse': mse.item(),
                        'kld': kld.item()
                    })

                # Track metrics
                total_encoder_loss += encoder_loss.item()
                total_mse += mse.item()
                total_kld += kld.item()
                total_batches += 1

        # Calculate averages
        avg_metrics = {
            'encoder_loss': total_encoder_loss / total_batches,
            'mse': total_mse / total_batches,
            'kld': total_kld / total_batches
        }

        return avg_metrics

    def standard_decoder_training(self, data_loader, epoch=0, callback=None):
        """Perform standard decoder training"""
        self.model.train()

        epoch_decoder_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(data_loader):
            if self.conditional:
                x, c = batch
                x, c = x.to(self.device), c.to(self.device)
            else:
                x = batch[0].to(self.device)
                c = None

            # IMPORTANT: Zero gradients for ALL optimizers
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Only update decoder parameters
            decoder_loss = self.decoder_loss_fn(x, c)
            decoder_loss.backward()

            # Monitor gradients during decoder training following callback pattern
            if callback and batch_idx % max(1, len(data_loader) // 5) == 0:
                grad_total, grad_max, grad_min = self.monitor_gradients(epoch, batch_idx, mode='aggressive_decoder')
                callback.log_gradients({
                    'decoder_total_norm': grad_total,
                    'decoder_max_norm': grad_max,
                    'decoder_min_norm': grad_min
                })

            # Only step decoder optimizer
            self.decoder_optimizer.step()
            self.decoder_scheduler.step()

            # Track batch results following callback pattern
            if callback:
                callback.after_batch(num_batches, {
                    'decoder_loss': decoder_loss.item()
                })

            epoch_decoder_loss += decoder_loss.item()
            num_batches += 1

        return epoch_decoder_loss / num_batches

    def monitor_gradients(self, epoch=0, iteration=0, mode=None):
        """Monitor gradients to detect exploding or vanishing issues"""
        # Calculate and store gradient norms
        total_norm = 0
        max_norm = 0
        min_norm = float('inf')
        layer_norms = {}

        # Define which parameters to monitor based on mode
        if mode == 'aggressive':
            # Only monitor encoder parameters (encoder.*)
            target_parameters = [(name, param) for name, param in self.model.named_parameters()
                                 if name.startswith('encoder.') and param.grad is not None]
            mode_prefix = "Encoder"
        elif mode == 'aggressive_decoder':
            # Only monitor decoder parameters (decoder.*)
            target_parameters = [(name, param) for name, param in self.model.named_parameters()
                                 if name.startswith('decoder.') and param.grad is not None]
            mode_prefix = "Decoder"
        else:
            # Monitor all parameters for standard training
            target_parameters = [(name, param) for name, param in self.model.named_parameters()
                                 if param.grad is not None]
            mode_prefix = "All"

        # Calculate gradient norms for the relevant parameters
        for name, param in target_parameters:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2

            max_norm = max(max_norm, param_norm)
            min_norm = min(min_norm, param_norm if param_norm > 0 else float('inf'))
            layer_norms[name] = param_norm

        total_norm = total_norm ** 0.5

        if mode == 'aggressive':
            # Store encoder-specific gradients during aggressive encoder training
            self.logs['aggressive_encoder_grad_norm'].append(total_norm)
            self.logs['aggressive_encoder_grad_max'].append(max_norm)
            self.logs['aggressive_encoder_grad_min'].append(min_norm)
        elif mode == 'aggressive_decoder':
            # Store decoder-specific gradients during aggressive decoder training
            self.logs['aggressive_decoder_grad_norm'].append(total_norm)
            self.logs['aggressive_decoder_grad_max'].append(max_norm)
            self.logs['aggressive_decoder_grad_min'].append(min_norm)
        else:
            # Store standard training gradients (all parameters)
            if 'grad_norm' not in self.logs:
                self.logs['grad_norm'] = []
            if 'max_grad' not in self.logs:
                self.logs['max_grad'] = []
            if 'min_grad' not in self.logs:
                self.logs['min_grad'] = []

            self.logs['grad_norm'].append(total_norm)
            self.logs['max_grad'].append(max_norm)
            self.logs['min_grad'].append(min_norm)

        # Detect issues
        if total_norm > 10.0:  # Threshold for exploding gradients
            print(f"WARNING: Potential exploding gradient detected at epoch {epoch}, iteration {iteration}")
            print(f"Total gradient norm: {total_norm:.4f}")
            # Print top 3 layers with largest gradients
            top_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Largest gradient layers:")
            for name, norm in top_layers:
                print(f"  {name}: {norm:.4f}")

        if max_norm < 1e-4:  # Threshold for vanishing gradients
            print(f"WARNING: Potential vanishing gradient detected at epoch {epoch}, iteration {iteration}")
            print(f"Maximum gradient norm: {max_norm:.8f}")

        return total_norm, max_norm, min_norm

    def loss_fn(self, recon_x, x, mean, log_var):
        """
        VAE loss function combining reconstruction loss and KL divergence.

        Args:
            recon_x: Reconstructed Pareto set
            x: Original Pareto set
            mean: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            Combined loss
        """
        # MSE reconstruction loss
        MSE = torch.nn.functional.mse_loss(
            recon_x, x, reduction='sum')

        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        beta = 0.001

        return (MSE + beta * KLD) / x.size(0), MSE / x.size(0), KLD / x.size(0)

    def prepare_data(self, X, Y=None, contexts=None):
        """
        Prepare data for training.

        Args:
            X: Pareto set solutions [batch_size, input_dim]
            Y: Pareto front values [batch_size, output_dim] - not used in current implementation
            contexts: Context/preference vectors [batch_size, context_dim]

        Returns:
            DataLoader for training
        """
        # Determine dataset size
        dataset_size = len(X)

        # Adjust batch size based on dataset size
        adjusted_batch_size = min(
            self.batch_size,  # Don't exceed configured max batch size
            max(1, dataset_size // 8)  # Aim for ~20 batches minimum
        )

        if contexts is not None and self.conditional:
            dataset = TensorDataset(
                torch.FloatTensor(X),
                torch.FloatTensor(contexts)
            )
        else:
            dataset = TensorDataset(torch.FloatTensor(X))

        # print(f"adjusted_batch_size is {adjusted_batch_size}")

        return DataLoader(
            dataset=dataset,
            batch_size=adjusted_batch_size,
            shuffle=True
        )

    def train(self, X, Y=None, contexts=None, callback=None):
        """
        Train the VAE model.

        Args:
            X: Pareto set solutions [num_samples, input_dim]
            Y: Pareto front values [num_samples, output_dim] - not used in current implementation
            contexts: Context/preference vectors [num_samples, context_dim]

        Returns:
            Training logs
        """
        data_loader = self.prepare_data(X=X, contexts=contexts)

        self.max_aggressive_epochs = self.epochs // 2
        self.current_aggressive_epoch = 0
        self.aggressive_phase = True

        # Track mutual information to decide when to stop aggressive training
        prev_mi = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_mse = 0
            epoch_kld = 0

            # Compute mutual information to decide training strategy
            current_mi = self.compute_mutual_information(data_loader)
            self.logs['mutual_info'].append(current_mi)

            # Decide whether to use aggressive training
            if (self.aggressive_training and
                    self.aggressive_phase and
                    self.current_aggressive_epoch < self.max_aggressive_epochs):

                print(f"Epoch {epoch + 1}: Aggressive training (MI: {current_mi:.4f})")

                # NEW: Enhanced aggressive training with logging
                encoder_metrics = self.aggressive_encoder_training(data_loader, num_updates=5, epoch=epoch, callback=callback)
                decoder_loss = self.standard_decoder_training(data_loader, epoch=epoch, callback=callback)

                # NEW: Store aggressive training metrics
                combined_loss = encoder_metrics['encoder_loss'] + decoder_loss
                self.logs['aggressive_loss'].append(combined_loss)
                self.logs['aggressive_encoder_loss'].append(encoder_metrics['encoder_loss'])
                self.logs['aggressive_decoder_loss'].append(decoder_loss)
                self.logs['aggressive_mse'].append(encoder_metrics['mse'])
                self.logs['aggressive_kld'].append(encoder_metrics['kld'])

                # Check stopping criterion: if MI stops climbing, stop aggressive training
                if current_mi <= prev_mi + 1e-4:  # Small threshold for MI improvement
                    self.aggressive_phase = False
                    print(f"Stopping aggressive training at epoch {epoch + 1} (MI plateaued)")

                self.current_aggressive_epoch += 1

                # Log aggressive training metrics
                self.logs['aggressive_epochs'].append(epoch)

                print(f"  Encoder Loss: {encoder_metrics['encoder_loss']:.4f}")
                print(f"  Decoder Loss: {decoder_loss:.4f}")
                print(f"  Combined Loss: {combined_loss:.4f}")

                if callback:
                    combined_loss = encoder_metrics['encoder_loss'] + decoder_loss
                    epoch_metrics = {
                        'mutual_info': current_mi,
                        'loss': combined_loss,  # Shared name with standard training
                        'mse': encoder_metrics['mse'],  # Shared name with standard training
                        'kld': encoder_metrics['kld'],  # Shared name with standard training
                        'encoder_loss': encoder_metrics['encoder_loss'],
                        'decoder_loss': decoder_loss,
                    }
                    callback.after_epoch(epoch, epoch_metrics)

            else:
                # Standard VAE training (both encoder and decoder together)
                print(f"Epoch {epoch + 1}: Standard training (MI: {current_mi:.4f})")

                # For visualization of latent space
                latent_points = []
                latent_contexts = []

                for iteration, batch in enumerate(data_loader):
                    # Unpack batch
                    if self.conditional:
                        x, c = batch
                        # print(batch[0].shape)
                        x, c = x.to(self.device), c.to(self.device)
                    else:
                        x = batch[0].to(self.device)
                        c = None

                    # Forward pass
                    if self.conditional:
                        recon_x, mean, log_var, z = self.model(x, c)
                    else:
                        recon_x, mean, log_var, z = self.model(x)

                    # Calculate loss
                    loss, mse, kld = self.loss_fn(recon_x, x, mean, log_var)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Monitor gradients following callback pattern for standard training
                    if (iteration + 1) % max(1, len(data_loader) // 5) == 0:
                        grad_total, grad_max, grad_min = self.monitor_gradients(epoch, iteration, mode='standard')

                        # Log gradients following callback pattern
                        if callback:
                            callback.log_gradients({
                                'total_norm': grad_total,
                                'max_norm': grad_max,
                                'min_norm': grad_min
                            })

                    self.optimizer.step()
                    self.scheduler.step()

                    # Track batch results following callback pattern for standard training
                    if callback:
                        callback.after_batch(iteration, {
                            'loss': loss.item(),
                            'mse': mse.item(),
                            'kld': kld.item()
                        })

                    # Track results
                    epoch_loss += loss.item()
                    epoch_mse += mse.item()
                    epoch_kld += kld.item()

                    # Store latent points for visualization
                    latent_points.append(z.detach().cpu().numpy())
                    if self.conditional:
                        latent_contexts.append(c.detach().cpu().numpy())

                    # Print progress
                    if (iteration + 1) % max(1, len(data_loader) // 5) == 0:
                        print(f"Epoch {epoch + 1}/{self.epochs}, Batch {iteration + 1}/{len(data_loader)}, "
                              f"Loss: {loss.item():.4f}")


                avg_loss = epoch_loss / len(data_loader)
                self.logs['loss'].append(avg_loss)
                self.logs['mse'].append(epoch_mse / len(data_loader))
                self.logs['kld'].append(epoch_kld / len(data_loader))
                print(f"Epoch {epoch + 1}/{self.epochs} completed, Avg Loss: {avg_loss:.4f}")

                if callback:
                    epoch_metrics = {
                        'mutual_info': current_mi,
                        'loss': avg_loss,  # DUSTIN: Shared name with aggressive training
                        'mse': epoch_mse / len(data_loader),  # DUSTIN: Shared name with aggressive training
                        'kld': epoch_kld / len(data_loader),  # DUSTIN: Shared name with aggressive training
                    }
                    callback.after_epoch(epoch, epoch_metrics)

            prev_mi = current_mi
            # Visualize results periodically
            # if (epoch + 1) % max(1, self.epochs // 10) == 0:
            #     self._visualize_results(epoch, latent_points, latent_contexts)

        return self.logs

    def validate(self, X_val, contexts_val=None):
        """Compute validation loss"""
        self.model.eval()
        with torch.no_grad():
            if self.conditional:
                recon_x, mean, log_var, _ = self.model(X_val, contexts_val)
            else:
                recon_x, mean, log_var, _ = self.model(X_val)

            loss, mse, kld = self.loss_fn(recon_x, X_val, mean, log_var)

        return loss.item(), mse.item(), kld.item()

    def _visualize_results(self, epoch, latent_points, latent_contexts=None):
        """
        Visualize training results.

        Args:
            epoch: Current epoch number
            latent_points: List of latent points from the epoch
            latent_contexts: List of context variables if available
        """
        # Combine all latent points
        z_all = np.vstack(latent_points)

        # Plot latent space
        if self.latent_dim >= 2:
            plt.figure(figsize=(10, 8))

            # If conditional, color by first dimension of context
            if self.conditional and latent_contexts:
                c_all = np.vstack(latent_contexts)
                plt.scatter(z_all[:, 0], z_all[:, 1], c=c_all[:, 0], cmap='viridis', alpha=0.5)
                plt.colorbar(label='Context')
            else:
                plt.scatter(z_all[:, 0], z_all[:, 1], alpha=0.5)

            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title(f'Latent Space Visualization (Epoch {epoch + 1})')
            plt.savefig(os.path.join(self.save_dir, self.timestamp, f'latent_space_epoch_{epoch + 1}.png'))
            plt.close()

        # Sample new solutions
        num_samples = min(10, self.batch_size)
        z = torch.randn(num_samples, self.latent_dim).to(self.device)

        if self.conditional:
            # Generate for different context values
            # Either use actual contexts or generate equidistant ones
            if latent_contexts:
                c_unique = np.unique(np.vstack(latent_contexts)[:, 0])
                if len(c_unique) >= num_samples:
                    c_values = c_unique[:num_samples]
                else:
                    c_values = np.linspace(c_unique.min(), c_unique.max(), num_samples)
            else:
                c_values = np.linspace(0, 1, num_samples)

            c = torch.FloatTensor(
                np.hstack([c_values[:, None], np.zeros((num_samples, self.context_dim - 1))])
            ).to(self.device)

            generated_x = self.model.inference(z, c=c)
        else:
            generated_x = self.model.inference(z)

        # Visualize generated solutions
        generated_x = generated_x.detach().cpu().numpy()

        # Plot some of the generated solutions
        plt.figure(figsize=(12, 8))
        for i in range(min(5, num_samples)):
            plt.subplot(1, 5, i + 1)
            plt.plot(generated_x[i])
            if self.conditional:
                plt.title(f'Context: {c_values[i]:.2f}')
            plt.ylim(0, 1)

        plt.suptitle(f'Generated Solutions (Epoch {epoch + 1})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, self.timestamp, f'generated_solutions_epoch_{epoch + 1}.png'))
        plt.close()

    def save_model(self, path=None):
        """
        Save trained model.

        Args:
            path: Path to save the model, defaults to timestamp directory
        """
        if path is None:
            path = os.path.join(self.save_dir, self.timestamp, 'model.pt')

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'latent_dim': self.latent_dim,
            'context_dim': self.context_dim,
            'conditional': self.conditional,
            'logs': self.logs
        }, path)

        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load trained model.

        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Recreate the model architecture
        # encoder_sizes = [checkpoint['input_dim'],
        #                  max(2 * checkpoint['input_dim'], 4 * checkpoint['latent_dim'], 128),
        #                  max(checkpoint['input_dim'], 2 * checkpoint['latent_dim'], 64)]
        encoder_sizes = [checkpoint['input_dim'],
                         max(checkpoint['input_dim'], 2 * latent_dim, 10)]
        # decoder_sizes = [max(checkpoint['input_dim'], 2 * checkpoint['latent_dim'], 64),
        #                  max(2 * checkpoint['input_dim'], 4 * checkpoint['latent_dim'], 128),
        #                  checkpoint['input_dim']]
        decoder_sizes = [max(checkpoint['input_dim'], 2 * latent_dim, 10),
                         checkpoint['input_dim']]

        self.model = VAE(
            encoder_layer_sizes=encoder_sizes,
            latent_size=checkpoint['latent_dim'],
            decoder_layer_sizes=decoder_sizes,
            conditional=checkpoint['conditional'],
            context_size=checkpoint['context_dim']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logs = checkpoint['logs']

        print(f"Model loaded from {path}")

    def generate_solutions(self, contexts=None, num_samples=10):
        """
        Generate new Pareto set solutions.

        Args:
            contexts: Context vectors to condition generation on
            num_samples: Number of solutions to generate per context

        Returns:
            Generated Pareto set solutions
        """
        self.model.eval()
        with torch.no_grad():
            if self.conditional and contexts is not None:
                # Convert contexts to tensor if needed
                if not isinstance(contexts, torch.Tensor):
                    contexts = torch.FloatTensor(contexts).to(self.device)

                # Expand contexts to match number of samples if needed
                if contexts.size(0) == 1 and num_samples > 1:
                    contexts = contexts.repeat(num_samples, 1)

                num_contexts = contexts.size(0)
                z = torch.randn(num_contexts, self.latent_dim).to(self.device)
                generated_x = self.model.inference(z, c=contexts)
            else:
                z = torch.randn(num_samples, self.latent_dim).to(self.device)
                generated_x = self.model.inference(z)

        return generated_x.cpu().numpy()


# Example usage
if __name__ == "__main__":
    # Generate some synthetic Pareto data for demonstration
    import numpy as np

    # Parameters
    input_dim = 10  # Dimension of decision variables
    output_dim = 2  # Number of objectives
    latent_dim = output_dim - 1  # Dimension of latent space
    context_dim = 3  # Dimension of context
    num_samples = 1000  # Number of samples

    # Generate synthetic data
    X = np.random.rand(num_samples, input_dim)  # Pareto set (random for demonstration)
    Y = np.random.rand(num_samples, output_dim)  # Pareto front (random for demonstration)
    contexts = np.random.rand(num_samples, context_dim)  # Context variables

    # Create and train the model
    trainer = ParetoVAETrainer(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        context_dim=context_dim,
        conditional=True,
        epochs=10,
        batch_size=32,
        trainer_id="CMOBO"
    )

    # Train the model
    logs = trainer.train(X, Y, contexts)

    # Save the model
    trainer.save_model()

    # Generate new solutions for a specific context
    new_context = np.array([[0.5, 0.5, 0.5]])  # Example context
    new_solutions = trainer.generate_solutions(new_context, num_samples=5)

    print(f"Generated solutions shape: {new_solutions.shape}")
    print("Sample solution:")
    print(new_solutions[0])

