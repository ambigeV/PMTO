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
                self.MLP.add_module(name="sigmoid", module=torch.nn.Sigmoid())
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
                 trainer_id = None,
                 save_dir='./results'):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.conditional = conditional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir

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
        encoder_sizes = [input_dim, max(input_dim, 2 * latent_dim, 10)]
        decoder_sizes = [max(input_dim, 2 * latent_dim, 10), input_dim]

        self.model = VAE(
            encoder_layer_sizes=encoder_sizes,
            latent_size=latent_dim,
            decoder_layer_sizes=decoder_sizes,
            conditional=conditional,
            context_size=context_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.epochs * 0.5), int(self.epochs * 0.75)],
            gamma=0.1
        )

        # Training logs
        self.logs = defaultdict(list)

    def monitor_gradients(self, epoch=0, iteration=0):
        """Monitor gradients to detect exploding or vanishing issues"""
        # Calculate and store gradient norms
        total_norm = 0
        max_norm = 0
        min_norm = float('inf')
        layer_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                max_norm = max(max_norm, param_norm)
                min_norm = min(min_norm, param_norm if param_norm > 0 else float('inf'))
                layer_norms[name] = param_norm

        total_norm = total_norm ** 0.5

        # Store in logs
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
        beta = 1

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

    def train(self, X, Y=None, contexts=None):
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

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_mse = 0
            epoch_kld = 0

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
                # Monitor gradients
                if (iteration + 1) % max(1, len(data_loader) // 5) == 0:
                    grad_total, grad_max, grad_min = self.monitor_gradients(epoch, iteration)
                self.optimizer.step()
                self.scheduler.step()

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

            # End of epoch

            avg_loss = epoch_loss / len(data_loader)
            self.logs['loss'].append(avg_loss)
            self.logs['mse'].append(epoch_mse / len(data_loader))
            self.logs['kld'].append(epoch_kld / len(data_loader))
            print(f"Epoch {epoch + 1}/{self.epochs} completed, Avg Loss: {avg_loss:.4f}")

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
