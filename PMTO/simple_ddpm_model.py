import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from collections import defaultdict


class Swish(nn.Module):
    """Swish activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class SimpleMLP(nn.Module):
    """
    Simple MLP diffusion model that imitates the Diffusion-BBO architecture.
    Takes input x, timestep t, and conditioning c (context + weights).
    """

    def __init__(self, input_dim, conditioning_dim, hidden_dim=128, num_layers=4):
        super().__init__()

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build the MLP layers
        layers = []

        # Input layer: [input, timestep, conditioning]
        input_size = input_dim + 1 + conditioning_dim
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(Swish())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Swish())

        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x, t, c):
        """
        Args:
            x: Input tensor [batch_size, input_dim]
            t: Timestep tensor [batch_size] or [batch_size, 1]
            c: Conditioning tensor [batch_size, conditioning_dim] (context + weights)
        """
        batch_size = x.shape[0]

        # Ensure t is the right shape
        if t.dim() == 1:
            t = t.unsqueeze(1).float()
        else:
            t = t.float()

        # Concatenate all inputs
        h = torch.cat([x, t, c], dim=1)

        # Forward pass
        noise_pred = self.main(h)

        return noise_pred


class SimpleDDPM(nn.Module):
    """
    Simple DDPM model for generating Pareto set solutions.
    Uses standard DDPM sampling instead of DDIM for simplicity.
    """

    def __init__(self, input_dim, conditioning_dim, timesteps=1000, hidden_dim=128, num_layers=4):
        super().__init__()

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.timesteps = timesteps

        # Initialize the simple MLP model
        self.model = SimpleMLP(
            input_dim=input_dim,
            conditioning_dim=conditioning_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # Initialize noise schedule (linear schedule like original DDPM)
        self.register_buffer('betas', self._linear_beta_schedule(timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                             F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        # Precompute values for sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """Linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to x_start according to timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, c, noise=None):
        """
        Compute the loss for training.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, c)

        loss = F.mse_loss(noise, predicted_noise)
        return loss

    def p_sample(self, x, t, c):
        """
        Single denoising step (reverse process).
        """
        betas_t = self.betas[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1)

        # Predict noise
        predicted_noise = self.model(x, t, c)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, shape, c):
        """
        DDPM sampling loop.
        """
        device = next(self.parameters()).device

        # Start from random noise
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, c)

        return x

    def sample(self, conditioning, num_samples):
        """
        Generate samples using DDPM sampling.

        Args:
            conditioning: Conditioning tensor [batch_size, conditioning_dim]
            num_samples: Number of samples to generate

        Returns:
            Generated samples [num_samples, input_dim]
        """
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)

        if conditioning.shape[0] == 1 and num_samples > 1:
            conditioning = conditioning.repeat(num_samples, 1)

        shape = (num_samples, self.input_dim)

        with torch.no_grad():
            samples = self.p_sample_loop(shape, conditioning)

        # Clamp to [0, 1] range to handle boundary constraints
        samples = torch.clamp(samples, 0, 1)

        return samples

    def forward(self, x, c):
        """
        Forward pass for training.
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Compute loss
        loss = self.p_losses(x, t, c)
        return loss


class SimpleParetoTrainer:
    """
    Simplified trainer class for Pareto set DDPM modeling.
    """

    def __init__(self,
                 input_dim,
                 conditioning_dim,
                 timesteps=1000,
                 hidden_dim=128,
                 num_layers=4,
                 learning_rate=1e-3,
                 batch_size=64,
                 epochs=50,
                 device=None,
                 trainer_id=None,
                 save_dir='./results'):

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.timesteps = timesteps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        if trainer_id is None:
            self.timestamp = f"run_{int(time.time())}"
        else:
            self.timestamp = trainer_id

        os.makedirs(os.path.join(save_dir, self.timestamp), exist_ok=True)

        # Create model
        self.model = SimpleDDPM(
            input_dim=input_dim,
            conditioning_dim=conditioning_dim,
            timesteps=timesteps,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.epochs * 0.8), int(self.epochs * 0.9)],
            gamma=0.1
        )

        # Training logs
        self.logs = defaultdict(list)

    def prepare_data(self, X, contexts):
        """
        Prepare data for training.

        Args:
            X: Pareto set solutions [num_samples, input_dim]
            contexts: Combined context and weight vectors [num_samples, conditioning_dim]

        Returns:
            DataLoader for training
        """
        dataset_size = len(X)

        # Adjust batch size based on dataset size
        adjusted_batch_size = min(
            self.batch_size,
            max(1, dataset_size // 8)
        )

        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(contexts)
        )

        return DataLoader(
            dataset=dataset,
            batch_size=adjusted_batch_size,
            shuffle=True
        )

    def train(self, X, contexts, callback=None):
        """
        Train the DDPM model.

        Args:
            X: Pareto set solutions [num_samples, input_dim]
            contexts: Combined context and weight vectors [num_samples, conditioning_dim]
            callback: Optional callback for logging

        Returns:
            Training logs
        """
        data_loader = self.prepare_data(X=X, contexts=contexts)

        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0

            self.model.train()

            for iteration, batch in enumerate(data_loader):
                x, c = batch
                x, c = x.to(self.device), c.to(self.device)

                # Forward pass
                loss = self.model(x, c)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Track batch results
                if callback:
                    callback.after_batch(iteration, {'loss': loss.item()})

                epoch_loss += loss.item()
                num_batches += 1

                # Print progress
                if iteration % max(1, len(data_loader) // 5) == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}, Batch {iteration + 1}/{len(data_loader)}, "
                          f"Loss: {loss.item():.4f}")

            self.scheduler.step()

            # Log epoch results
            avg_loss = epoch_loss / num_batches
            self.logs['loss'].append(avg_loss)

            if callback:
                callback.after_epoch(epoch, {'loss': avg_loss})

            print(f"Epoch {epoch + 1}/{self.epochs} completed, Avg Loss: {avg_loss:.4f}")

        return self.logs

    def generate_solutions(self, contexts, num_samples=10):
        """
        Generate new Pareto set solutions.

        Args:
            contexts: Combined context and weight vectors
            num_samples: Number of solutions to generate

        Returns:
            Generated Pareto set solutions
        """
        self.model.eval()

        if not isinstance(contexts, torch.Tensor):
            contexts = torch.FloatTensor(contexts).to(self.device)

        # Expand contexts to match number of samples if needed
        if contexts.size(0) == 1 and num_samples > 1:
            contexts = contexts.repeat(num_samples, 1)

        generated_x = self.model.sample(contexts, num_samples)

        return generated_x.cpu().numpy()

    def save_model(self, path=None):
        """Save trained model."""
        if path is None:
            path = os.path.join(self.save_dir, self.timestamp, 'simple_ddpm_model.pt')

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'conditioning_dim': self.conditioning_dim,
            'timesteps': self.timesteps,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'logs': self.logs
        }, path)

        print(f"Simple DDPM model saved to {path}")

    def load_model(self, path):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)

        # Recreate model
        self.model = SimpleDDPM(
            input_dim=checkpoint['input_dim'],
            conditioning_dim=checkpoint['conditioning_dim'],
            timesteps=checkpoint['timesteps'],
            hidden_dim=checkpoint.get('hidden_dim', 128),
            num_layers=checkpoint.get('num_layers', 4)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logs = checkpoint['logs']

        # Update trainer attributes
        self.hidden_dim = checkpoint.get('hidden_dim', 128)
        self.num_layers = checkpoint.get('num_layers', 4)

        print(f"Simple DDPM model loaded from {path}")