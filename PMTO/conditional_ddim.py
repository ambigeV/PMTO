import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta values in DDIM.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ConditionalMLPUNet(nn.Module):
    """
    Lightweight MLP-based UNet for conditional diffusion.
    Takes input x, timestep t, and conditioning c.
    """

    def __init__(self, input_dim, condition_dim, hidden_dim=256, num_layers=4, max_timesteps=1000):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.max_timesteps = max_timesteps  # ADD THIS

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim // 2)

        # Main network - downsampling path
        self.down_layers = nn.ModuleList()
        current_dim = hidden_dim + hidden_dim // 2  # input + time + condition

        for i in range(num_layers // 2):
            self.down_layers.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))
            current_dim = hidden_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Main network - upsampling path with skip connections
        self.up_layers = nn.ModuleList()
        for i in range(num_layers // 2):
            # Skip connection doubles the input dimension
            skip_dim = hidden_dim * 2 if i > 0 else hidden_dim * 2
            self.up_layers.append(nn.Sequential(
                nn.Linear(skip_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, x, t, c):
        """
        Args:
            x: Input tensor [batch_size, input_dim]
            t: Timestep tensor [batch_size] or [batch_size, 1]
            c: Condition tensor [batch_size, condition_dim]
        """
        batch_size = x.shape[0]

        # Ensure t is the right shape
        if t.dim() == 1:
            t = t.unsqueeze(1).float()
        else:
            t = t.float()

        # Normalize timestep to [0, 1]
        # t_normalized = t / 1000.0  # Assuming max timestep is 1000
        t_normalized = t / self.max_timesteps  # NEW - use parameter

        # Embeddings
        t_emb = self.time_embed(t_normalized)  # [batch_size, hidden_dim//2]
        c_emb = self.condition_embed(c)  # [batch_size, hidden_dim//2]
        x_emb = self.input_proj(x)  # [batch_size, hidden_dim//2]

        # Combine all embeddings
        h = torch.cat([x_emb, t_emb, c_emb], dim=1)  # [batch_size, hidden_dim + hidden_dim//2]

        # Store skip connections
        skip_connections = []

        # Downsampling path
        for layer in self.down_layers:
            h = layer(h)
            skip_connections.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Upsampling path with skip connections
        for i, layer in enumerate(self.up_layers):
            if i < len(skip_connections):
                skip = skip_connections[-(i + 1)]
                h = torch.cat([h, skip], dim=1)
            h = layer(h)

        # Output projection
        epsilon = self.output_proj(h)

        return epsilon


class ConditionalDDIM(nn.Module):
    """
    Conditional DDIM model for generating Pareto set solutions.
    """

    def __init__(self, input_dim, condition_dim, timesteps=1000, hidden_dim=256, num_layers=4):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.timesteps = timesteps

        # Initialize the UNet model
        self.model = ConditionalMLPUNet(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_timesteps=timesteps  # ADD THIS PARAMETER
        )

        # Initialize noise schedule
        self.register_buffer('betas', cosine_beta_schedule(timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                             F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        # Precompute values for sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - self.alphas_cumprod))

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

    def ddim_sample(self, shape, c, num_steps=20, eta=0.0):
        """
        DDIM sampling with conditioning.

        Args:
            shape: Shape of samples to generate [batch_size, input_dim]
            c: Conditioning tensor [batch_size, condition_dim]
            num_steps: Number of DDIM steps (default: 20)
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)
        """
        batch_size = shape[0]
        device = next(self.parameters()).device

        # Create sampling schedule
        step_size = self.timesteps // num_steps
        timesteps = torch.arange(0, self.timesteps, step_size).long()
        timesteps = torch.cat([timesteps, torch.tensor([self.timesteps - 1])])
        timesteps = timesteps[-num_steps:]  # Take last num_steps
        timesteps = torch.flip(timesteps, [0])  # Reverse for sampling

        # Start from random noise
        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size).to(device)

            # Predict noise
            with torch.no_grad():
                predicted_noise = self.model(x, t_batch, c)

            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t]

            if i < len(timesteps) - 1:
                alpha_cumprod_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

            # Compute direction to x_t
            if i < len(timesteps) - 1:
                # DDIM update
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
                ) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)

                noise = torch.randn_like(x) if eta > 0 else 0

                x = (
                        torch.sqrt(alpha_cumprod_t_prev) * pred_x0 +
                        torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * predicted_noise +
                        sigma_t * noise
                )
            else:
                x = pred_x0

        # Clamp output to [0, 1] range like VAE
        x = torch.clamp(x, 0, 1)
        return x

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


class ParetoDDIMTrainer:
    """
    Trainer class for Pareto set/front DDIM modeling.
    """

    def __init__(self,
                 input_dim,
                 output_dim=None,
                 condition_dim=0,
                 timesteps=1000,
                 hidden_dim=256,
                 num_layers=4,
                 learning_rate=0.001,
                 batch_size=64,
                 epochs=100,
                 device=None,
                 trainer_id=None,
                 save_dir='./results',
                 num_sampling_steps=20):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim  # ADD THIS LINE
        self.num_layers = num_layers  # ADD THIS LINE
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir
        self.num_sampling_steps = num_sampling_steps

        # Determine device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create directory for saving results
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create timestamp for this run
        if trainer_id is None:
            # Generate a default ID if none provided
            import time
            self.timestamp = f"run_{int(time.time())}"
            # OR use process ID: self.timestamp = f"run_{os.getpid()}"
            # OR use datetime: self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.timestamp = trainer_id

        os.makedirs(os.path.join(save_dir, self.timestamp), exist_ok=True)

        # Create model
        self.model = ConditionalDDIM(
            input_dim=input_dim,
            condition_dim=condition_dim,
            timesteps=timesteps,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.epochs * 0.5), int(self.epochs * 0.75)],
            gamma=0.1
        )

        # Training logs
        self.logs = defaultdict(list)
        self._initialize_logs()

    def _initialize_logs(self):
        """Initialize all log containers"""
        log_keys = [
            'loss', 'grad_norm', 'max_grad', 'min_grad'
        ]

        for key in log_keys:
            self.logs[key] = []

    def prepare_data(self, X, contexts=None):
        """
        Prepare data for training.

        Args:
            X: Pareto set solutions [batch_size, input_dim]
            contexts: Context/preference vectors [batch_size, condition_dim]

        Returns:
            DataLoader for training
        """
        dataset_size = len(X)

        # Adjust batch size based on dataset size
        adjusted_batch_size = min(
            self.batch_size,
            max(1, dataset_size // 8)
        )

        if contexts is not None:
            dataset = TensorDataset(
                torch.FloatTensor(X),
                torch.FloatTensor(contexts)
            )
        else:
            dataset = TensorDataset(torch.FloatTensor(X))

        return DataLoader(
            dataset=dataset,
            batch_size=adjusted_batch_size,
            shuffle=True
        )

    def monitor_gradients(self, epoch=0, iteration=0):
        """Monitor gradients to detect issues"""
        total_norm = 0
        max_norm = 0
        min_norm = float('inf')

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                min_norm = min(min_norm, param_norm if param_norm > 0 else float('inf'))

        total_norm = total_norm ** 0.5

        # Store gradients
        self.logs['grad_norm'].append(total_norm)
        self.logs['max_grad'].append(max_norm)
        self.logs['min_grad'].append(min_norm)

        # Detect issues
        if total_norm > 10.0:
            print(f"WARNING: Potential exploding gradient detected at epoch {epoch}, iteration {iteration}")
            print(f"Total gradient norm: {total_norm:.4f}")

        if max_norm < 1e-4:
            print(f"WARNING: Potential vanishing gradient detected at epoch {epoch}, iteration {iteration}")
            print(f"Maximum gradient norm: {max_norm:.8f}")

        return total_norm, max_norm, min_norm

    def train(self, X, contexts=None, callback=None):
        """
        Train the DDIM model.

        Args:
            X: Pareto set solutions [num_samples, input_dim]
            contexts: Context/preference vectors [num_samples, condition_dim]
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
                if contexts is not None:
                    x, c = batch
                    x, c = x.to(self.device), c.to(self.device)
                else:
                    x = batch[0].to(self.device)
                    c = torch.zeros(x.shape[0], self.condition_dim).to(self.device)

                # Forward pass
                loss = self.model(x, c)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Monitor gradients
                if iteration % max(1, len(data_loader) // 5) == 0:
                    grad_total, grad_max, grad_min = self.monitor_gradients(epoch, iteration)

                    if callback:
                        callback.log_gradients({
                            'total_norm': grad_total,
                            'max_norm': grad_max,
                            'min_norm': grad_min
                        })

                self.optimizer.step()

                # Track batch results
                if callback:
                    callback.after_batch(iteration, {
                        'loss': loss.item()
                    })

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
                callback.after_epoch(epoch, {
                    'loss': avg_loss
                })

            print(f"Epoch {epoch + 1}/{self.epochs} completed, Avg Loss: {avg_loss:.4f}")

        return self.logs

    def generate_solutions(self, contexts=None, num_samples=10):
        """
        Generate new Pareto set solutions.

        Args:
            contexts: Context vectors to condition generation on
            num_samples: Number of solutions to generate

        Returns:
            Generated Pareto set solutions
        """
        self.model.eval()

        with torch.no_grad():
            if contexts is not None:
                if not isinstance(contexts, torch.Tensor):
                    contexts = torch.FloatTensor(contexts).to(self.device)

                # Expand contexts to match number of samples if needed
                if contexts.size(0) == 1 and num_samples > 1:
                    contexts = contexts.repeat(num_samples, 1)

                batch_size = contexts.size(0)
                shape = (batch_size, self.input_dim)

                generated_x = self.model.ddim_sample(
                    shape=shape,
                    c=contexts,
                    num_steps=self.num_sampling_steps
                )
            else:
                shape = (num_samples, self.input_dim)
                dummy_context = torch.zeros(num_samples, self.condition_dim).to(self.device)

                generated_x = self.model.ddim_sample(
                    shape=shape,
                    c=dummy_context,
                    num_steps=self.num_sampling_steps
                )

        return generated_x.cpu().numpy()

    def save_model(self, path=None):
        """Save trained model."""
        if path is None:
            path = os.path.join(self.save_dir, self.timestamp, 'ddim_model.pt')

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,  # ADD THIS LINE
            'num_layers': self.num_layers,  # ADD THIS LINE
            'condition_dim': self.condition_dim,
            'timesteps': self.timesteps,
            'logs': self.logs
        }, path)

        print(f"DDIM model saved to {path}")

    def load_model(self, path):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)

        # Recreate model
        self.model = ConditionalDDIM(
            input_dim=checkpoint['input_dim'],
            condition_dim=checkpoint['condition_dim'],
            timesteps=checkpoint['timesteps'],
            hidden_dim=checkpoint.get('hidden_dim', 256),  # ADD THIS with default
            num_layers=checkpoint.get('num_layers', 4)  # ADD THIS with default
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logs = checkpoint['logs']

        # Also update the trainer's attributes
        self.hidden_dim = checkpoint.get('hidden_dim', 256)  # ADD THIS
        self.num_layers = checkpoint.get('num_layers', 4)  # ADD THIS

        print(f"DDIM model loaded from {path}")


def test_cosine_beta_schedule():
    """Test the cosine beta schedule function."""
    print("Testing cosine_beta_schedule...")

    timesteps = 100
    betas = cosine_beta_schedule(timesteps)

    assert len(betas) == timesteps, f"Expected {timesteps} betas, got {len(betas)}"
    assert torch.all(betas >= 0) and torch.all(betas <= 0.999), "Betas should be in [0, 0.999]"
    assert betas[0] < betas[-1], "Betas should generally increase"

    print(f"✓ Beta schedule test passed. Min: {betas.min():.4f}, Max: {betas.max():.4f}")
    return True


def test_conditional_mlp_unet():
    """Test the ConditionalMLPUNet model."""
    print("\nTesting ConditionalMLPUNet...")

    batch_size = 4
    input_dim = 10
    condition_dim = 5
    hidden_dim = 64
    num_layers = 4

    model = ConditionalMLPUNet(
        input_dim=input_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    t = torch.randint(0, 1000, (batch_size,))
    c = torch.randn(batch_size, condition_dim)

    try:
        output = model(x, t, c)
        assert output.shape == (
        batch_size, input_dim), f"Expected output shape {(batch_size, input_dim)}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        print(f"✓ Forward pass test passed. Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass test failed: {str(e)}")
        return False

    # Test with different timestep formats
    t_2d = torch.randint(0, 1000, (batch_size, 1))
    try:
        output_2d = model(x, t_2d, c)
        assert output_2d.shape == output.shape, "Output shape should be consistent"
        print("✓ 2D timestep format test passed")
    except Exception as e:
        print(f"✗ 2D timestep test failed: {str(e)}")
        return False

    return True


def test_conditional_ddim():
    """Test the ConditionalDDIM model."""
    print("\nTesting ConditionalDDIM...")

    input_dim = 8
    condition_dim = 4
    timesteps = 100
    hidden_dim = 32
    num_layers = 4

    model = ConditionalDDIM(
        input_dim=input_dim,
        condition_dim=condition_dim,
        timesteps=timesteps,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    batch_size = 4
    x = torch.rand(batch_size, input_dim)  # Data in [0, 1]
    c = torch.randn(batch_size, condition_dim)

    # Test q_sample (forward diffusion)
    t = torch.randint(0, timesteps, (batch_size,))
    x_noisy = model.q_sample(x, t)
    assert x_noisy.shape == x.shape, "Noisy sample should have same shape as input"
    assert not torch.isnan(x_noisy).any(), "Noisy sample contains NaN"
    print("✓ Forward diffusion test passed")

    # Test loss computation
    try:
        loss = model(x, c)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss > 0, "Loss should be positive"
        print(f"✓ Loss computation test passed. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Loss computation test failed: {str(e)}")
        return False

    # Test DDIM sampling
    try:
        shape = (2, input_dim)
        c_sample = torch.randn(2, condition_dim)
        num_steps = 10

        with torch.no_grad():
            samples = model.ddim_sample(shape, c_sample, num_steps=num_steps)

        assert samples.shape == shape, f"Expected shape {shape}, got {samples.shape}"
        assert torch.all(samples >= 0) and torch.all(samples <= 1), "Samples should be in [0, 1]"
        assert not torch.isnan(samples).any(), "Samples contain NaN"
        print(f"✓ DDIM sampling test passed. Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
    except Exception as e:
        print(f"✗ DDIM sampling test failed: {str(e)}")
        return False

    return True


def test_pareto_ddim_trainer():
    """Test the ParetoDDIMTrainer class."""
    print("\nTesting ParetoDDIMTrainer...")

    # Create synthetic Pareto set data
    n_samples = 50
    input_dim = 6
    output_dim = 2
    condition_dim = 4
    hidden_dim = 32  # Specific hidden_dim for testing
    num_layers = 4

    # Generate synthetic Pareto-optimal solutions
    # These should be in [0, 1] range as per the model design
    X = torch.rand(n_samples, input_dim)
    contexts = torch.randn(n_samples, condition_dim)

    # Initialize trainer with specific architecture
    trainer = ParetoDDIMTrainer(
        input_dim=input_dim,
        output_dim=output_dim,
        condition_dim=condition_dim,
        timesteps=100,
        hidden_dim=hidden_dim,  # Explicitly set
        num_layers=num_layers,  # Explicitly set
        learning_rate=0.001,
        batch_size=16,
        epochs=5,  # Small number for testing
        trainer_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_dir='./test_results',
        num_sampling_steps=10
    )

    print(f"✓ Trainer initialized. Device: {trainer.device}")

    # Test data preparation
    try:
        data_loader = trainer.prepare_data(X, contexts)
        assert len(data_loader) > 0, "DataLoader is empty"
        print(f"✓ Data preparation test passed. Batches: {len(data_loader)}")
    except Exception as e:
        print(f"✗ Data preparation test failed: {str(e)}")
        return False

    # Test training (short run)
    try:
        print("Starting training test...")
        logs = trainer.train(X, contexts)

        assert 'loss' in logs, "Training logs should contain loss"
        assert len(logs['loss']) == trainer.epochs, "Should have loss for each epoch"
        assert all(loss > 0 for loss in logs['loss']), "All losses should be positive"

        final_loss = logs['loss'][-1]
        print(f"✓ Training test passed. Final loss: {final_loss:.4f}")
    except Exception as e:
        print(f"✗ Training test failed: {str(e)}")
        return False

    # Test generation
    try:
        # Test with contexts
        test_contexts = torch.randn(3, condition_dim)
        generated = trainer.generate_solutions(contexts=test_contexts, num_samples=3)

        assert generated.shape == (3, input_dim), f"Expected shape (3, {input_dim}), got {generated.shape}"
        assert not np.isnan(generated).any(), "Generated samples contain NaN"
        assert np.all(generated >= 0) and np.all(generated <= 1), "Generated samples should be in [0, 1]"

        print(f"✓ Generation test passed. Generated shape: {generated.shape}")
        print(f"  Sample range: [{generated.min():.4f}, {generated.max():.4f}]")
    except Exception as e:
        print(f"✗ Generation test failed: {str(e)}")
        return False

    # Test model saving and loading - CORRECTED VERSION
    try:
        save_path = os.path.join(trainer.save_dir, trainer.timestamp, 'test_model.pt')
        trainer.save_model(save_path)

        # Create new trainer with SAME architecture parameters
        new_trainer = ParetoDDIMTrainer(
            input_dim=input_dim,
            output_dim=output_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,  # MUST match original
            num_layers=num_layers,  # MUST match original
            timesteps=100,  # MUST match original
            trainer_id="test_load"
        )
        new_trainer.load_model(save_path)

        # Generate with loaded model
        loaded_generated = new_trainer.generate_solutions(contexts=test_contexts, num_samples=3)
        assert loaded_generated.shape == generated.shape, "Loaded model should generate same shape"

        print("✓ Save/load test passed")
    except Exception as e:
        print(f"✗ Save/load test failed: {str(e)}")
        print(f"  This likely means the architecture parameters don't match.")
        print(f"  Original: hidden_dim={hidden_dim}, num_layers={num_layers}")
        return False

    return True


def test_gradient_monitoring():
    """Test gradient monitoring functionality."""
    print("\nTesting gradient monitoring...")

    # Create a small model and trainer
    trainer = ParetoDDIMTrainer(
        input_dim=4,
        condition_dim=2,
        hidden_dim=16,
        num_layers=2,
        epochs=1
    )

    # Create some dummy gradients
    for param in trainer.model.parameters():
        param.grad = torch.randn_like(param) * 0.01

    try:
        total_norm, max_norm, min_norm = trainer.monitor_gradients(epoch=0, iteration=0)

        assert total_norm > 0, "Total norm should be positive"
        assert max_norm >= min_norm, "Max norm should be >= min norm"
        assert len(trainer.logs['grad_norm']) == 1, "Gradient norm should be logged"

        print(f"✓ Gradient monitoring test passed")
        print(f"  Total norm: {total_norm:.6f}")
        print(f"  Max norm: {max_norm:.6f}")
        print(f"  Min norm: {min_norm:.6f}")
    except Exception as e:
        print(f"✗ Gradient monitoring test failed: {str(e)}")
        return False

    return True


def test_edge_cases():
    """Test edge cases and potential failure modes."""
    print("\nTesting edge cases...")

    # Test with single sample
    try:
        trainer = ParetoDDIMTrainer(input_dim=2, condition_dim=1, epochs=1)
        X_single = torch.rand(1, 2)
        c_single = torch.rand(1, 1)

        # This should handle single sample gracefully
        data_loader = trainer.prepare_data(X_single, c_single)
        assert len(data_loader) > 0, "Should handle single sample"
        print("✓ Single sample test passed")
    except Exception as e:
        print(f"✗ Single sample test failed: {str(e)}")

    # Test with mismatched dimensions
    try:
        trainer = ParetoDDIMTrainer(input_dim=3, condition_dim=2)
        X_wrong = torch.rand(10, 4)  # Wrong input dim
        c_wrong = torch.rand(10, 2)

        # This should fail gracefully during training
        print("✓ Dimension mismatch handling test setup complete")
    except Exception as e:
        print(f"Note: Dimension mismatch test: {str(e)}")

    return True


def visualize_results(trainer, n_samples=100):
    """Visualize some generated samples."""
    print("\nGenerating visualization...")

    try:
        # Generate samples with different contexts
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Generate samples
        contexts = torch.randn(n_samples, trainer.condition_dim)
        samples = trainer.generate_solutions(contexts=contexts, num_samples=n_samples)

        # Plot first two dimensions
        axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=20)
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        axes[0].set_title('Generated Samples (First 2 Dimensions)')
        axes[0].grid(True, alpha=0.3)

        # Plot distribution
        axes[1].hist(samples.flatten(), bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Generated Values')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(trainer.save_dir, trainer.timestamp, 'test_visualization.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✓ Visualization saved to {save_path}")
        return True
    except Exception as e:
        print(f"✗ Visualization failed: {str(e)}")
        return False


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Running DDIM Implementation Tests")
    print("=" * 60)

    tests = [
        # ("Cosine Beta Schedule", test_cosine_beta_schedule),
        # ("Conditional MLP UNet", test_conditional_mlp_unet),
        # ("Conditional DDIM", test_conditional_ddim),
        ("Pareto DDIM Trainer", test_pareto_ddim_trainer),
        ("Gradient Monitoring", test_gradient_monitoring),
        ("Edge Cases", test_edge_cases)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {str(e)}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    # Run visualization if trainer test passed
    if results.get("Pareto DDIM Trainer", False):
        print("\nRunning visualization test...")
        trainer = ParetoDDIMTrainer(
            input_dim=4,
            condition_dim=2,
            hidden_dim=32,
            epochs=10,
            trainer_id="visualization_test"
        )
        # Quick training for visualization
        X = torch.rand(100, 4)
        c = torch.randn(100, 2)
        trainer.train(X, c)
        visualize_results(trainer)

    return passed == total


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run all tests
    success = run_all_tests()