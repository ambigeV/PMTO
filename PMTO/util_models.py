"""
A simple FC Pareto Set model.
"""

import torch
import torch.nn as nn
# import gpytorch
"""
Utility functions for computing gradients of GP models
"""


def compute_gp_gradients(model, likelihood, x):
    """
    Compute gradients of GP mean and std w.r.t input x

    Args:
        model: Trained GPyTorch model
        likelihood: GPyTorch likelihood
        x: Input tensor with shape [batch_size, input_dim]

    Returns:
        mean_grad: Gradients of mean with shape [batch_size, input_dim]
        std_grad: Gradients of std with shape [batch_size, input_dim]
    """
    # Ensure x requires gradients
    x = x.clone().detach().requires_grad_(True)

    # Set model to eval mode
    model.eval()
    likelihood.eval()

    # Forward pass through the model
    with torch.enable_grad():
        output = model(x)
        mean = output.mean
        variance = output.variance
        std = variance.sqrt()

        # Compute gradients of mean (for each batch element)
        mean_grads = []
        for i in range(x.shape[0]):  # For each batch element
            model.zero_grad()
            x.grad = None
            mean_i = mean[i]  # Get scalar for single batch element
            mean_i.backward(retain_graph=True)
            mean_grads.append(x.grad[i].clone())  # Get gradient for this batch element

        # Stack to get [batch_size, input_dim]
        mean_grad = torch.stack(mean_grads, dim=0)

        # Compute gradients of std (for each batch element)
        std_grads = []
        for i in range(x.shape[0]):
            model.zero_grad()
            x.grad = None
            std_i = std[i]
            std_i.backward(retain_graph=True)
            std_grads.append(x.grad[i].clone())

        # Stack to get [batch_size, input_dim]
        std_grad = torch.stack(std_grads, dim=0)

    return mean_grad, std_grad


class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        self.fc1 = nn.Linear(self.n_obj, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_dim)

    def forward(self, pref):
        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        x = torch.sigmoid(x)

        return x
