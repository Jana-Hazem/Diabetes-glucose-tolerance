"""
model_architecture.py
---------------------
Bergman Minimal Model – Physics-Informed Neural Network (PINN)

Network: fully-connected MLP with Tanh activations.
Input  : normalised time  t_norm ∈ [0, 1]
Outputs: (G_norm, I_norm)  — blood glucose and plasma insulin,
         both normalised to unit scale (divide by their dataset mean).
"""

import torch
import torch.nn as nn


class BergmanPINN(nn.Module):
    """
    Multi-layer perceptron that approximates the solution of the
    Bergman Minimal ODE system.

    Parameters
    ----------
    hidden_layers : int
        Number of hidden layers (default 6).
    hidden_size   : int
        Neurons per hidden layer (default 128).
    """

    def __init__(self, hidden_layers: int = 6, hidden_size: int = 128):
        super().__init__()

        layers = [nn.Linear(1, hidden_size), nn.Tanh()]

        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]

        # Two outputs: G_norm and I_norm
        layers += [nn.Linear(hidden_size, 2)]

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Normalised time values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (N, 2)
            Column 0 → G_norm predictions
            Column 1 → I_norm predictions
        """
        return self.net(t)


# ---------------------------------------------------------------------------
# ODE residuals (Bergman Minimal Model)
# ---------------------------------------------------------------------------

def bergman_residuals(
    t:           torch.Tensor,
    G_norm_pred: torch.Tensor,
    I_norm_pred: torch.Tensor,
    G_scale:     torch.Tensor,
    I_scale:     torch.Tensor,
    t_max:       torch.Tensor,
    *,
    p1:    float = 0.028,
    p3:    float = 5.035e-5,
    Gb:    float = 81.14,
    n:     float = 0.09,
    gamma: float = 0.004,
    h:     float = 80.0,
):
    """
    Compute ODE residuals for the Bergman Minimal Model using automatic
    differentiation.

    The ODE system (in physical units) is:
        dG/dt = -p1*G - p3*I*G + p1*Gb
        dI/dt =  gamma * max(G - h, 0) - n*I

    Chain-rule correction converts normalised derivatives back to physical:
        d(G_norm)/d(t_norm) = (t_max / G_scale) * dG/dt

    Parameters
    ----------
    t            : collocation points (requires_grad=True), shape (N, 1)
    G_norm_pred  : network output for G, shape (N, 1)
    I_norm_pred  : network output for I, shape (N, 1)
    G_scale      : scalar – mean of G in the training set (mg/dL)
    I_scale      : scalar – mean of I in the training set (µU/mL)
    t_max        : scalar – maximum time value in the dataset (hours)
    p1, p3, Gb, n, gamma, h : Bergman model parameters

    Returns
    -------
    res_G : torch.Tensor  – glucose ODE residual,  shape (N, 1)
    res_I : torch.Tensor  – insulin ODE residual,  shape (N, 1)
    """
    G_pred = G_norm_pred * G_scale
    I_pred = I_norm_pred * I_scale

    dGdt_norm = torch.autograd.grad(
        G_norm_pred, t,
        grad_outputs=torch.ones_like(G_norm_pred),
        create_graph=True,
    )[0]

    dIdt_norm = torch.autograd.grad(
        I_norm_pred, t,
        grad_outputs=torch.ones_like(I_norm_pred),
        create_graph=True,
    )[0]

    rhs_G = -p1 * G_pred - p3 * I_pred * G_pred + p1 * Gb
    rhs_I = gamma * torch.clamp(G_pred - h, min=0.0) - n * I_pred

    res_G = dGdt_norm - (t_max / G_scale) * rhs_G
    res_I = dIdt_norm - (t_max / I_scale) * rhs_I

    return res_G, res_I
