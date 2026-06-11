# =============================================================================
# model_architecture.py
# RANDAL PINN — Schiesser/Randall Glucose Tolerance Model
# Reference: Schiesser (2014), Ch. 2 — ncase=2
# =============================================================================

import torch
import torch.nn as nn


class RandallPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Schiesser/Randall 2-ODE model.

    Architecture:
        Input  : t_norm ∈ [0, 1]  (normalised time)
        Hidden : `hidden_layers` fully-connected layers, each with `neurons`
                 units and Tanh activation
        Output : [G_norm, I_norm]  (normalised glucose and insulin)

    Parameters
    ----------
    hidden_layers : int
        Number of hidden layers (default: 6)
    neurons : int
        Neurons per hidden layer (default: 128)
    """

    def __init__(self, hidden_layers: int = 6, neurons: int = 128):
        super().__init__()

        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, 2))   # outputs: G_norm, I_norm

        self.net = nn.Sequential(*layers)

        # Xavier initialisation for stable training
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        t : torch.Tensor of shape (N, 1)
            Normalised time values in [0, 1]

        Returns
        -------
        torch.Tensor of shape (N, 2)
            Column 0 : G_norm  (normalised glucose)
            Column 1 : I_norm  (normalised insulin)
        """
        return self.net(t)


def build_model(hidden_layers: int = 6,
                neurons: int = 128,
                device: str = 'cpu') -> RandallPINN:
    """
    Instantiate RandallPINN, move to device, and print a parameter summary.

    Parameters
    ----------
    hidden_layers : int
        Number of hidden layers
    neurons : int
        Neurons per layer
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    RandallPINN
        Model on the specified device
    """
    model = RandallPINN(hidden_layers=hidden_layers, neurons=neurons).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'RandallPINN  |  {hidden_layers} hidden layers x {neurons} neurons  '
          f'|  Tanh  |  {n_params:,} parameters  |  device={device}')
    return model


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = build_model(device=device)
    t_test = torch.linspace(0, 1, 10).unsqueeze(1).to(device)
    out    = model(t_test)
    print(f'Test forward pass — input shape: {t_test.shape}, '
          f'output shape: {out.shape}')
    print(model)
