"""
loss_functions.py
-----------------
Loss terms for the Bergman PINN training loop.

Three components
----------------
1. data_loss_weighted  – MSE against RK45 reference, with exponential
                         up-weighting in the early spike region.
2. physics_loss        – Mean-squared ODE residuals at collocation points.
3. ic_loss             – Penalises deviation from known initial conditions.
"""

import torch
from model_architecture import BergmanPINN, bergman_residuals


# ---------------------------------------------------------------------------
# 1. Data fidelity loss (weighted MSE)
# ---------------------------------------------------------------------------

def data_loss_weighted(
    model:   BergmanPINN,
    t_norm:  torch.Tensor,
    G_norm:  torch.Tensor,
    I_norm:  torch.Tensor,
    t_max:   torch.Tensor,
    *,
    spike_weight: float = 15.0,
    spike_decay:  float = 3.0,
) -> torch.Tensor:
    """
    Weighted MSE between network predictions and the RK45 reference data.

    The weight function
        w(t) = 1 + spike_weight * exp(-spike_decay * t_raw)
    assigns higher importance to the early-time glucose spike (first ~2 h).

    Parameters
    ----------
    model        : trained / in-training BergmanPINN instance
    t_norm       : normalised time points, shape (N, 1)
    G_norm       : normalised glucose reference, shape (N, 1)
    I_norm       : normalised insulin reference, shape (N, 1)
    t_max        : scalar – used to recover physical time for weighting
    spike_weight : multiplier at t=0 (default 15)
    spike_decay  : exponential decay rate  (default 3)

    Returns
    -------
    torch.Tensor (scalar) – combined weighted data loss for G and I
    """
    out    = model(t_norm)
    G_pred = out[:, 0:1]
    I_pred = out[:, 1:2]

    t_raw  = t_norm * t_max
    weight = 1.0 + spike_weight * torch.exp(-spike_decay * t_raw)

    loss_G = torch.mean(weight * (G_pred - G_norm) ** 2)
    loss_I = torch.mean(weight * (I_pred - I_norm) ** 2)
    return loss_G + loss_I


# ---------------------------------------------------------------------------
# 2. Physics (ODE residual) loss
# ---------------------------------------------------------------------------

def physics_loss(
    model:    BergmanPINN,
    t_colloc: torch.Tensor,
    G_scale:  torch.Tensor,
    I_scale:  torch.Tensor,
    t_max:    torch.Tensor,
) -> torch.Tensor:
    """
    Mean-squared ODE residuals evaluated at collocation points.

    Parameters
    ----------
    model     : BergmanPINN instance
    t_colloc  : collocation time points (will be cloned + grad enabled),
                shape (M, 1)
    G_scale   : mean glucose value used for normalisation (mg/dL)
    I_scale   : mean insulin value used for normalisation (µU/mL)
    t_max     : maximum time in the dataset (hours)

    Returns
    -------
    torch.Tensor (scalar) – sum of mean-squared residuals for G and I
    """
    t_c = t_colloc.clone().requires_grad_(True)
    out = model(t_c)
    G_n = out[:, 0:1]
    I_n = out[:, 1:2]

    res_G, res_I = bergman_residuals(t_c, G_n, I_n, G_scale, I_scale, t_max)
    return torch.mean(res_G ** 2) + torch.mean(res_I ** 2)


# ---------------------------------------------------------------------------
# 3. Initial-condition loss
# ---------------------------------------------------------------------------

def ic_loss(
    model:   BergmanPINN,
    G_scale: torch.Tensor,
    I_scale: torch.Tensor,
    *,
    G0: float = 81.14,
    I0: float = 5.671,
) -> torch.Tensor:
    """
    Penalises the network prediction at t=0 deviating from known ICs.

    Normalised form:  ((G_pred - G0) / G_scale)^2 + ((I_pred - I0) / I_scale)^2

    Parameters
    ----------
    model   : BergmanPINN instance
    G_scale : normalisation factor for G (mg/dL)
    I_scale : normalisation factor for I (µU/mL)
    G0      : initial blood glucose (mg/dL)  – default 81.14
    I0      : initial plasma insulin (µU/mL) – default 5.671

    Returns
    -------
    torch.Tensor (scalar) – IC loss
    """
    t0  = torch.tensor([[0.0]])
    out = model(t0)
    G_pred = out[0, 0] * G_scale
    I_pred = out[0, 1] * I_scale

    return (G_pred - G0) ** 2 / G_scale ** 2 + \
           (I_pred - I0) ** 2 / I_scale ** 2


# ---------------------------------------------------------------------------
# 4. Combined loss (convenience wrapper)
# ---------------------------------------------------------------------------

def total_loss(
    model:    BergmanPINN,
    t_norm:   torch.Tensor,
    G_norm:   torch.Tensor,
    I_norm:   torch.Tensor,
    t_colloc: torch.Tensor,
    G_scale:  torch.Tensor,
    I_scale:  torch.Tensor,
    t_max:    torch.Tensor,
    lam:      float,
    *,
    ic_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate all loss terms into a single scalar.

        L_total = L_data + lam * L_physics + ic_weight * L_ic

    Parameters
    ----------
    lam       : physics loss weight (0 during warm-up, ramps up afterwards)
    ic_weight : initial-condition loss weight (default 0.05)

    Returns
    -------
    (L_total, L_data, L_physics) – all as scalar tensors
    """
    Ld  = data_loss_weighted(model, t_norm, G_norm, I_norm, t_max)
    Lic = ic_loss(model, G_scale, I_scale)

    if lam > 0.0:
        Lp   = physics_loss(model, t_colloc, G_scale, I_scale, t_max)
        loss = Ld + lam * Lp + ic_weight * Lic
    else:
        Lp   = torch.tensor(0.0)
        loss = Ld + ic_weight * Lic

    return loss, Ld, Lp
