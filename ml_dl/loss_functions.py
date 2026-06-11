# =============================================================================
# loss_functions.py
# RANDAL PINN — Schiesser/Randall Glucose Tolerance Model
# Reference: Schiesser (2014), Ch. 2 — ncase=2
# =============================================================================

import torch
import torch.nn as nn

# ── Schiesser / Randall Model Parameters ─────────────────────────────────────
# Units: mg glucose or insulin / 100 ml extracellular fluid (mGml / mIml)
#        time in hours
# Source: Schiesser (2014), Chapter 2, Listing 2.1

Ex  = 15000.0   # extracellular space (ml)
Cg  = 150.0     # glucose capacitance = Ex/100
Ci  = 150.0     # insulin capacitance = Ex/100
Q   = 8400.0    # liver release of glucose (mG/hr)
Gt  = 80000.0   # glucose infusion rate (mG/hr) — ncase=2: with infusion
Dd  = 24.7      # first-order glucose loss (mG/hr/mGml)
Gg  = 13.9      # controlled glucose loss (mG/hr/mGml/mIml)
Gk  = 250.0     # renal threshold (mGml)
Mu  = 72.0      # renal loss rate (mG/hr/mGml)
G0  = 51.0      # pancreas threshold (mGml)
Bb  = 14.3      # insulin release rate — normal sensitivity (ncase=2)
Aa  = 76.0      # first-order insulin clearance rate (mI/hr/mIml)

# Initial conditions
G_init = 81.14  # mGml
I_init = 5.671  # mIml


def randall_residuals(
    t_n: torch.Tensor,
    G_n_pred: torch.Tensor,
    I_n_pred: torch.Tensor,
    G_scale: torch.Tensor,
    I_scale: torch.Tensor,
    t_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PINN physics residuals for the Schiesser/Randall 2-ODE system.

    Physical ODEs (Schiesser 2014, Ch. 2, Listing 2.2):
        Cg * dG/dt = Q + In(t) - Gg*I*G - Dd*G              [G < Gk]
        Cg * dG/dt = Q + In(t) - Gg*I*G - Dd*G - Mu*(G-Gk)  [G >= Gk]
        Ci * dI/dt = -Aa*I                                   [G < G0]
        Ci * dI/dt = -Aa*I + Bb*(G - G0)                    [G >= G0]

    Glucose infusion: In(t) = Gt  for t <= 0.5 h, else 0  (ncase=2)

    Parameters
    ----------
    t_n       : (N,1) normalised time, requires_grad=True
    G_n_pred  : (N,1) normalised glucose prediction
    I_n_pred  : (N,1) normalised insulin prediction
    G_scale   : scalar tensor — mean glucose used for normalisation
    I_scale   : scalar tensor — mean insulin used for normalisation
    t_max     : scalar tensor — max time (12 h) used for normalisation

    Returns
    -------
    res_G, res_I : (N,1) ODE residuals in normalised space
    """
    # Un-normalise to physical units
    G_pred = G_n_pred * G_scale   # mGml
    I_pred = I_n_pred * I_scale   # mIml
    t_h    = t_n * t_max          # hours

    # Glucose infusion function
    In = torch.where(
        t_h <= 0.5,
        torch.tensor(Gt, dtype=torch.float32, device=t_n.device),
        torch.tensor(0.0,                     device=t_n.device),
    )

    # Automatic differentiation w.r.t. normalised time
    dGdt_n = torch.autograd.grad(
        G_n_pred, t_n,
        grad_outputs=torch.ones_like(G_n_pred),
        create_graph=True,
    )[0]
    dIdt_n = torch.autograd.grad(
        I_n_pred, t_n,
        grad_outputs=torch.ones_like(I_n_pred),
        create_graph=True,
    )[0]

    # Glucose RHS — piecewise on renal threshold Gk
    rhs_G_base = Q + In - Gg * I_pred * G_pred - Dd * G_pred
    renal       = Mu * torch.clamp(G_pred - Gk, min=0.0)
    rhs_G       = rhs_G_base - renal   # mG/hr

    # Insulin RHS — piecewise on pancreas threshold G0
    rhs_I = -Aa * I_pred + Bb * torch.clamp(G_pred - G0, min=0.0)  # mI/hr

    # Chain-rule residuals (normalised space):
    #   d(G_n)/d(t_n) = (t_max / G_scale) * dG/dt  =>
    #   residual = d(G_n)/d(t_n) - (t_max / (Cg * G_scale)) * rhs_G
    res_G = dGdt_n - (t_max / (Cg * G_scale)) * rhs_G
    res_I = dIdt_n - (t_max / (Ci * I_scale)) * rhs_I

    return res_G, res_I


def compute_loss(
    model: nn.Module,
    t_n: torch.Tensor,
    G_n: torch.Tensor,
    I_n: torch.Tensor,
    t_n_phys: torch.Tensor,
    lambda_phys: float,
    G_scale: torch.Tensor,
    I_scale: torch.Tensor,
    t_max: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total PINN loss:
        L = L_data  +  lambda_phys * L_physics  +  0.05 * L_IC

    Spike-region upweighting: t < 2 h (glucose peak) weighted 16×.

    Parameters
    ----------
    model       : RandallPINN instance
    t_n         : (N,1) normalised training time, on device
    G_n         : (N,1) normalised glucose training data
    I_n         : (N,1) normalised insulin training data
    t_n_phys    : (M,1) normalised collocation points for physics loss
    lambda_phys : float, current physics loss weight
    G_scale     : normalisation scalar (mean glucose)
    I_scale     : normalisation scalar (mean insulin)
    t_max       : normalisation scalar (12 h)
    device      : 'cuda' or 'cpu'

    Returns
    -------
    loss, L_data, L_phys, L_ic : scalar tensors
    """
    # ── Data loss ────────────────────────────────────────────────────────────
    out      = model(t_n)
    G_pred_n = out[:, 0:1]
    I_pred_n = out[:, 1:2]

    # Upweight spike region (t < 2 h normalised = 2/t_max)
    spike_mask = (t_n < (2.0 / t_max.item())).float().to(device)
    w = 1.0 + 15.0 * spike_mask   # 16× in spike region, 1× elsewhere

    L_data = (w * (G_pred_n - G_n) ** 2).mean() + \
             (w * (I_pred_n - I_n) ** 2).mean()

    # ── Physics loss ─────────────────────────────────────────────────────────
    t_p   = t_n_phys.requires_grad_(True)
    out_p = model(t_p)
    G_p   = out_p[:, 0:1]
    I_p   = out_p[:, 1:2]

    res_G, res_I = randall_residuals(t_p, G_p, I_p, G_scale, I_scale, t_max)
    L_phys = res_G.pow(2).mean() + res_I.pow(2).mean()

    # ── Initial condition loss ────────────────────────────────────────────────
    t_ic   = torch.zeros(1, 1, dtype=torch.float32).to(device)
    out_ic = model(t_ic)
    G_ic   = out_ic[0, 0]
    I_ic   = out_ic[0, 1]
    L_ic   = (G_ic - G_init / G_scale.item()) ** 2 + \
             (I_ic - I_init / I_scale.item()) ** 2

    loss = L_data + lambda_phys * L_phys + 0.05 * L_ic
    return loss, L_data, L_phys, L_ic


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from model_architecture import build_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = build_model(device=device)

    # Dummy tensors
    t_n     = torch.linspace(0, 1, 49).unsqueeze(1).to(device)
    G_n     = torch.ones_like(t_n)
    I_n     = torch.ones_like(t_n)
    t_phys  = torch.linspace(0, 1, 100).unsqueeze(1).to(device)
    G_scale = torch.tensor(100.0)
    I_scale = torch.tensor(9.0)
    t_max   = torch.tensor(12.0)

    loss, Ld, Lp, Lic = compute_loss(
        model, t_n, G_n, I_n, t_phys,
        lambda_phys=0.1,
        G_scale=G_scale, I_scale=I_scale, t_max=t_max,
        device=device,
    )
    print(f'Test loss={loss.item():.5f} | L_data={Ld.item():.5f} '
          f'| L_phys={Lp.item():.5f} | L_ic={Lic.item():.5f}')
