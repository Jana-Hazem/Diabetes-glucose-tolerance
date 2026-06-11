#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RANDAL PINN — Physics-Informed Neural Network
Schiesser / Randall Diabetes Glucose Tolerance Test Model

Reference: Schiesser, W.E. (2014), Differential Equation Analysis in
Biomedical Science and Engineering: ODE Applications with R, Wiley — Chapter 2.
Original model from: Randall, J.E. (1980), Microcomputers and Physiological
Simulation, Addison-Wesley, p. 69.

Trains a PINN on RK45 simulation data generated from the Schiesser/Randall
2-ODE glucose tolerance test model:

    Cg dG/dt = Q + In - Gg*I*G - Dd*G                  (G < Gk)
    Cg dG/dt = Q + In - Gg*I*G - Dd*G - Mu*(G-Gk)      (G >= Gk)
    Ci dI/dt = -Aa*I                                   (G < G0)
    Ci dI/dt = -Aa*I + Bb*(G - G0)                     (G >= G0)

Outputs:
    G(t) — extracellular glucose (mg glucose / 100 ml extracellular fluid)
    I(t) — extracellular insulin (mg insulin / 100 ml extracellular fluid)

Usage:
    python train.py --data RK45_Results.csv --outdir PINN_outputs --epochs 20000

Converted from a Colab notebook: Colab-only cells (file-upload widget,
`google.colab.files`, zip download) have been removed/replaced with plain
filesystem I/O so this runs as a normal script.
"""

import os
import time
import zipfile
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────
# Schiesser / Randall model parameters (Schiesser 2014, Ch. 2, Listing 2.1)
# Units: mg glucose or insulin / 100 ml extracellular fluid (mGml / mIml)
#        time in hours
# ──────────────────────────────────────────────────────────────────────────
PARAMS = dict(
    Ex=15000.0,   # extracellular space (ml)
    Cg=150.0,     # glucose capacitance = Ex/100
    Ci=150.0,     # insulin capacitance = Ex/100
    Q=8400.0,     # liver release of glucose (mG/hr)
    Gt=80000.0,   # glucose infusion rate (mG/hr) — ncase=2: with infusion
    Dd=24.7,      # first-order glucose loss (mG/hr/mGml)
    Gg=13.9,      # controlled glucose loss (mG/hr/mGml/mIml)
    Gk=250.0,     # renal threshold (mGml)
    Mu=72.0,      # renal loss rate (mG/hr/mGml)
    G0=51.0,      # pancreas threshold (mGml)
    Bb=14.3,      # insulin release rate — normal sensitivity (ncase=2)
    Aa=76.0,      # first-order insulin clearance rate (mI/hr/mIml)
)

# Initial conditions (from book / CSV)
G_INIT = 81.14  # mGml
I_INIT = 5.671  # mIml


# ──────────────────────────────────────────────────────────────────────────
# Network architecture
# ──────────────────────────────────────────────────────────────────────────
class RandallPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Schiesser/Randall 2-ODE model.
    Input : t_norm in [0, 1]
    Output: [G_norm, I_norm]
    """

    def __init__(self, hidden_layers=6, neurons=128):
        super().__init__()
        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, 2))  # outputs: G_norm, I_norm
        self.net = nn.Sequential(*layers)

        # Xavier initialisation
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


# ──────────────────────────────────────────────────────────────────────────
# ODE residuals
# ──────────────────────────────────────────────────────────────────────────
def randall_residuals(t_n, G_n_pred, I_n_pred, G_scale, I_scale, t_max, params):
    """
    Compute PINN residuals for the Schiesser/Randall 2-ODE glucose model.

    Physical ODEs (Schiesser 2014, Ch. 2, Listing 2.2):
      Cg * dG/dt = Q + In(t) - Gg*I*G - Dd*G              [G < Gk]
      Cg * dG/dt = Q + In(t) - Gg*I*G - Dd*G - Mu*(G-Gk)  [G >= Gk]
      Ci * dI/dt = -Aa*I                                  [G < G0]
      Ci * dI/dt = -Aa*I + Bb*(G - G0)                    [G >= G0]

    t_n : normalised time (0 -> 1), requires_grad=True
    G_n_pred, I_n_pred : normalised network predictions
    """
    Cg, Q, Gt = params["Cg"], params["Q"], params["Gt"]
    Dd, Gg, Gk, Mu = params["Dd"], params["Gg"], params["Gk"], params["Mu"]
    Ci, Aa, Bb, G0 = params["Ci"], params["Aa"], params["Bb"], params["G0"]

    # Un-normalise to physical units
    G_pred = G_n_pred * G_scale  # mGml
    I_pred = I_n_pred * I_scale  # mIml

    # Recover physical time (hours)
    t_h = t_n * t_max

    # Glucose infusion: In = Gt for t <= 0.5 h, else 0
    In = torch.where(
        t_h <= 0.5,
        torch.tensor(Gt, dtype=torch.float32, device=t_n.device),
        torch.tensor(0.0, device=t_n.device),
    )

    # Automatic differentiation w.r.t. normalised time
    dGdt_n = torch.autograd.grad(
        G_n_pred, t_n, grad_outputs=torch.ones_like(G_n_pred), create_graph=True
    )[0]
    dIdt_n = torch.autograd.grad(
        I_n_pred, t_n, grad_outputs=torch.ones_like(I_n_pred), create_graph=True
    )[0]

    # Glucose RHS (physical) — piecewise on Gk
    rhs_G_base = Q + In - Gg * I_pred * G_pred - Dd * G_pred
    renal = Mu * torch.clamp(G_pred - Gk, min=0.0)
    rhs_G = rhs_G_base - renal  # mG/hr

    # Insulin RHS (physical) — piecewise on G0
    rhs_I = -Aa * I_pred + Bb * torch.clamp(G_pred - G0, min=0.0)  # mI/hr

    # Chain-rule: d(G_n)/d(t_n) = (t_max / G_scale) * dG/dt  (physical)
    # => residual = d(G_n)/d(t_n) - (t_max / (Cg * G_scale)) * rhs_G
    res_G = dGdt_n - (t_max / (Cg * G_scale)) * rhs_G
    res_I = dIdt_n - (t_max / (Ci * I_scale)) * rhs_I

    return res_G, res_I


# ──────────────────────────────────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────────────────────────────────
def compute_loss(model, t_n, G_n, I_n, t_n_phys, lambda_phys,
                  G_scale, I_scale, t_max, params, device):
    """
    Total loss = L_data  (fit to RK45 data)
               + lambda * L_physics  (Schiesser/Randall ODE residuals)
               + 0.05 * L_IC  (initial conditions)

    Spike-region upweighting: t < 2 h (glucose peak) weighted 16x.
    """
    # ── Data loss ───────────────────────────────────────────────────────
    out = model(t_n)
    G_pred_n = out[:, 0:1]
    I_pred_n = out[:, 1:2]

    # Upweight spike region (t < 2/12 normalised)
    spike_mask = (t_n < (2.0 / t_max.item())).float().to(device)
    w = 1.0 + 15.0 * spike_mask  # 16x in spike region

    err_G = w * (G_pred_n - G_n) ** 2
    err_I = w * (I_pred_n - I_n) ** 2
    L_data = err_G.mean() + err_I.mean()

    # ── Physics loss ────────────────────────────────────────────────────
    t_p = t_n_phys.requires_grad_(True)
    out_p = model(t_p)
    G_p = out_p[:, 0:1]
    I_p = out_p[:, 1:2]

    res_G, res_I = randall_residuals(t_p, G_p, I_p, G_scale, I_scale, t_max, params)
    L_phys = res_G.pow(2).mean() + res_I.pow(2).mean()

    # ── Initial condition loss ──────────────────────────────────────────
    t_ic = torch.zeros(1, 1, dtype=torch.float32).to(device)
    out_ic = model(t_ic)
    G_ic = out_ic[0, 0]  # G(0) normalised prediction
    I_ic = out_ic[0, 1]  # I(0) normalised prediction
    L_ic = (G_ic - G_INIT / G_scale.item()) ** 2 + \
           (I_ic - I_INIT / I_scale.item()) ** 2

    loss = L_data + lambda_phys * L_phys + 0.05 * L_ic
    return loss, L_data, L_phys, L_ic


# ──────────────────────────────────────────────────────────────────────────
# Prediction utility
# ──────────────────────────────────────────────────────────────────────────
def predict(model, t_hours_list, t_max, G_scale, I_scale, device, verbose=True):
    """Predict G(t) and I(t) at arbitrary time points (in hours)."""
    model.eval()
    t_arr = np.array(t_hours_list, dtype=np.float32).reshape(-1, 1)
    t_tensor = torch.tensor(t_arr / t_max.item(), dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(t_tensor)
    G_out = (out[:, 0:1] * G_scale).cpu().numpy().flatten()
    I_out = (out[:, 1:2] * I_scale).cpu().numpy().flatten()
    if verbose:
        for t, g, i in zip(t_hours_list, G_out, I_out):
            print(f'  t={t:5.2f} h  ->  G={g:7.3f} mGml   I={i:7.4f} mIml')
    return G_out, I_out


# ──────────────────────────────────────────────────────────────────────────
# Main training / evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────
def main(args):
    print('PyTorch version:', torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    out_dir = args.outdir
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_model_randall.pt')

    # ── Load RK45 Data ────────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    print('Columns found:', df.columns.tolist())
    print(df.head())

    t_data = torch.tensor(df['t'].values, dtype=torch.float32).unsqueeze(1)
    G_true = torch.tensor(df['G'].values, dtype=torch.float32).unsqueeze(1)
    I_true = torch.tensor(df['I'].values, dtype=torch.float32).unsqueeze(1)

    # ── Normalisation ────────────────────────────────────────────────────
    t_max = t_data.max()       # 12 h
    t_norm = t_data / t_max    # normalised time in [0, 1]

    G_scale = G_true.mean()    # ~100 mGml — mean-normalise G
    I_scale = I_true.mean()    # ~9   mIml — mean-normalise I

    G_norm = G_true / G_scale
    I_norm = I_true / I_scale

    print(f'\nt_max   = {t_max.item():.2f} h')
    print(f'G_scale = {G_scale.item():.4f} mGml')
    print(f'I_scale = {I_scale.item():.4f} mIml')
    print(f'G_norm range: {G_norm.min().item():.3f} -> {G_norm.max().item():.3f}')
    print(f'I_norm range: {I_norm.min().item():.3f} -> {I_norm.max().item():.3f}')

    print('Schiesser / Randall Model parameters loaded (ncase=2).')
    for k, v in PARAMS.items():
        print(f'  {k}={v}')
    print(f'  G(0)={G_INIT} mGml,  I(0)={I_INIT} mIml')

    # Visualise the glucose infusion function In(t)
    t_vis = torch.linspace(0, 12, 300).unsqueeze(1)
    In_vis = torch.where(t_vis <= 0.5, torch.tensor(PARAMS["Gt"]), torch.tensor(0.0))
    plt.figure(figsize=(6, 3))
    plt.plot(t_vis.squeeze().numpy(), In_vis.squeeze().numpy(), color='steelblue')
    plt.xlabel('t (hours)')
    plt.ylabel('In(t) (mG/hr)')
    plt.title('Glucose Infusion Function In(t) — ncase=2')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'infusion_function.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Model ─────────────────────────────────────────────────────────────
    model = RandallPINN(hidden_layers=6, neurons=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('Model: 6 hidden layers x 128 neurons, tanh activation')
    print(f'Total trainable parameters: {n_params:,}')
    print(model)

    # ── Training setup ───────────────────────────────────────────────────
    EPOCHS = args.epochs
    LR = args.lr
    PHYS_RAMP_END = args.phys_ramp_end
    LAMBDA_MAX = args.lambda_max
    N_PHYS = args.n_phys

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6)

    # Data tensors -- move to device
    t_n_train = t_norm.to(device)
    G_n_train = G_norm.to(device)
    I_n_train = I_norm.to(device)

    # Physics collocation points: uniform over [0, 1]
    t_phys = torch.linspace(0, 1, N_PHYS).unsqueeze(1).to(device)

    history = {'loss': [], 'L_data': [], 'L_phys': [], 'L_ic': [], 'lr': []}
    best_loss = float('inf')
    lv = float('nan')

    for epoch in range(1, EPOCHS + 1):
        model.train()

        lam = 0.0 if epoch < PHYS_RAMP_END else \
            LAMBDA_MAX * min((epoch - PHYS_RAMP_END) / 1000.0, 1.0)

        optimizer.zero_grad()
        loss, Ld, Lp, Lic = compute_loss(
            model, t_n_train, G_n_train, I_n_train, t_phys, lam,
            G_scale, I_scale, t_max, PARAMS, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.detach())

        lv = loss.item()
        history['loss'].append(lv)
        history['L_data'].append(Ld.item())
        history['L_phys'].append(Lp.item())
        history['L_ic'].append(Lic.item())
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if lv < best_loss:
            best_loss = lv
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 500 == 0 or epoch == 1:
            print(f'Epoch {epoch:5d} | Loss={lv:.5f} | L_data={Ld:.5f} '
                  f'| L_phys={Lp:.5f} | L_ic={Lic:.5f} '
                  f'| lambda={lam:.4f} | lr={optimizer.param_groups[0]["lr"]:.2e}')

    print(f'\nTraining complete. Best loss: {best_loss:.5f}')
    model.load_state_dict(torch.load(ckpt_path))
    print('Best model loaded.')
    # Force save final epoch (better than early checkpoint in this case)
    torch.save(model.state_dict(), ckpt_path)
    print(f'Final model saved. Loss at epoch {EPOCHS}: {lv:.5f}')

    # ── Loss curves ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(history['loss'], color='navy', label='Total')
    axes[0].semilogy(history['L_data'], color='green', label='L_data', alpha=0.7)
    axes[0].semilogy(history['L_phys'], color='orange', label='L_phys', alpha=0.7)
    axes[0].semilogy(history['L_ic'], color='red', label='L_ic', alpha=0.7)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (log)')
    axes[0].set_title('Training Loss Curves'); axes[0].legend()

    axes[1].plot(history['lr'], color='purple')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('LR Schedule (ReduceLROnPlateau)')

    lam_curve = [0.0 if e < PHYS_RAMP_END else
                 LAMBDA_MAX * min((e - PHYS_RAMP_END) / 1000.0, 1.0)
                 for e in range(1, EPOCHS + 1)]
    axes[2].plot(lam_curve, color='darkorange')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('lambda')
    axes[2].set_title('Physics Loss Weight lambda Ramp')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: loss_curves.png')

    # ── Evaluation & metrics ─────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out_eval = model(t_n_train)
        G_pred = (out_eval[:, 0:1] * G_scale).cpu().numpy().flatten()
        I_pred = (out_eval[:, 1:2] * I_scale).cpu().numpy().flatten()

    G_rk45 = G_true.numpy().flatten()
    I_rk45 = I_true.numpy().flatten()
    t_hr = t_data.numpy().flatten()

    rmse_G = np.sqrt(np.mean((G_pred - G_rk45) ** 2))
    rmse_I = np.sqrt(np.mean((I_pred - I_rk45) ** 2))
    mae_G = np.mean(np.abs(G_pred - G_rk45))
    mae_I = np.mean(np.abs(I_pred - I_rk45))

    print('=== Evaluation Metrics (vs RK45 ground truth) ===')
    print(f'  RMSE  G(t) : {rmse_G:.4f} mGml')
    print(f'  RMSE  I(t) : {rmse_I:.6f} mIml')
    print(f'  MAE   G(t) : {mae_G:.4f} mGml')
    print(f'  MAE   I(t) : {mae_I:.6f} mIml')

    # ── G(t)/I(t) comparison plot ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(t_hr, G_rk45, 'k.-', label='RK45 (truth)', zorder=3)
    axes[0].plot(t_hr, G_pred, 'r-', label=f'PINN   RMSE={rmse_G:.2f}', lw=2)
    axes[0].set_xlabel('t (h)'); axes[0].set_ylabel('G(t) (mGml)')
    axes[0].set_title('Glucose G(t) — Schiesser/Randall Model')
    axes[0].axhline(PARAMS["Gk"], color='purple', ls='--', lw=1,
                     label=f'Gk={PARAMS["Gk"]} mGml (renal)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t_hr, I_rk45, 'k.-', label='RK45 (truth)', zorder=3)
    axes[1].plot(t_hr, I_pred, 'b-', label=f'PINN   RMSE={rmse_I:.4f}', lw=2)
    axes[1].set_xlabel('t (h)'); axes[1].set_ylabel('I(t) (mIml)')
    axes[1].set_title('Insulin I(t) — Schiesser/Randall Model')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('RANDAL PINN vs RK45 — Schiesser Ch.2 (ncase=2)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'GI_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: GI_comparison.png')

    # ── Residual error plots ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(t_hr, np.abs(G_pred - G_rk45), width=0.2, color='salmon', edgecolor='darkred')
    axes[0].set_xlabel('t (h)'); axes[0].set_ylabel('|G_PINN - G_RK45| (mGml)')
    axes[0].set_title('Pointwise Glucose Error')
    axes[0].grid(alpha=0.3)

    axes[1].bar(t_hr, np.abs(I_pred - I_rk45), width=0.2, color='skyblue', edgecolor='navy')
    axes[1].set_xlabel('t (h)'); axes[1].set_ylabel('|I_PINN - I_RK45| (mIml)')
    axes[1].set_title('Pointwise Insulin Error')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Absolute Prediction Errors — Schiesser/Randall PINN', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'residual_errors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: residual_errors.png')

    # ── Save metrics CSV ─────────────────────────────────────────────────
    metrics = {
        'metric': ['RMSE_G', 'RMSE_I', 'MAE_G', 'MAE_I', 'best_loss'],
        'value': [rmse_G, rmse_I, mae_G, mae_I, best_loss],
        'units': ['mGml', 'mIml', 'mGml', 'mIml', 'dimensionless'],
    }
    pd.DataFrame(metrics).to_csv(os.path.join(out_dir, 'metrics.csv'),
                                  index=False, float_format='%.6f')
    print('Saved: metrics.csv')
    print(pd.DataFrame(metrics).to_string(index=False))

    # ── Generate config.yaml ─────────────────────────────────────────────
    config = f"""# RANDAL PINN — Schiesser/Randall Glucose Tolerance Model
# Schiesser (2014), Ch. 2 — ncase=2 (normal Bb, with glucose infusion)

model:
  type: PINN
  ode_system: Schiesser-Randall-2ODE
  hidden_layers: 6
  neurons_per_layer: 128
  activation: tanh
  outputs: [G_norm, I_norm]

training:
  epochs: {EPOCHS}
  lr_initial: {LR}
  scheduler: ReduceLROnPlateau
  scheduler_patience: 500
  scheduler_factor: 0.5
  lambda_physics_max: {LAMBDA_MAX}
  physics_ramp_start_epoch: {PHYS_RAMP_END}
  lambda_ic: 0.05
  spike_upweight: 16
  spike_region_hours: 2.0
  n_collocation_points: {N_PHYS}
  best_loss: {best_loss:.6f}

parameters:  # Schiesser (2014) Ch.2, Listing 2.1 — ncase=2
  Ex: {PARAMS["Ex"]}
  Cg: {PARAMS["Cg"]}
  Ci: {PARAMS["Ci"]}
  Q: {PARAMS["Q"]}
  Gt: {PARAMS["Gt"]}
  Dd: {PARAMS["Dd"]}
  Gg: {PARAMS["Gg"]}
  Gk: {PARAMS["Gk"]}
  Mu: {PARAMS["Mu"]}
  G0: {PARAMS["G0"]}
  Bb: {PARAMS["Bb"]}
  Aa: {PARAMS["Aa"]}

initial_conditions:
  G0_mGml: {G_INIT}
  I0_mIml: {I_INIT}

normalisation:
  t_max_h: {t_max.item():.4f}
  G_scale_mGml: {G_scale.item():.4f}
  I_scale_mIml: {I_scale.item():.4f}

metrics:
  RMSE_G_mGml: {rmse_G:.4f}
  RMSE_I_mIml: {rmse_I:.6f}
  MAE_G_mGml:  {mae_G:.4f}
  MAE_I_mIml:  {mae_I:.6f}
"""

    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        f.write(config)
    print('Saved: config.yaml')

    # ── Key clinical time points ─────────────────────────────────────────
    print('── Key clinical time points ─────────────────────────────────────────')
    print('  (t=0: basal, t=0.25: mid-infusion, t=0.5: peak infusion,')
    print('   t=1: post-peak, t=2,4,8,12: return to equilibrium)')
    predict(model, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0],
            t_max, G_scale, I_scale, device)

    # ── Extrapolation beyond t=12h ───────────────────────────────────────
    print('── Extrapolation (beyond training window, t=12->24 h) ────────────────')
    predict(model, [13.0, 15.0, 18.0, 21.0, 24.0], t_max, G_scale, I_scale, device)

    # Visualise full trajectory including extrapolation
    t_full = np.linspace(0, 24, 800)
    model.eval()
    t_tensor_full = torch.tensor((t_full / t_max.item()).reshape(-1, 1),
                                  dtype=torch.float32).to(device)
    with torch.no_grad():
        out_full = model(t_tensor_full)
    G_full = (out_full[:, 0:1] * G_scale).cpu().numpy().flatten()
    I_full = (out_full[:, 1:2] * I_scale).cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax in axes:
        ax.axvspan(0, 12, alpha=0.06, color='green', label='Training region')
        ax.axvspan(12, 24, alpha=0.06, color='orange', label='Extrapolation')

    axes[0].plot(t_full, G_full, 'r-', lw=2, label='PINN prediction')
    axes[0].plot(t_hr, G_rk45, 'k.', ms=6, label='RK45 data', zorder=5)
    axes[0].axhline(G_INIT, color='gray', ls=':', lw=1, label=f'Basal G={G_INIT}')
    axes[0].set_xlabel('t (h)'); axes[0].set_ylabel('G(t) (mGml)')
    axes[0].set_title('G(t): Training + Extrapolation'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t_full, I_full, 'b-', lw=2, label='PINN prediction')
    axes[1].plot(t_hr, I_rk45, 'k.', ms=6, label='RK45 data', zorder=5)
    axes[1].axhline(I_INIT, color='gray', ls=':', lw=1, label=f'Basal I={I_INIT}')
    axes[1].set_xlabel('t (h)'); axes[1].set_ylabel('I(t) (mIml)')
    axes[1].set_title('I(t): Training + Extrapolation'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('PINN Extrapolation — Schiesser/Randall Model', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'extrapolation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: extrapolation.png')

    # ── Final summary ─────────────────────────────────────────────────────
    print('=' * 65)
    print('    RANDAL PINN (SCHIESSER/RANDALL) — FINAL RESULTS SUMMARY')
    print('=' * 65)

    print('\n--- ODE System (Schiesser 2014, Ch. 2 / Randall 1980) ---')
    print('  Cg*dG/dt = Q + In - Gg*I*G - Dd*G              [G < Gk]')
    print('  Cg*dG/dt = Q + In - Gg*I*G - Dd*G - Mu*(G-Gk)  [G >= Gk]')
    print('  Ci*dI/dt = -Aa*I                               [G < G0]')
    print('  Ci*dI/dt = -Aa*I + Bb*(G - G0)                 [G >= G0]')
    print(f'  Case: ncase=2 (normal Bb={PARAMS["Bb"]}, with glucose infusion Gt={PARAMS["Gt"]})')

    print('\n--- Model Architecture ---')
    print('  Type         : Physics-Informed Neural Network (PINN)')
    print('  Inputs       : 1 (t_norm)')
    print('  Outputs      : 2 (G_norm, I_norm)')
    print('  Hidden layers: 6')
    print('  Neurons/layer: 128')
    print('  Activation   : Tanh')
    print('  Parameters   :', sum(p.numel() for p in model.parameters()))

    print('\n--- Training Setup ---')
    print(f'  Optimizer    : Adam')
    print(f'  Epochs       : {EPOCHS}')
    print(f'  LR initial   : {LR}  (ReduceLROnPlateau)')
    print(f'  Loss         : L = L_data + lambda*L_physics + 0.05*L_IC')
    print(f'  lambda sched : 0 for first {PHYS_RAMP_END} epochs, ramp to {LAMBDA_MAX}')
    print(f'  Weighting    : 16x upweight on spike region (t < 2h)')

    print('\n--- Performance Metrics ---')
    print(f'  RMSE  G(t)   : {rmse_G:.4f} mGml')
    print(f'  RMSE  I(t)   : {rmse_I:.6f} mIml')
    print(f'  MAE   G(t)   : {mae_G:.4f} mGml')
    print(f'  MAE   I(t)   : {mae_I:.6f} mIml')
    print(f'  Best loss    : {best_loss:.5f}')

    print('\n--- Clinical Interpretation ---')
    mask_3h = t_full <= 3
    peak_G_idx = np.argmax(G_full[mask_3h])
    peak_G_val = G_full[mask_3h][peak_G_idx]
    peak_t = t_full[mask_3h][peak_G_idx]
    print(f'  Peak glucose  : {peak_G_val:.2f} mGml at t~={peak_t:.2f} h (postprandial spike)')
    print(f'  Renal threshold Gk={PARAMS["Gk"]} mGml: glucose stays below renal threshold')
    print(f'  Basal targets : G->{G_INIT} mGml, I->{I_INIT} mIml as t->infinity')
    print(f'  Pancreas sens : Bb={PARAMS["Bb"]} (normal, ncase=2)')
    print('=' * 65)

    # ── Dense predictions + extrapolation CSVs ───────────────────────────
    t_train_dense = np.linspace(0, 12, 500).tolist()
    model.eval()
    t_td = torch.tensor(np.array(t_train_dense).reshape(-1, 1) / t_max.item(),
                         dtype=torch.float32).to(device)
    with torch.no_grad():
        out_td = model(t_td)
    G_td = (out_td[:, 0:1] * G_scale).cpu().numpy().flatten()
    I_td = (out_td[:, 1:2] * I_scale).cpu().numpy().flatten()

    G_rk45_interp = np.interp(t_train_dense, t_hr, G_rk45)
    I_rk45_interp = np.interp(t_train_dense, t_hr, I_rk45)

    df_dense = pd.DataFrame({
        't_hours': t_train_dense,
        'G_RK45': G_rk45_interp,
        'G_PINN': G_td,
        'G_error': np.abs(G_td - G_rk45_interp),
        'I_RK45': I_rk45_interp,
        'I_PINN': I_td,
        'I_error': np.abs(I_td - I_rk45_interp),
        'region': 'training',
    })

    t_extrap = np.linspace(12, 24, 200).tolist()
    t_te = torch.tensor(np.array(t_extrap).reshape(-1, 1) / t_max.item(),
                         dtype=torch.float32).to(device)
    with torch.no_grad():
        out_te = model(t_te)
    G_te = (out_te[:, 0:1] * G_scale).cpu().numpy().flatten()
    I_te = (out_te[:, 1:2] * I_scale).cpu().numpy().flatten()

    df_extrap = pd.DataFrame({
        't_hours': t_extrap,
        'G_RK45': np.nan, 'G_PINN': G_te, 'G_error': np.nan,
        'I_RK45': np.nan, 'I_PINN': I_te, 'I_error': np.nan,
        'region': 'extrapolation',
    })

    df_all = pd.concat([df_dense, df_extrap], ignore_index=True)

    df_dense.to_csv(os.path.join(out_dir, 'dense_predictions.csv'), index=False, float_format='%.6f')
    df_extrap.to_csv(os.path.join(out_dir, 'extrapolation.csv'), index=False, float_format='%.6f')
    df_all.to_csv(os.path.join(out_dir, 'all_predictions.csv'), index=False, float_format='%.6f')

    print('Files saved:')
    for fname in ['dense_predictions.csv', 'extrapolation.csv', 'all_predictions.csv']:
        size = os.path.getsize(os.path.join(out_dir, fname))
        print(f'  {fname:40s}  {size/1024:.1f} KB')

    print(f'\nOverall RMSE (dense, training region):')
    print(f'  G: {np.sqrt(np.mean(df_dense["G_error"]**2)):.4f} mGml')
    print(f'  I: {np.sqrt(np.mean(df_dense["I_error"]**2)):.6f} mIml')

    # ── Inference timing (CPU benchmark) ─────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    t_n_cpu = t_n_train.cpu()
    G_scale_cpu = G_scale.cpu()
    I_scale_cpu = I_scale.cpu()

    cpu_model = RandallPINN(hidden_layers=6, neurons=128)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()

    # Warm-up
    with torch.no_grad():
        _ = cpu_model(t_n_cpu)

    N_RUNS = 100
    rows = []
    for i in range(N_RUNS):
        start = time.perf_counter()
        with torch.no_grad():
            out = cpu_model(t_n_cpu)
            _ = (out[:, 0:1] * G_scale_cpu).numpy()
            _ = (out[:, 1:2] * I_scale_cpu).numpy()
        end = time.perf_counter()
        rows.append({'run': i + 1,
                      'inference_time_s': end - start,
                      'inference_time_ms': (end - start) * 1000})

    df_timing = pd.DataFrame(rows)
    print(f'Inference timing over {N_RUNS} runs (CPU):')
    print(f'  Mean : {df_timing["inference_time_ms"].mean():.4f} ms')
    print(f'  Std  : {df_timing["inference_time_ms"].std():.4f} ms')
    print(f'  Min  : {df_timing["inference_time_ms"].min():.4f} ms')
    print(f'  Max  : {df_timing["inference_time_ms"].max():.4f} ms')

    df_timing.to_csv(os.path.join(out_dir, 'inference_time.csv'), index=False)
    print('Saved: inference_time.csv')

    # ── Package all outputs into a zip ───────────────────────────────────
    import shutil
    shutil.copy(ckpt_path, os.path.join(out_dir, 'best_model_randall.pt'))

    zip_path = os.path.join(out_dir, 'PINN_outputs_Schiesser.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in sorted(os.listdir(out_dir)):
            full_path = os.path.join(out_dir, filename)
            if os.path.isfile(full_path) and filename != os.path.basename(zip_path):
                zipf.write(full_path, filename)

    print('Contents of ZIP:')
    with zipfile.ZipFile(zip_path) as zipf:
        for info in zipf.infolist():
            print(f'  {info.filename:50s} {info.file_size/1024:.1f} KB')

    print(f'\nAll outputs written to: {os.path.abspath(out_dir)}')
    print(f'Zipped bundle: {os.path.abspath(zip_path)}')


def parse_args():
    p = argparse.ArgumentParser(description='Train RANDAL PINN (Schiesser/Randall glucose model)')
    p.add_argument('--data', type=str, default='RK45_Results.csv',
                    help='Path to RK45 simulation CSV with columns t, G, I')
    p.add_argument('--outdir', type=str, default='PINN_outputs',
                    help='Directory to write checkpoints, plots, and CSV outputs')
    p.add_argument('--epochs', type=int, default=20_000)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--phys-ramp-end', dest='phys_ramp_end', type=int, default=2000)
    p.add_argument('--lambda-max', dest='lambda_max', type=float, default=0.1)
    p.add_argument('--n-phys', dest='n_phys', type=int, default=500)
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
