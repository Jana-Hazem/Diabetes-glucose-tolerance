import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import zipfile

# ─────────────────────────────────────────────
# 1. Load & Normalise Data
# ─────────────────────────────────────────────
df     = pd.read_csv('RK45_Results.csv')
t_data = torch.tensor(df['t'].values, dtype=torch.float32).unsqueeze(1)
G_true = torch.tensor(df['G'].values, dtype=torch.float32).unsqueeze(1)
I_true = torch.tensor(df['I'].values, dtype=torch.float32).unsqueeze(1)

t_max   = t_data.max()
t_norm  = t_data / t_max
G_scale = G_true.mean()
I_scale = I_true.mean()
G_norm  = G_true / G_scale
I_norm  = I_true / I_scale

print(f"t_max   = {t_max.item():.2f} hours")
print(f"G_scale = {G_scale.item():.4f} mg/dL")
print(f"I_scale = {I_scale.item():.4f} µU/mL")
print(f"G_norm range: {G_norm.min().item():.3f} to {G_norm.max().item():.3f}")
print(f"I_norm range: {I_norm.min().item():.3f} to {I_norm.max().item():.3f}")

# ─────────────────────────────────────────────
# 2. Bergman Minimal Model Parameters
# ─────────────────────────────────────────────
p1    = 0.028
p3    = 5.035e-5
Gb    = 81.14
n     = 0.09
gamma = 0.004
h     = 80.0

# ─────────────────────────────────────────────
# 3. Physics Residuals
# ─────────────────────────────────────────────
def bergman_residuals(t, G_norm_pred, I_norm_pred, G_scale, I_scale, t_max):
    G_pred = G_norm_pred * G_scale
    I_pred = I_norm_pred * I_scale

    dGdt_norm = torch.autograd.grad(
        G_norm_pred, t,
        grad_outputs=torch.ones_like(G_norm_pred),
        create_graph=True)[0]

    dIdt_norm = torch.autograd.grad(
        I_norm_pred, t,
        grad_outputs=torch.ones_like(I_norm_pred),
        create_graph=True)[0]

    rhs_G = -p1 * G_pred - p3 * I_pred * G_pred + p1 * Gb
    rhs_I =  gamma * torch.clamp(G_pred - h, min=0.0) - n * I_pred

    res_G = dGdt_norm - (t_max / G_scale) * rhs_G
    res_I = dIdt_norm - (t_max / I_scale) * rhs_I

    return res_G, res_I

# ─────────────────────────────────────────────
# 4. Network Architecture
# ─────────────────────────────────────────────
class BergmanPINN(nn.Module):
    def __init__(self, hidden_layers=6, hidden_size=128):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

# ─────────────────────────────────────────────
# 5. Loss Functions
# ─────────────────────────────────────────────
def data_loss_weighted(model, t_norm, G_norm, I_norm, t_max):
    out    = model(t_norm)
    G_pred = out[:, 0:1]
    I_pred = out[:, 1:2]
    t_raw  = t_norm * t_max
    weight = 1.0 + 15.0 * torch.exp(-3.0 * t_raw)
    loss_G = torch.mean(weight * (G_pred - G_norm)**2)
    loss_I = torch.mean(weight * (I_pred - I_norm)**2)
    return loss_G + loss_I


def physics_loss(model, t_colloc, G_scale, I_scale, t_max):
    t_c = t_colloc.clone().requires_grad_(True)
    out = model(t_c)
    G_n = out[:, 0:1]
    I_n = out[:, 1:2]
    res_G, res_I = bergman_residuals(t_c, G_n, I_n, G_scale, I_scale, t_max)
    return torch.mean(res_G**2) + torch.mean(res_I**2)


def ic_loss(model, G_scale, I_scale):
    t0  = torch.tensor([[0.0]])
    out = model(t0)
    G0  = out[0, 0] * G_scale
    I0  = out[0, 1] * I_scale
    return ((G0 -  81.14)**2 / G_scale**2 +
            (I0 -   5.671)**2 / I_scale**2)

# ─────────────────────────────────────────────
# 6. Training
# ─────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)

t_colloc_uniform = torch.linspace(0, 1, 300).unsqueeze(1)
t_colloc_spike   = torch.linspace(0, 0.2, 100).unsqueeze(1)
t_colloc         = torch.cat([t_colloc_uniform, t_colloc_spike], dim=0)

model     = BergmanPINN(hidden_layers=6, hidden_size=128)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=500, factor=0.5, min_lr=1e-6
)

history   = {'total': [], 'data': [], 'physics': []}
best_loss = float('inf')
EPOCHS    = 10000

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # Phase 1 (0–2000):  data + IC only
    # Phase 2 (2000+):   slowly add physics
    if epoch < 2000:
        lam = 0.0
    else:
        lam = min(0.05, 0.05 * (epoch - 2000) / 3000)

    Ld  = data_loss_weighted(model, t_norm, G_norm, I_norm, t_max)
    Lic = ic_loss(model, G_scale, I_scale)

    if lam > 0:
        Lp   = physics_loss(model, t_colloc, G_scale, I_scale, t_max)
        loss = Ld + lam * Lp + 0.05 * Lic
    else:
        Lp   = torch.tensor(0.0)
        loss = Ld + 0.05 * Lic

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)

    history['total'].append(loss.item())
    history['data'].append(Ld.item())
    history['physics'].append(Lp.item())

    if Ld.item() < best_loss:
        best_loss = Ld.item()
        torch.save(model.state_dict(), "checkpoints/best_model.pt")

    if (epoch + 1) % 1000 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:5d} | Data: {Ld.item():.5f} | "
              f"Physics: {Lp.item():.5f} | λ={lam:.4f} | LR={lr_now:.2e}")

print(f"\nDone! Best data loss: {best_loss:.5f}")

# ─────────────────────────────────────────────
# 7. Loss Curves Plot
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(history['data'],    label='Data loss',    color='orange')
ax.semilogy(history['physics'], label='Physics loss', color='green')
ax.semilogy(history['total'],   label='Total loss',   color='steelblue')
ax.axvline(x=2000, color='gray', linestyle='--', alpha=0.6, label='Physics starts')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('PINN training loss')
ax.legend()
plt.tight_layout()
plt.savefig("ml_loss_curves.png", dpi=150)
plt.show()
print("Saved: ml_loss_curves.png")

# ─────────────────────────────────────────────
# 8. Evaluation & Metrics
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()
with torch.no_grad():
    out    = model(t_norm)
    G_pred = (out[:, 0:1] * G_scale).numpy()
    I_pred = (out[:, 1:2] * I_scale).numpy()

t_np  = t_data.numpy().flatten()
G_ref = G_true.numpy().flatten()
I_ref = I_true.numpy().flatten()

rmse_G = np.sqrt(np.mean((G_pred.flatten() - G_ref)**2))
rmse_I = np.sqrt(np.mean((I_pred.flatten() - I_ref)**2))
mae_G  = np.mean(np.abs(G_pred.flatten() - G_ref))
mae_I  = np.mean(np.abs(I_pred.flatten() - I_ref))

print(f"RMSE G: {rmse_G:.4f} mg/dL  |  RMSE I: {rmse_I:.4f} µU/mL")
print(f"MAE  G: {mae_G:.4f}  mg/dL  |  MAE  I: {mae_I:.4f} µU/mL")

# ─────────────────────────────────────────────
# 9. G(t) and I(t) Comparison Plots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(t_np, G_ref,            'b-o', ms=4, label='RK45 reference')
axes[0].plot(t_np, G_pred.flatten(), 'r--', lw=2, label='PINN prediction')
axes[0].set_xlabel('t (hours)')
axes[0].set_ylabel('G (mg/dL)')
axes[0].set_title('Blood glucose G(t)')
axes[0].legend()
axes[1].plot(t_np, I_ref,            'b-o', ms=4, label='RK45 reference')
axes[1].plot(t_np, I_pred.flatten(), 'r--', lw=2, label='PINN prediction')
axes[1].set_xlabel('t (hours)')
axes[1].set_ylabel('I (µU/mL)')
axes[1].set_title('Plasma insulin I(t)')
axes[1].legend()
plt.tight_layout()
plt.savefig("ml_GI_comparison.png", dpi=150)
plt.show()
print("Saved: ml_GI_comparison.png")

# ─────────────────────────────────────────────
# 10. Save All Outputs & Zip
# ─────────────────────────────────────────────
os.makedirs("PINN_outputs", exist_ok=True)

# Loss curves
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(history['data'],    label='Data loss',    color='orange')
ax.semilogy(history['physics'], label='Physics loss', color='green')
ax.semilogy(history['total'],   label='Total loss',   color='steelblue')
ax.axvline(x=2000, color='gray', linestyle='--', alpha=0.6, label='Physics starts')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('PINN training loss')
ax.legend()
plt.tight_layout()
plt.savefig("PINN_outputs/ml_loss_curves.png", dpi=150)
plt.close()

# G/I comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(t_np, G_ref,            'b-o', ms=4, label='RK45 reference')
axes[0].plot(t_np, G_pred.flatten(), 'r--', lw=2, label='PINN prediction')
axes[0].set_xlabel('t (hours)')
axes[0].set_ylabel('G (mg/dL)')
axes[0].set_title('Blood glucose G(t)')
axes[0].legend()
axes[1].plot(t_np, I_ref,            'b-o', ms=4, label='RK45 reference')
axes[1].plot(t_np, I_pred.flatten(), 'r--', lw=2, label='PINN prediction')
axes[1].set_xlabel('t (hours)')
axes[1].set_ylabel('I (µU/mL)')
axes[1].set_title('Plasma insulin I(t)')
axes[1].legend()
plt.tight_layout()
plt.savefig("PINN_outputs/ml_GI_comparison.png", dpi=150)
plt.close()

# Model checkpoint
shutil.copy("checkpoints/best_model.pt", "PINN_outputs/best_model.pt")

# Metrics CSV
metrics = pd.DataFrame({
    'method': ['PINN'],
    'RMSE_G': [rmse_G],
    'RMSE_I': [rmse_I],
    'MAE_G':  [mae_G],
    'MAE_I':  [mae_I],
})
metrics.to_csv("PINN_outputs/ml_metrics.csv", index=False)
metrics.to_csv("ml_metrics.csv", index=False)
print("Saved: ml_metrics.csv")
print(metrics.to_string(index=False))

# Predictions CSV
predictions = pd.DataFrame({
    't':      t_np,
    'G_RK45': G_ref,
    'G_PINN': G_pred.flatten(),
    'I_RK45': I_ref,
    'I_PINN': I_pred.flatten(),
})
predictions.to_csv("PINN_outputs/ml_predictions.csv", index=False)

# Zip
zip_path = "PINN_outputs.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for filename in os.listdir("PINN_outputs"):
        zipf.write(os.path.join("PINN_outputs", filename), filename)

print("\nContents of ZIP:")
for filename in os.listdir("PINN_outputs"):
    size = os.path.getsize(f"PINN_outputs/{filename}")
    print(f"  {filename:35s} {size/1024:.1f} KB")

print("\nAll outputs saved to PINN_outputs/ and zipped as PINN_outputs.zip")
