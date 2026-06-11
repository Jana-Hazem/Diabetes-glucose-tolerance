import os
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

RK4_CSV    = os.path.join(BASE, "rk4_results.csv")
RK45_CSV   = os.path.join(BASE, "RK45_Results.csv")
PREDS_CSV  = os.path.join(BASE, "all_predictions.csv")
METRICS_CSV = os.path.join(BASE, "metrics.csv")
INFER_CSV  = os.path.join(BASE, "inference_time.csv")
DENSE_CSV  = os.path.join(BASE, "dense_predictions.csv")
EXTRAP_CSV = os.path.join(BASE, "extrapolation.csv")

# ── Helpers ────────────────────────────────────────────────────────────────
def rmse(pred, ref):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(ref)) ** 2)))

def mae(pred, ref):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(ref))))

def load_csv_timed(path, label):
    """Load a CSV, measure peak memory, and extract wall-clock time column if present."""
    tracemalloc.start()
    df = pd.read_csv(path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb = peak / 1024.0
    # Accept either 'wall_time_sec' or 'wall_time'
    wall_col = next((c for c in ["wall_time_sec", "wall_time"] if c in df.columns), None)
    wall_sec = float(df[wall_col].iloc[-1]) if wall_col else float("nan")
    print(f"  {label:<28s}: {len(df):>5} rows | "
          f"wall_time = {wall_sec:.6f} s | mem_peak = {peak_kb:.1f} KB")
    return df, wall_sec, peak_kb

# ── Load RK4 and RK45 ─────────────────────────────────────────────────────
print("=" * 65)
print("Loading solver results …")
rk4_df,  rk4_time,  rk4_mem  = load_csv_timed(RK4_CSV,  "RK4 (reference)")
rk45_df, rk45_time, rk45_mem = load_csv_timed(RK45_CSV, "Scheme 2 (RK45)")

# ── Load PINN predictions ──────────────────────────────────────────────────
# all_predictions.csv has columns: t_hours, G_RK45, G_PINN, G_error,
#                                   I_RK45, I_PINN, I_error, region
# Use the training-region rows (t_hours 0–12 h) that overlap with the ODE solvers.
print("\nLoading PINN predictions …")
tracemalloc.start()
preds_df = pd.read_csv(PREDS_CSV)
_, preds_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
preds_mem = preds_peak / 1024.0

train_df = preds_df[preds_df["region"] == "training"].copy()
print(f"  {'all_predictions.csv':<28s}: {len(preds_df):>5} rows total | "
      f"{len(train_df)} training-region rows | mem_peak = {preds_mem:.1f} KB")

# ── PINN CPU / inference time ──────────────────────────────────────────────
# inference_time.csv records 100 repeated inference runs (in seconds).
# We use the mean as the representative inference wall-clock time.
pinn_time = float("nan")
if os.path.exists(INFER_CSV):
    infer_df = pd.read_csv(INFER_CSV)
    pinn_time = float(infer_df["inference_time_s"].mean())
    print(f"  {'inference_time.csv':<28s}: {len(infer_df):>5} runs   | "
          f"mean inference = {pinn_time:.6f} s")
else:
    print("  [WARNING] inference_time.csv not found — PINN CPU time = NaN.")

# ── metrics.csv: informational print only — NOT used in comparison table ──
# These are computed against RK45 on a 500-point grid.  Using them in
# the comparison table would be unfair: different reference (RK45 ≠ RK4) and
# different grid density (500 pts ≠ 49 pts).  We print them so the reader can
# see the difference, but all comparison numbers come from the block below.
if os.path.exists(METRICS_CSV):
    met_info = pd.read_csv(METRICS_CSV).set_index("metric")["value"]
    print(f"\n  [INFO] metrics.csv (PINN vs RK45, 500-pt grid — informational only):")
    for key in ["RMSE_G", "RMSE_I", "MAE_G", "MAE_I"]:
        val = met_info.get(key, float("nan"))
        print(f"         {key} = {float(val):.6f}")
    print("         These numbers are NOT used in comparison_table.csv.")
else:
    print("\n  [INFO] metrics.csv not found — skipped (not needed for fair comparison).")

# ── Interpolate both Scheme 2 and PINN onto the RK4 49-point grid ────────
# This is the single, consistent evaluation protocol:
#   • Same reference  : RK4 solution
#   • Same grid       : 49 time points, t = 0–12 h
#   • Same method     : linear interpolation from each solver's native grid
# Applying this identically to both compared methods ensures any difference
# in RMSE/MAE reflects solver accuracy alone, not grid density or choice of
# reference.
print("\nInterpolating Scheme 2 and PINN onto RK4 49-pt grid (fair comparison) …")
t_ref = rk4_df["t"].values
G_ref = rk4_df["G"].values
I_ref = rk4_df["I"].values

# Scheme 2 (RK45): native grid varies (adaptive steps) → interpolate
G_rk45 = np.interp(t_ref, rk45_df["t"].values, rk45_df["G"].values)
I_rk45 = np.interp(t_ref, rk45_df["t"].values, rk45_df["I"].values)

rk45_rmse_G = rmse(G_rk45, G_ref)
rk45_rmse_I = rmse(I_rk45, I_ref)
rk45_mae_G  = mae(G_rk45,  G_ref)
rk45_mae_I  = mae(I_rk45,  I_ref)

# PINN: native grid = 500 uniform points (training region) → interpolate
G_pinn = np.interp(t_ref, train_df["t_hours"].values, train_df["G_PINN"].values)
I_pinn = np.interp(t_ref, train_df["t_hours"].values, train_df["I_PINN"].values)

pinn_rmse_G = rmse(G_pinn, G_ref)
pinn_rmse_I = rmse(I_pinn, I_ref)
pinn_mae_G  = mae(G_pinn,  G_ref)
pinn_mae_I  = mae(I_pinn,  I_ref)

print(f"  {'Scheme 2 (RK45)':<28s}: RMSE_G={rk45_rmse_G:.6f}  MAE_G={rk45_mae_G:.6f}")
print(f"  {'PINN':<28s}: RMSE_G={pinn_rmse_G:.6f}  MAE_G={pinn_mae_G:.6f}")

# ── comparison_table.csv ──────────────────────────────────────────────────
records = [
    {
        "Method"       : "RK4",
        "Role"         : "Reference",
        "RMSE_G"       : 0.0,
        "RMSE_I"       : 0.0,
        "MAE_G"        : 0.0,
        "MAE_I"        : 0.0,
        "CPU_time_sec" : round(rk4_time,   6),
        "Peak_mem_KB"  : round(rk4_mem,    1),
    },
    {
        "Method"       : "Scheme 2 (RK45)",
        "Role"         : "Compared",
        "RMSE_G"       : round(rk45_rmse_G, 6),
        "RMSE_I"       : round(rk45_rmse_I, 6),
        "MAE_G"        : round(rk45_mae_G,  6),
        "MAE_I"        : round(rk45_mae_I,  6),
        "CPU_time_sec" : round(rk45_time,   6),
        "Peak_mem_KB"  : round(rk45_mem,    1),
    },
    {
        "Method"       : "PINN",
        "Role"         : "Compared",
        "RMSE_G"       : round(pinn_rmse_G, 6),
        "RMSE_I"       : round(pinn_rmse_I, 6),
        "MAE_G"        : round(pinn_mae_G,  6),
        "MAE_I"        : round(pinn_mae_I,  6),
        "CPU_time_sec" : round(pinn_time,   6) if not np.isnan(pinn_time) else float("nan"),
        "Peak_mem_KB"  : round(preds_mem,   1),
    },
]
comp_df = pd.DataFrame(records)
comp_df.to_csv(os.path.join(BASE, "comparison_table.csv"), index=False)
print("\ncomparison_table.csv:")
print(comp_df.to_string(index=False))

# ── benchmark_metrics.csv ─────────────────────────────────────────────────
bm = comp_df.copy()
bm["RMSE_combined"] = (bm["RMSE_G"] + bm["RMSE_I"]).round(6)
bm["MAE_combined"]  = (bm["MAE_G"]  + bm["MAE_I"]).round(6)
compared_mask = bm["Role"] == "Compared"
bm.loc[compared_mask, "Rank_Accuracy"] = (
    bm.loc[compared_mask, "RMSE_combined"]
    .rank(method="min").astype("Int64")
)
bm.loc[compared_mask, "Rank_Speed"] = (
    bm.loc[compared_mask, "CPU_time_sec"]
    .rank(method="min", na_option="bottom").astype("Int64")
)
bm["Reference_used"] = "RK4 (rk4_results.csv)"
bm.to_csv(os.path.join(BASE, "benchmark_metrics.csv"), index=False)
print("\nbenchmark_metrics.csv saved.")

# ── Compared-only slice (for bar charts) ──────────────────────────────────
cmp     = comp_df[comp_df["Role"] == "Compared"].reset_index(drop=True)
METHODS = cmp["Method"].tolist()          # ["Scheme 2 (RK45)", "PINN"]
COLORS  = ["#16A34A", "#DC2626"]          # green, red

def bar_chart(vals_G, vals_I, ylabel, title, fname):
    x, w = np.arange(len(METHODS)), 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars_G = ax.bar(x - w / 2, vals_G, w, label="G(t)",
                    color=COLORS, alpha=0.85, edgecolor="white")
    bars_I = ax.bar(x + w / 2, vals_I, w, label="I(t)",
                    color=COLORS, alpha=0.45, edgecolor="white", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    ax.text(0.98, 0.02, "Reference: RK4 solution",
            transform=ax.transAxes, fontsize=8, color="grey",
            ha="right", va="bottom", style="italic")
    fig.tight_layout()
    p = os.path.join(FIG, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p}")

# ── Figure 1 & 2: RMSE and MAE bar charts ─────────────────────────────────
print("\nGenerating figures …")
bar_chart(
    cmp["RMSE_G"], cmp["RMSE_I"],
    "RMSE relative to RK4 reference solution",
    "RMSE relative to RK4 Reference — G(t) and I(t)",
    "rmse_bar_chart.png",
)
bar_chart(
    cmp["MAE_G"], cmp["MAE_I"],
    "MAE relative to RK4 reference solution",
    "MAE relative to RK4 Reference — G(t) and I(t)",
    "mae_bar_chart.png",
)

# ── Figure 3: Timing bar chart (all three methods) ────────────────────────
all_methods = ["RK4\n(reference)", "Scheme 2\n(RK45)", "PINN\n(inference)"]
all_colors  = ["#2563EB", "#16A34A", "#DC2626"]
times = comp_df["CPU_time_sec"].tolist()

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(all_methods, times, color=all_colors, alpha=0.85, edgecolor="white")
ax.set_ylabel("Wall-clock time (s)", fontsize=11)
ax.set_title("Solver Speed Comparison", fontsize=13, fontweight="bold")
for bar, val in zip(bars, times):
    label = f"{val:.4f}s" if not (isinstance(val, float) and np.isnan(val)) else "N/A"
    ax.text(bar.get_x() + bar.get_width() / 2,
            (bar.get_height() or 0.001) * 1.05,
            label, ha="center", va="bottom", fontsize=9)
ax.text(0.98, 0.02,
        "PINN time = mean of 100 inference runs",
        transform=ax.transAxes, fontsize=7.5, color="grey",
        ha="right", va="bottom", style="italic")
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
p = os.path.join(FIG, "timing_bar_chart.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"  Saved → {p}")

# ── Figure 4: Trajectory comparison (RK4 vs RK45 vs PINN) ────────────────
# Use training-region PINN predictions (t = 0–12 h) aligned with the ODE solvers.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

plot_data = [
    (rk4_df["t"].values,         rk4_df["G"].values, rk4_df["I"].values,
     "RK4 (reference)", "#2563EB", "-",  2.0),
    (rk45_df["t"].values,        rk45_df["G"].values, rk45_df["I"].values,
     "Scheme 2 (RK45)", "#16A34A", "--", 1.5),
    (train_df["t_hours"].values, train_df["G_PINN"].values, train_df["I_PINN"].values,
     "PINN",            "#DC2626", ":",  1.5),
]
for t, G, I, label, color, ls, lw in plot_data:
    axes[0].plot(t, G, color=color, lw=lw, ls=ls, label=label)
    axes[1].plot(t, I, color=color, lw=lw, ls=ls, label=label)

for ax, ylabel, title in [
    (axes[0], "G (mg/dL)",  "Blood Glucose G(t)"),
    (axes[1], "I (µU/mL)", "Plasma Insulin I(t)"),
]:
    ax.set_xlabel("t (hours)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

fig.suptitle(
    "All-Method Trajectory Comparison — Randall's Model",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
p = os.path.join(FIG, "trajectories_comparison.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {p}")

# ── Figure 5: PINN pointwise error profile (bonus) ────────────────────────
# Uses dense_predictions.csv (training region, t = 0–12 h).
# G_error and I_error columns are |PINN − RK45|.
if os.path.exists(DENSE_CSV):
    dense = pd.read_csv(DENSE_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, err_col, var, unit in [
        (axes[0], "G_error", "G", "mg/dL"),
        (axes[1], "I_error", "I", "µU/mL"),
    ]:
        ax.plot(dense["t_hours"], dense[err_col],
                color="#DC2626", lw=1.2, label=f"|PINN − RK45| {var}")
        ax.fill_between(dense["t_hours"], 0, dense[err_col],
                        color="#DC2626", alpha=0.15)
        ax.axvline(12, color="grey", ls="--", lw=1, label="Training boundary t=12h")
        ax.set_xlabel("t (hours)", fontsize=11)
        ax.set_ylabel(f"|PINN − RK45| ({unit})", fontsize=11)
        ax.set_title(f"PINN Absolute Error — {var}(t)", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.4)
    fig.suptitle("PINN Pointwise Error Profile (Training Window)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIG, "pinn_error_profile.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p}")
else:
    print("  [SKIP] dense_predictions.csv not found — skipping error profile.")

# ── Figure 6: PINN extrapolation beyond t = 12 h (bonus) ─────────────────
# dense_predictions.csv  → training region (t = 0–12 h, RK45 reference present)
# extrapolation.csv      → extrapolation region (t = 12–24 h, G_RK45 / I_RK45 = NaN)
if os.path.exists(DENSE_CSV) and os.path.exists(EXTRAP_CSV):
    dense  = pd.read_csv(DENSE_CSV)
    ex     = pd.read_csv(EXTRAP_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, rk_col, pinn_col, var, unit in [
        (axes[0], "G_RK45", "G_PINN", "G", "mg/dL"),
        (axes[1], "I_RK45", "I_PINN", "I", "µU/mL"),
    ]:
        # Training window: both RK45 and PINN available
        ax.plot(dense["t_hours"], dense[rk_col],  color="#000000", lw=2,
                label="RK45 reference")
        ax.plot(dense["t_hours"], dense[pinn_col], color="#DC2626", lw=1.5, ls="--",
                label="PINN (training)")
        # Extrapolation window: PINN only (RK45 is NaN)
        ax.plot(ex["t_hours"], ex[pinn_col], color="#F97316", lw=1.5, ls=":",
                label="PINN (extrapolation)")
        ax.axvline(12, color="grey", ls="--", lw=1, label="t = 12 h boundary")
        ax.set_xlabel("t (hours)", fontsize=11)
        ax.set_ylabel(f"{var} ({unit})", fontsize=11)
        ax.set_title(f"{var}(t) — Extrapolation Beyond t = 12 h", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(ls="--", alpha=0.4)

    fig.suptitle("PINN Extrapolation Performance (t = 12–24 h)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIG, "pinn_extrapolation.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p}")
else:
    print("  [SKIP] dense_predictions.csv and/or extrapolation.csv not found.")

# ── Done ───────────────────────────────────────────────────────────────────
print("\n✓  benchmark.py complete.")
print(f"   comparison_table.csv  → {BASE}")
print(f"   benchmark_metrics.csv → {BASE}")
print(f"   Figures               → {FIG}/")