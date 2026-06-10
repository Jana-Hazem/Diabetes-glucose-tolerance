# =============================================================================
# FILE:   rk4_solver.py
# MEMBER: 4 — RK4 Solver
# COURSE: SBEG108 · Numerical Methods in Biomedical Engineering, Spring 2026
#
# PURPOSE:
#   Implements 4th-order Runge-Kutta (RK4) from scratch for the Schiesser
#   (2014) glucose-insulin ODE system.  Runs all four clinical cases and
#   produces:
#     - rk4_results_case<N>.csv      standard run (h=0.01 h, output @ 0.25 h)
#     - rk4_reference_case<N>.csv    fine-step reference (h=0.001 h)
#     - rk4_convergence.csv          RMSE vs step-size for convergence study
#     - rk4_trajectories_case<N>.png G(t) and I(t) plots
#     - rk4_convergence.png          log-log convergence plot
#
# TIME UNITS : hours (h)              ← confirmed with Member 5
# OUTPUT GRID: every 0.25 h (quarter-hour) ← Member 5's scheme
# TIME SPAN  : 0 to 12 h
#
# CSV SCHEMA (agreed with Member 8):
#   t, G, I, wall_time_sec
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ode_model.model_definition import (
    schiesser_ode,
    default_params,
    DEFAULT_INITIAL,
    T_START, T_END, DT_OUT
)

# =============================================================================
# Core RK4 step
# =============================================================================
def rk4_step(f, t, y, h, params):
    """Single RK4 step."""
    k1 = f(t,         y,            params)
    k2 = f(t + h/2.0, y + h/2.0*k1, params)
    k3 = f(t + h/2.0, y + h/2.0*k2, params)
    k4 = f(t + h,     y + h*k3,     params)
    return y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# =============================================================================
# RK4 solver — returns dense output on the quarter-hour grid
# =============================================================================
def run_rk4(h, t_end, y0, params, output_path, t_start=0.0, dt_out=0.25):
    """
    Run RK4 from t_start to t_end with internal step h.
    Records output every dt_out hours (quarter-hour grid).

    CSV columns: t, G, I, wall_time_sec
    """
    t   = t_start
    y   = np.array(y0, dtype=float)
    results = []
    next_out = t_start          # next output time
    wall_t0  = time.perf_counter()

    # Safety: ensure h divides dt_out cleanly (round to avoid drift)
    steps_per_out = max(1, round(dt_out / h))
    h_actual = dt_out / steps_per_out  # exact sub-step size

    while next_out <= t_end + 1e-12:
        # Record at this output point
        wall = time.perf_counter() - wall_t0
        results.append([round(t, 10), float(y[0]), float(y[1]), wall])

        if next_out >= t_end - 1e-12:
            break

        # Advance exactly steps_per_out sub-steps to reach next output time
        for _ in range(steps_per_out):
            y = rk4_step(schiesser_ode, t, y, h_actual, params)
            t += h_actual
        t = next_out + dt_out   # snap to exact grid to prevent drift
        next_out = t

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "G", "I", "wall_time_sec"])
        writer.writerows(results)

    return np.array(results)   # shape (N, 4): t, G, I, wall


# =============================================================================
# Convergence study
# =============================================================================
def convergence_study(params, y0, t_end, output_path, dt_out=0.25):
    """
    Run RK4 at multiple step sizes, compare to fine-step reference (h=0.001 h).
    Returns RMSE of G and I vs h.
    """
    h_ref  = 0.001
    ref    = run_rk4(h_ref, t_end, y0, params,
                     "/tmp/rk4_conv_ref.csv", dt_out=dt_out)
    G_ref  = ref[:, 1]
    I_ref  = ref[:, 2]

    step_sizes = [0.25, 0.1, 0.05, 0.01, 0.005]
    rows = []
    for h in step_sizes:
        res   = run_rk4(h, t_end, y0, params,
                        f"/tmp/rk4_conv_h{h}.csv", dt_out=dt_out)
        G_rmse = np.sqrt(np.mean((res[:, 1] - G_ref)**2))
        I_rmse = np.sqrt(np.mean((res[:, 2] - I_ref)**2))
        rows.append([h, G_rmse, I_rmse])
        print(f"  h={h:.3f} h | RMSE_G={G_rmse:.4e} | RMSE_I={I_rmse:.4e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["h", "RMSE_G", "RMSE_I"])
        writer.writerows(rows)

    return np.array(rows)


# =============================================================================
# Plots
# =============================================================================
CASE_LABELS = {
    1: "Case 1: Normal, no infusion",
    2: "Case 2: Normal + OGTT",
    3: "Case 3: Reduced sensitivity (hyperglycaemia)",
    4: "Case 4: Elevated sensitivity (hypoglycaemia)",
}

def plot_trajectories(results, case_num, out_path):
    """Plot G(t) and I(t) for a single case."""
    t = results[:, 0]
    G = results[:, 1]
    I = results[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, G, color="steelblue", linewidth=1.8)
    axes[0].set_ylabel("G(t)  [mg / 100 ml]", fontsize=11)
    axes[0].set_title(CASE_LABELS[case_num], fontsize=12)
    axes[0].axhline(250, color="red",   linestyle="--", linewidth=0.8,
                    label="Renal threshold (Gk=250)")
    axes[0].axhline(51,  color="orange", linestyle="--", linewidth=0.8,
                    label="Pancreatic threshold (G0=51)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, I, color="darkorange", linewidth=1.8)
    axes[1].set_ylabel("I(t)  [mg / 100 ml]", fontsize=11)
    axes[1].set_xlabel("Time  [h]", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_convergence(conv_data, out_path):
    """Log-log plot of RMSE vs h — slope should be ~4 for RK4."""
    h       = conv_data[:, 0]
    G_rmse  = conv_data[:, 1]
    I_rmse  = conv_data[:, 2]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h, G_rmse, "o-", color="steelblue",   label="RMSE  G(t)", linewidth=1.8)
    ax.loglog(h, I_rmse, "s-", color="darkorange",  label="RMSE  I(t)", linewidth=1.8)

    # Reference line with slope 4
    h_line = np.array([min(h), max(h)])
    ax.loglog(h_line, G_rmse[-1] * (h_line / h[-1])**4,
              "k--", linewidth=1.0, label="Slope = 4 (reference)")

    ax.set_xlabel("Step size  h  [h]", fontsize=11)
    ax.set_ylabel("RMSE  [mg / 100 ml]", fontsize=11)
    ax.set_title("RK4 Convergence Study — Case 2 (Normal + OGTT)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    y0    = DEFAULT_INITIAL   # [81.14, 5.671]
    t_end = T_END             # 12 h
    dt_out = DT_OUT           # 0.25 h (quarter-hour)

    results_dir = "results"
    figures_dir = "results/figures"

    # ------------------------------------------------------------------
    # Run all 4 cases
    # ------------------------------------------------------------------
    for ncase in [1, 2, 3, 4]:
        params = default_params(ncase)
        print(f"\n=== Case {ncase}: Bb={params['Bb']:.4f}, Gt={params['Gt']} ===")

        # Standard run  (h = 0.01 h → 25 sub-steps per quarter-hour output)
        std_path = f"{results_dir}/rk4_results_case{ncase}.csv"
        print(f"  Running standard (h=0.01 h) ...")
        std_res = run_rk4(0.01, t_end, y0, params, std_path, dt_out=dt_out)
        print(f"  Saved: {std_path}  ({len(std_res)} output rows)")

        # Fine-step reference (h = 0.001 h → 250 sub-steps per quarter-hour)
        ref_path = f"{results_dir}/rk4_reference_case{ncase}.csv"
        print(f"  Running fine-step reference (h=0.001 h) ...")
        ref_res = run_rk4(0.001, t_end, y0, params, ref_path, dt_out=dt_out)
        print(f"  Saved: {ref_path}  ({len(ref_res)} output rows)")

        # Trajectory plot
        plot_trajectories(std_res, ncase,
                          f"{figures_dir}/rk4_trajectories_case{ncase}.png")

    # ------------------------------------------------------------------
    # Convergence study (Case 2 — main clinical case)
    # ------------------------------------------------------------------
    print("\n=== Convergence study (Case 2) ===")
    params2   = default_params(2)
    conv_data = convergence_study(params2, y0, t_end,
                                  f"{results_dir}/rk4_convergence.csv",
                                  dt_out=dt_out)
    plot_convergence(conv_data, f"{figures_dir}/rk4_convergence.png")

    print("\n✅  All done.")
