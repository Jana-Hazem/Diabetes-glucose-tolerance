# =============================================================================
# FILE:   schiesser_reproduction.R
# MEMBER: 3 — ODE Reproduction (Schiesser Ch. 2)
# COURSE: SBEG108 · Numerical Methods in Biomedical Engineering, Spring 2026
#
# PURPOSE:
#   Reproduce all numerical results and figures from Schiesser (2014),
#   Chapter 2 "Diabetes Glucose Tolerance Test" (pp. 79–136).
#   Outputs are saved as PNG files to ../results/figures/.
#
# HOW TO RUN:
#   setwd("C:/Users/user/schiesser_ch2")   # project root
#   source("ode_model/schiesser_reproduction.R")
#
# DEPENDENCIES:
#   install.packages("deSolve")            # run once in R console
#
# OUTPUTS (saved to results/figures/):
#   fig2_1_glucose_G.png  — G(t) for all 4 cases  (reproduces Fig 2.1, p.101)
#   fig2_2_insulin_I.png  — I(t) for all 4 cases  (reproduces Fig 2.2, p.103)
#   fig2_3_dGdt.png       — dG/dt(t) for all 4 cases
#   fig2_4_dIdt.png       — dI/dt(t) for all 4 cases
# =============================================================================

cat("\n=== Schiesser Chapter 2: Diabetes Glucose Tolerance Test ===\n")

# ---- 0. Setup ----------------------------------------------------------------
library(deSolve)

# Source the model definition (Member 2 deliverable)
source("ode_model/model_definition.R")

# Create output directory if it does not exist
fig_dir <- file.path("results", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("PNG output directory: %s\n\n", normalizePath(fig_dir)))


# ---- 1. Integration settings -------------------------------------------------
# 49 output points: t = 0, 0.25, 0.50, ..., 12  h
# This gives 3 points in the infusion window (t=0, 0.25, 0.5)
# and enough resolution for clear plots without crowding.
# (Schiesser p. 90, Listing 2.1: nout=49)
nout  <- 49
times <- seq(from = 0, to = 12, by = 12 / (nout - 1))   # eq. 2.1, p.80

# Storage matrices: rows = time points, cols = 4 cases
Gplot  <- matrix(0, nrow = nout, ncol = 4)
Iplot  <- matrix(0, nrow = nout, ncol = 4)
dGplot <- matrix(0, nrow = nout, ncol = 4)
dIplot <- matrix(0, nrow = nout, ncol = 4)
tplot  <- numeric(nout)
ncall_vec <- integer(4)


# ---- 2. Loop over four simulation cases --------------------------------------
# Case descriptions (Schiesser p. 89-90):
#   ncase=1  Bb=14.3         Gt=0      Normal; no infusion → steady state
#   ncase=2  Bb=14.3         Gt=80000  Normal; with infusion → returns to basal
#   ncase=3  Bb=0.2*14.3     Gt=80000  Reduced sensitivity  → hyperglycaemia
#   ncase=4  Bb=2.0*14.3     Gt=80000  Elevated sensitivity → hypoglycaemia

cat(sprintf("%-12s  %-10s  %-10s  %-10s  %-10s  %s\n",
            "Case", "t=0.25 G", "t=0.25 I", "t=12 G", "t=12 I", "ncall"))
cat(strrep("-", 65), "\n")

for (ncase in 1:4) {

  # Load parameters for this case (from model_definition.R)
  params <- default_params(ncase)
  list2env(params, envir = environment())   # make Cg, Ci, Q, ... visible here
  # Also export to .GlobalEnv so schiesser_ode() can see them via <<- scoping
  list2env(params, envir = .GlobalEnv)

  # Initialise call counter (schiesser_ode increments ncall <<-)
  ncall <- 0

  # Initial conditions: G(0)=81.14 mg/100ml, I(0)=5.671 mg/100ml
  # (Schiesser p. 94, Listing 2.1)
  yini <- c(G = 81.14, I = 5.671)

  # ODE integration via lsoda (adaptive stiff/non-stiff switcher in deSolve)
  # (Schiesser Listing 2.1: ode() defaults to lsoda)
  out <- ode(y     = yini,
             times = times,
             func  = schiesser_ode,
             parms = NULL)

  # Store trajectories for plotting
  Gplot[, ncase] <- out[, 2]   # G(t)  col 2 of out
  Iplot[, ncase] <- out[, 3]   # I(t)  col 3
  if (ncase == 1) tplot <- out[, 1]
  ncall_vec[ncase] <- ncall

  # Compute derivatives at each output point for Figs 2.3 / 2.4
  for (it in 1:nout) {
    t_val <- tplot[it]
    G_val <- Gplot[it, ncase]
    I_val <- Iplot[it, ncase]
    # Re-evaluate the RHS (same logic as schiesser_ode, inline for clarity)
    In_val <- if ((t_val >= 0) & (t_val <= 0.51)) Gt else 0
    if (G_val < Gk) {
      dGplot[it, ncase] <- (1/Cg)*(Q + In_val - Gg*I_val*G_val - Dd*G_val)
    } else {
      dGplot[it, ncase] <- (1/Cg)*(Q + In_val - Gg*I_val*G_val - Dd*G_val
                                   - Mu*(G_val - Gk))
    }
    if (G_val < G0) {
      dIplot[it, ncase] <- (1/Ci)*(-Aa*I_val)
    } else {
      dIplot[it, ncase] <- (1/Ci)*(-Aa*I_val + Bb*(G_val - G0))
    }
  }

  # Print summary row: values at t=0.25 h (it=2) and t=12 h (it=49)
  cat(sprintf("ncase = %d:  %10.2f  %10.3f  %10.2f  %10.3f  %5d\n",
              ncase,
              Gplot[2, ncase], Iplot[2, ncase],
              Gplot[nout, ncase], Iplot[nout, ncase],
              ncall_vec[ncase]))
}


# ---- 3. Validation against book Tables 2.1 / 2.2 ----------------------------
# Expected values from Schiesser pp. 100-103 (Tables 2.1 and 2.2)
book <- data.frame(
  ncase = c(1,    2,      3,      4),
  G025  = c(81.14, 201.71, 204.22, 198.66),
  I025  = c(5.671,   7.100,   5.420,   9.167),
  G12   = c(81.14,  81.02, 129.23,  69.49),
  I12   = c(5.671,   5.664,   2.914,   6.926)
)

cat("\n--- Validation vs. Schiesser Tables 2.1/2.2 ---\n")
cat(sprintf("%-8s  %-8s  %-8s  %-8s  %-8s\n",
            "ncase", "|ΔG025|", "|ΔI025|", "|ΔG12|", "|ΔI12|"))
for (i in 1:4) {
  cat(sprintf("%-8d  %-8.4f  %-8.4f  %-8.4f  %-8.4f\n",
              i,
              abs(Gplot[2, i]    - book$G025[i]),
              abs(Iplot[2, i]    - book$I025[i]),
              abs(Gplot[nout, i] - book$G12[i]),
              abs(Iplot[nout, i] - book$I12[i])))
}
cat("(Errors < 0.01 confirm faithful reproduction)\n\n")


# ---- 4. Figure helpers -------------------------------------------------------
case_labels <- c("1: Normal, no infusion",
                 "2: Normal + infusion",
                 "3: Reduced Bb (hyperglycaemia)",
                 "4: Elevated Bb (hypoglycaemia)")
ltys  <- c(1, 2, 3, 4)
cols  <- c("black", "blue", "red", "darkgreen")
pchs  <- c("1", "2", "3", "4")

save_png <- function(filename, expr_fn) {
  path <- file.path(fig_dir, filename)
  png(path, width = 900, height = 600, res = 120)
  expr_fn()
  dev.off()
  cat(sprintf("  Saved: %s\n", normalizePath(path)))
}


# ---- 5. Fig 2.1 — G(t) extracellular glucose --------------------------------
# Reproduces Figure 2.1, Schiesser p.101
save_png("fig2_1_glucose_G.png", function() {
  plot(tplot, Gplot[, 1],
       xlab = "t (hr)",
       ylab = "G(t)  (mg glucose / 100 ml extracellular fluid)",
       xlim = c(0, 12), ylim = c(0, 300),
       type = "b", lty = ltys[1], col = cols[1], pch = pchs[1], lwd = 2,
       main = "Fig 2.1 — Extracellular glucose G(t), ncase = 1,2,3,4\n(Schiesser 2014, p. 101, eq. 2.1b/2.1c)")
  for (nc in 2:4) {
    lines(tplot, Gplot[, nc],
          type = "b", lty = ltys[nc], col = cols[nc], pch = pchs[nc], lwd = 2)
  }
  legend("topright", legend = case_labels,
         lty = ltys, col = cols, pch = pchs, lwd = 2, cex = 0.8)
  abline(h = 250, lty = 3, col = "grey50")   # renal threshold Gk
  text(11, 252, "Gk = 250", cex = 0.7, col = "grey50")
})


# ---- 6. Fig 2.2 — I(t) extracellular insulin --------------------------------
# Reproduces Figure 2.2, Schiesser p.103
save_png("fig2_2_insulin_I.png", function() {
  plot(tplot, Iplot[, 1],
       xlab = "t (hr)",
       ylab = "I(t)  (mg insulin / 100 ml extracellular fluid)",
       xlim = c(0, 12), ylim = c(0, 25),
       type = "b", lty = ltys[1], col = cols[1], pch = pchs[1], lwd = 2,
       main = "Fig 2.2 — Extracellular insulin I(t), ncase = 1,2,3,4\n(Schiesser 2014, p. 103, eq. 2.2b/2.2c)")
  for (nc in 2:4) {
    lines(tplot, Iplot[, nc],
          type = "b", lty = ltys[nc], col = cols[nc], pch = pchs[nc], lwd = 2)
  }
  legend("topright", legend = case_labels,
         lty = ltys, col = cols, pch = pchs, lwd = 2, cex = 0.8)
  abline(h = 51, lty = 3, col = "grey50")    # pancreatic threshold G0
  text(11, 52.5, "G0 = 51", cex = 0.7, col = "grey50")
})


# ---- 7. Fig 2.3 — dG/dt (rate of change of glucose) ------------------------
save_png("fig2_3_dGdt.png", function() {
  ylim_dG <- range(dGplot) * c(1.1, 1.1)
  plot(tplot, dGplot[, 1],
       xlab = "t (hr)",
       ylab = "dG/dt  (mg glucose / (100 ml · hr))",
       xlim = c(0, 12), ylim = ylim_dG,
       type = "b", lty = ltys[1], col = cols[1], pch = pchs[1], lwd = 2,
       main = "Fig 2.3 — Rate of change of glucose dG/dt, ncase = 1,2,3,4\n(Schiesser 2014, eq. 2.1b/2.1c)")
  for (nc in 2:4) {
    lines(tplot, dGplot[, nc],
          type = "b", lty = ltys[nc], col = cols[nc], pch = pchs[nc], lwd = 2)
  }
  abline(h = 0, lty = 2, col = "grey40")
  legend("topright", legend = case_labels,
         lty = ltys, col = cols, pch = pchs, lwd = 2, cex = 0.8)
})


# ---- 8. Fig 2.4 — dI/dt (rate of change of insulin) ------------------------
save_png("fig2_4_dIdt.png", function() {
  ylim_dI <- range(dIplot) * c(1.1, 1.1)
  plot(tplot, dIplot[, 1],
       xlab = "t (hr)",
       ylab = "dI/dt  (mg insulin / (100 ml · hr))",
       xlim = c(0, 12), ylim = ylim_dI,
       type = "b", lty = ltys[1], col = cols[1], pch = pchs[1], lwd = 2,
       main = "Fig 2.4 — Rate of change of insulin dI/dt, ncase = 1,2,3,4\n(Schiesser 2014, eq. 2.2b/2.2c)")
  for (nc in 2:4) {
    lines(tplot, dIplot[, nc],
          type = "b", lty = ltys[nc], col = cols[nc], pch = pchs[nc], lwd = 2)
  }
  abline(h = 0, lty = 2, col = "grey40")
  legend("topright", legend = case_labels,
         lty = ltys, col = cols, pch = pchs, lwd = 2, cex = 0.8)
})

cat("\nDone. All 4 figures saved to results/figures/\n")
