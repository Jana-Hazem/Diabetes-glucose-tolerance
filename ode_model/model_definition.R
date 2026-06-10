# =============================================================================
# FILE:   model_definition.R
# MEMBER: 2 — ODE Model Lead
# COURSE: SBEG108 · Numerical Methods in Biomedical Engineering, Spring 2026
# MODEL:  Schiesser (2014) Diabetes Glucose Tolerance Test
#         Chapter 2, pp. 79–103
#
# DESCRIPTION:
#   Defines the right-hand side of the two-ODE system for extracellular
#   glucose G(t) and extracellular insulin I(t) during an Oral Glucose
#   Tolerance Test (OGTT).  The model is taken directly from Schiesser
#   (2014), equations (2.1b)/(2.1c) for glucose and (2.2b)/(2.2c) for
#   insulin (pp. 80–84).
#
#   NOTE ON "BERGMAN" LABEL IN THE TASK SHEET:
#   The task sheet refers to the "Bergman Minimal Model."  Schiesser
#   Chapter 2 is based on the earlier Randall (1980) model (Schiesser
#   p. 79) — a 2-ODE system in G and I only, without a remote-insulin
#   compartment X.  The files here faithfully reproduce Schiesser Ch. 2.
#
# EQUATIONS:
#   Glucose balance  (eq. 2.1b / 2.1c, p. 80):
#     Cg * dG/dt = Q + In - Gg*I*G - Dd*G            if G <  Gk
#     Cg * dG/dt = Q + In - Gg*I*G - Dd*G - Mu*(G-Gk) if G >= Gk
#
#   Insulin balance  (eq. 2.2b / 2.2c, p. 84):
#     Ci * dI/dt = -Aa*I                  if G <  G0
#     Ci * dI/dt = -Aa*I + Bb*(G - G0)   if G >= G0
#
# PARAMETERS (all from Schiesser p. 90, Listing 2.1):
#   Ex   = 15000  ml        total extracellular space
#   Cg   = 150             glucose capacitance = Ex/100
#   Ci   = 150             insulin capacitance = Ex/100
#   Q    = 8400   mg/h     basal liver release of glucose
#   Dd   = 24.7   h⁻¹      first-order glucose metabolism rate
#   Gg   = 13.9   (mg/h)/[(mg/100ml)²]  insulin-controlled glucose loss
#   Gk   = 250    mg/100ml renal threshold (glucose above this → renal loss)
#   Mu   = 72     h⁻¹      renal glucose removal rate coefficient
#   G0   = 51     mg/100ml pancreatic threshold (glucose above this → insulin)
#   Aa   = 76     h⁻¹      first-order insulin degradation rate
#   Bb   = 14.3   (mg/h)/(mg/100ml)  pancreatic insulin release rate
#                           (normal); varied across cases (see below)
#   Gt   = 80000  mg       total glucose infused during 0 ≤ t ≤ 0.5 h
#                           (Gt = 0 for ncase = 1, no infusion)
#
# INITIAL CONDITIONS (Schiesser p. 94, Listing 2.1):
#   G(0) = 81.14  mg glucose / 100 ml extracellular fluid
#   I(0) =  5.671 mg insulin / 100 ml extracellular fluid
#
# CASES (Schiesser p. 89–90):
#   ncase = 1: Bb = 14.3,       Gt = 0      normal, no infusion (steady state)
#   ncase = 2: Bb = 14.3,       Gt = 80000  normal response → returns to baseline
#   ncase = 3: Bb = 0.2*14.3,   Gt = 80000  reduced sensitivity → hyperglycaemia
#   ncase = 4: Bb = 2.0*14.3,   Gt = 80000  elevated sensitivity → hypoglycaemia
#
# USAGE:
#   Source this file, then call schiesser_ode(t, y, parms) as the ODE
#   function for deSolve::ode().  The parameter vector parms must be a
#   named list or NULL (parameters are read from the calling environment).
#
#   Example:
#     library(deSolve)
#     source("model_definition.R")
#     params <- default_params(ncase = 2)
#     list2env(params, envir = .GlobalEnv)
#     ncall <- 0
#     out <- ode(y = c(G = 81.14, I = 5.671),
#                times = seq(0, 12, by = 0.25),
#                func  = schiesser_ode,
#                parms = NULL)
# =============================================================================


# -----------------------------------------------------------------------------
# default_params()
#
# Returns a named list of all model parameters for a given case number.
#
# Arguments:
#   ncase  integer in {1,2,3,4}.  Selects Bb and Gt (see table above).
#
# Returns:
#   A named list with elements:
#     Ex, Cg, Ci, Q, Dd, Gg, Gk, Mu, G0, Aa, Bb, Gt
#   These match the variable names used in schiesser_ode() below.
# -----------------------------------------------------------------------------
default_params <- function(ncase = 2) {

  # Fixed parameters (Schiesser p. 90, Listing 2.1)
  Ex <- 15000   # ml          total extracellular space
  Cg <- 150     # dimensionless  glucose capacitance (= Ex/100)
  Ci <- 150     # dimensionless  insulin capacitance (= Ex/100)
  Q  <- 8400    # mg/h        basal liver release of glucose
  Dd <- 24.7    # h^-1        first-order glucose metabolism rate
  Gg <- 13.9    # (mg/h)/[(mg/100ml)^2]  insulin-glucose interaction rate
  Gk <- 250     # mg/100ml    renal threshold for glucose removal
  Mu <- 72      # h^-1        renal glucose removal rate coefficient
  G0 <- 51      # mg/100ml    pancreatic threshold for insulin secretion
  Aa <- 76      # h^-1        first-order insulin degradation rate

  # Case-dependent parameters (Schiesser p. 90, Listing 2.1)
  if (ncase == 1) { Bb <- 14.3;        Gt <- 0     }   # normal, no infusion
  if (ncase == 2) { Bb <- 14.3;        Gt <- 80000 }   # normal + infusion
  if (ncase == 3) { Bb <- 0.2 * 14.3;  Gt <- 80000 }   # reduced sensitivity
  if (ncase == 4) { Bb <- 2.0 * 14.3;  Gt <- 80000 }   # elevated sensitivity

  list(Ex = Ex, Cg = Cg, Ci = Ci, Q  = Q,  Dd = Dd,
       Gg = Gg, Gk = Gk, Mu = Mu, G0 = G0, Aa = Aa,
       Bb = Bb, Gt = Gt)
}


# -----------------------------------------------------------------------------
# schiesser_ode()
#
# ODE right-hand side for deSolve::ode().
#
# Arguments:
#   t      current time (h)
#   y      named numeric vector with elements G (mg/100ml) and I (mg/100ml)
#   parms  unused (pass NULL); parameters are read from the calling
#          environment via <<- / parent-frame scoping
#
# Returns:
#   list(c(dGdt, dIdt)) — required format for deSolve::ode()
#
# Side-effects:
#   Increments ncall in the parent environment (for call-count diagnostics).
#
# Equations:
#   Glucose (eq. 2.1b / 2.1c, Schiesser p. 80):
#     if G <  Gk:  dG/dt = (1/Cg) * [ Q + In - Gg*I*G - Dd*G ]
#     if G >= Gk:  dG/dt = (1/Cg) * [ Q + In - Gg*I*G - Dd*G - Mu*(G-Gk) ]
#
#   Insulin (eq. 2.2b / 2.2c, Schiesser p. 84):
#     if G <  G0:  dI/dt = (1/Ci) * [ -Aa*I ]
#     if G >= G0:  dI/dt = (1/Ci) * [ -Aa*I + Bb*(G-G0) ]
#
#   Glucose infusion (Schiesser p. 89):
#     In = Gt   for 0   <= t <= 0.5 h  (0.51 used to avoid fp equality)
#     In = 0    for t   >  0.5 h
# -----------------------------------------------------------------------------
schiesser_ode <- function(t, y, parms) {

  # Unpack state variables
  G <- y[1]   # extracellular glucose  (mg glucose / 100 ml)
  I <- y[2]   # extracellular insulin  (mg insulin / 100 ml)

  # Glucose infusion function (eq. 2.3, Schiesser p. 89)
  # Note: t <= 0.51 instead of t <= 0.5 to avoid floating-point equality test
  if ((t >= 0) & (t <= 0.51)) { In <- Gt } else { In <- 0 }

  # --- Glucose ODE (eqs. 2.1b and 2.1c, p. 80) ---
  # Biological interpretation of each term:
  #   Q        : constant liver production of glucose (source)
  #   In       : external glucose infusion during OGTT (source, time-limited)
  #  -Gg*I*G  : insulin-controlled glucose uptake by tissues (nonlinear sink)
  #  -Dd*G    : first-order peripheral glucose metabolism (linear sink)
  #  -Mu*(G-Gk): renal (kidney) excretion when glucose exceeds threshold Gk
  if (G < Gk) {
    dGdt <- (1 / Cg) * (Q + In - Gg * I * G - Dd * G)
  } else {
    dGdt <- (1 / Cg) * (Q + In - Gg * I * G - Dd * G - Mu * (G - Gk))
  }

  # --- Insulin ODE (eqs. 2.2b and 2.2c, p. 84) ---
  # Biological interpretation:
  #  -Aa*I        : first-order insulin degradation / clearance (sink)
  #  +Bb*(G-G0)   : pancreatic insulin secretion triggered when G > G0 (source)
  if (G < G0) {
    dIdt <- (1 / Ci) * (-Aa * I)
  } else {
    dIdt <- (1 / Ci) * (-Aa * I + Bb * (G - G0))
  }

  # Increment derivative-evaluation counter (diagnostic)
  ncall <<- ncall + 1

  # Return in deSolve format: list of a single vector c(dG/dt, dI/dt)
  return(list(c(dGdt, dIdt)))
}


# -----------------------------------------------------------------------------
# PARAMETER TABLE (for LaTeX / report reference)
#
# Symbol  Value       Units                             Physiological meaning
# ------  ----------  --------------------------------  -------------------------
# Cg      150         (number of 100-ml ECF volumes)   Glucose capacitance
# Ci      150         (number of 100-ml ECF volumes)   Insulin capacitance
# Q       8400        mg glucose / h                   Basal liver glucose output
# Dd      24.7        h^-1                             1st-order glucose clearance
# Gg      13.9        h^-1 / (mg insulin / 100 ml)     Insulin-mediated glucose uptake
# Gk      250         mg glucose / 100 ml              Renal glucose threshold
# Mu      72          h^-1                             Renal loss rate coefficient
# G0      51          mg glucose / 100 ml              Pancreatic secretion threshold
# Aa      76          h^-1                             Insulin degradation rate
# Bb      14.3        mg insulin/(h · mg glucose/100ml) Pancreatic sensitivity (normal)
# Gt      80000       mg glucose                       Total OGTT glucose dose
# G(0)    81.14       mg glucose / 100 ml              Initial glucose (basal)
# I(0)     5.671      mg insulin / 100 ml              Initial insulin (basal)
# -----------------------------------------------------------------------------
