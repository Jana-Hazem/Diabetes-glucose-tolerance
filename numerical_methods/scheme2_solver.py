import os
import numpy as np
import pandas as pd
import time
from scipy.integrate import solve_ivp

# ---------------------------------------------------------
# 1. THE EXACT REFERENCE MODEL (Schiesser Ch. 2)
# ---------------------------------------------------------
def schiesser_model(t, y, Cg, Ci, Q, Dd, Gg, Gk, Mu, G0, Aa, Bb, Gt):
    """
    Computes the derivatives for the 2-variable Schiesser Model.
    y = [G, I] (Extracellular Glucose, Extracellular Insulin)
    """
    G, I = y
    
    # Glucose Infusion Function (In)
    # The patient drinks the sugar dose (Gt) only during the first 0.5 hours
    if t >= 0 and t <= 0.5:
        In_infusion = Gt
    else:
        In_infusion = 0
        
    # Equation 1: Extracellular Glucose (G)
    if G < Gk:
        dG_dt = (1/Cg) * (Q + In_infusion - (Gg * I * G) - Dd * G)
    else:
        # Kidneys filter glucose if it exceeds renal threshold (Gk)
        dG_dt = (1/Cg) * (Q + In_infusion - (Gg * I * G) - Dd * G - Mu * (G - Gk))
        
    # Equation 2: Extracellular Insulin (I)
    if G < G0:
        dI_dt = (1/Ci) * (-Aa * I)
    else:
        # Pancreas secretes insulin if glucose exceeds threshold (G0)
        dI_dt = (1/Ci) * (-Aa * I + Bb * (G - G0))
        
    return [dG_dt, dI_dt]

# ---------------------------------------------------------
# 2. EXACT PARAMETERS & INITIAL CONDITIONS (Case 2: Normal OGTT)
# ---------------------------------------------------------
# Constant Parameters from Schiesser Page 88 & 90
Cg = 150; Ci = 150; Q = 8400; Dd = 24.7
Gg = 13.9; Gk = 250; Mu = 72; G0 = 51; Aa = 76

# Test Case 2 Variables (Normal Pancreas Sensitivity, Takes Sugar Drink)
Bb = 14.3; Gt = 80000  

params = (Cg, Ci, Q, Dd, Gg, Gk, Mu, G0, Aa, Bb, Gt)

# Initial conditions from Page 94: G(0)=81.14, I(0)=5.671
y0 = [81.14, 5.671] 

#The book calculates time in HOURS
# Time interval: t = 0 to 12 hours, capturing 49 data points
t_span = (0, 12)
t_eval = np.linspace(0, 12, 49)

# ---------------------------------------------------------
# 3. RUN THE SOLVER WITH ADAPTIVE RK45
# ---------------------------------------------------------
print("Starting SciPy Explicit Adaptive Solver (RK45)...")
start_time = time.time()

# Using method='RK45'
solution = solve_ivp(
    fun=schiesser_model, 
    t_span=t_span, 
    y0=y0, 
    method='RK45', 
    t_eval=t_eval, 
    args=params
)

wall_time = time.time() - start_time
print(f"Solver finished in {wall_time:.6f} seconds.")

df_results = pd.DataFrame({
    't': solution.t,
    'G': solution.y[0], 
    'I': solution.y[1],
    'wall_time': wall_time
})

output_path = '../results/scheme2_results.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_results.to_csv(output_path, index=False)

print(f"Results successfully saved to {output_path}")