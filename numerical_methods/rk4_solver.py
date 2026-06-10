import numpy as np
import time
import csv
from ode_model.model_definition import randall_ode  # Member 2's file

def rk4_step(f, t, y, h, params):
    k1 = f(t,        y,           params)
    k2 = f(t + h/2,  y + h/2*k1, params)
    k3 = f(t + h/2,  y + h/2*k2, params)
    k4 = f(t + h,    y + h*k3,   params)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def run_rk4(h, t_end, y0, params, output_path):
    t = 0.0
    y = np.array(y0)  # [G, I]
    results = []

    t_start = time.time()
    while t <= t_end:
        wall = time.time() - t_start
        results.append([t, y[0], y[1], wall])
        y = rk4_step(randall_ode, t, y, h, params)
        t += h

    # Save CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'G', 'I', 'wall_time_sec'])
        writer.writerows(results)

# Normal run
run_rk4(h=1.0, t_end=180, y0=[300, baseline_I], params={...},
        output_path='results/rk4_results.csv')

# Fine-step reference for ML group (Member 7 depends on this!)
run_rk4(h=0.01, t_end=180, y0=[300, baseline_I], params={...},
        output_path='results/rk4_reference.csv')
