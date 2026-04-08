import numpy as np
import cupy as cp
import fbmfast_gpu as fbm
import time
import os

start_time = time.time()

# SET UP
iterations = 20
Dr_values = [0.1]
H1_values = [0.5] # H for Equation of Angle
H2_values = [0.5, 0.99] # H for Equation of Motion --> 2D (x and y)
Alpha_values = [0.25]
Fm = 1
gamma_s = 1
dt = 0.1
T = 100
Nts = int(T/dt)
Nts2 = np.linspace(0, T, Nts)

for Hval1 in range(len(H1_values)):
    start_time_model = time.time()
    for Hval2 in range(len(H2_values)):
        for Drval in range(len(Dr_values)):
            for Alphaval in range(len(Alpha_values)):
                print(f"Running batch: Dr={Dr_values[Drval]}, H1={H1_values[Hval1]}, H2={H2_values[Hval2]}")
                start_time_batch = time.time()
                
                # Values of Hurst Index for Correlations
                H1 = H1_values[Hval1]
                H2 = H2_values[Hval2]
                Dr = Dr_values[Drval]
                alpha = Alpha_values[Alphaval]
                
                # Definition for Fractional Brownian Motion 1 -> Noise Angle
                dfW1 = cp.zeros((iterations, Nts))
                fbm_noise1 = fbm.fbm_batch(Nts, H1, dt, iterations)
                dfW1[:, 1:] = cp.diff(fbm_noise1)

                # Definition for Fractional Brownian Motion 2 -> Correlated Noise
                dfW2_x = cp.zeros((iterations, Nts))
                fbm_noise2_x = fbm.fbm_batch(Nts, H2, dt, iterations)
                dfW2_x[:, 1:] = cp.diff(fbm_noise2_x)
                dfW2_y = cp.zeros((iterations, Nts))
                fbm_noise2_y = fbm.fbm_batch(Nts, H2, dt, iterations)
                dfW2_y[:, 1:] = cp.diff(fbm_noise2_y)

                # Initial conditions
                x_array = cp.zeros((iterations, Nts))
                y_array = cp.zeros((iterations, Nts))
                theta_array = cp.zeros((iterations, Nts))
                theta_array[:, 0] = cp.random.uniform(0, 2*cp.pi, iterations)
                
                # New test for saving values for quantificatino of alpha
                fmpi_x_array = cp.zeros((iterations, Nts))
                fmpi_y_array = cp.zeros((iterations, Nts))
                xi_x_array = cp.zeros((iterations, Nts))
                xi_y_array = cp.zeros((iterations, Nts))

                for t in range(1, Nts):
                    # AS OF THEORY CORRECTIONS MAY 2024 --> Xi(t) = dfW(t)/dt
                    # Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)/dt
                    dtheta = cp.sqrt(2*Dr)*dfW1[:, t]
                    theta_array[:, t] = theta_array[:, t-1] + dtheta
                    # Equation of Motion: Fm*{cos(theta), sin(theta)} = gamma_s*v_i - alpha*dfW2[t]/dt
                    force_term_dx = Fm*cp.cos(theta_array[:, t])/gamma_s
                    stoch_term_dx = alpha*dfW2_x[:, t]/gamma_s
                    force_term_dy = Fm*cp.sin(theta_array[:, t])/gamma_s
                    stoch_term_dy = alpha*dfW2_y[:, t]/gamma_s
                    # Update of position
                    x_array[:, t] = x_array[:, t-1] + force_term_dx*dt + stoch_term_dx
                    y_array[:, t] = y_array[:, t-1] + force_term_dy*dt + stoch_term_dy
                    fmpi_x_array[:, t] = force_term_dx*dt
                    fmpi_y_array[:, t] = force_term_dy*dt
                    xi_x_array[:, t] = stoch_term_dx
                    xi_y_array[:, t] = stoch_term_dy
                
                # For saving the data
                Dr_str = str.replace(str(Dr), '.', '_')
                H1_str = str.replace(str(H1), '.', '_')
                H2_str = str.replace(str(H2), '.', '_')
                Alpha_str = str.replace(str(alpha), '.', '_')
                
                # Define the save folder
                main_path = f'DATA/Alpha_{Alpha_str}/H1_{H1_str}_H2_{H2_str}/Dr_{Dr_str}/'
                os.makedirs(main_path, exist_ok=True)

                filename = f'Batch_Dr_{Dr_str}_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}.npz'
                path_save = os.path.join(main_path, filename)
                
                # Save the npz file with all the arrays
                np.savez(
                    path_save,
                    x_array=cp.asnumpy(x_array),
                    y_array=cp.asnumpy(y_array),
                    theta_array=cp.asnumpy(theta_array),
                    fmpi_x_array=cp.asnumpy(fmpi_x_array),
                    fmpi_y_array=cp.asnumpy(fmpi_y_array),
                    xi_x_array=cp.asnumpy(xi_x_array),
                    xi_y_array=cp.asnumpy(xi_y_array)
                )
                delta_time = time.time() - start_time_batch
                print(f"Completed Batch in {delta_time:.2f} seconds.")
    delta_time_model = time.time() - start_time_model
    print(f"Completed ALL Drs in {delta_time_model:.2f} seconds.")
delta_time_total = time.time() - start_time
print(f"Completed ALL in {delta_time_total:.2f} seconds.")
