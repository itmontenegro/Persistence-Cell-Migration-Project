import pandas as pd
import numpy as np
import fbmfast as fbm
import time
import os

start_time = time.time()

# SET UP
iterations = 50
Dr_values = [1]
H1_values = [0.5] # H for Equation of Angle
H2_values = [0.5] # H for Equation of Motion --> 2D (x and y)

# Model parameters
Fm = 1
gamma_s = 1
alpha = 0.25

# Simulation parameters
dt = 0.1
T = 100
Nts = int(T/dt)
Nts2 = np.linspace(0, T, Nts)

for Hval1 in range(len(H1_values)):
    start_time_model = time.time()
    for Hval2 in range(len(H2_values)):
        for Drval in range(len(Dr_values)):
            
            # Values of Hurst Index for Correlations
            H1 = H1_values[Hval1]
            H2 = H2_values[Hval2]
            Dr = Dr_values[Drval]
            
            # {iterations} sims per model
            for i in range(iterations):
                start_time_sim = time.time()

                # Definition for Fractional Brownian Motion 1 -> Noise Angle
                dfW1 = np.zeros(Nts)
                fbm_noise1 = fbm.fbm(Nts2, H1)
                dfW1[1:] = np.diff(fbm_noise1)

                # Definition for Fractional Brownian Motion 2 -> Correlated Noise
                dfW2_x = np.zeros(Nts)
                fbm_noise2_x = fbm.fbm(Nts2, H2)
                dfW2_x[1:] = np.diff(fbm_noise2_x)
                dfW2_y = np.zeros(Nts)
                fbm_noise2_y = fbm.fbm(Nts2, H2)
                dfW2_y[1:] = np.diff(fbm_noise2_y)

                # Initial conditions
                x_array = np.zeros(Nts)
                y_array = np.zeros(Nts)
                theta_array = np.zeros(Nts)
                theta_array[0] = np.random.uniform(0, 2*np.pi)
                
                # New test for saving values for quantificatino of alpha
                fmpi_x_array = np.zeros(Nts)
                fmpi_y_array = np.zeros(Nts)
                xi_x_array = np.zeros(Nts)
                xi_y_array = np.zeros(Nts)

                for t in range(1, Nts):
                    # AS OF THEORY CORRECTIONS MAY 2024 --> Xi(t) = dfW(t)/dt
                    # Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)/dt
                    dtheta = np.sqrt(2*Dr)*dfW1[t]
                    theta_array[t] = theta_array[t-1] + dtheta
                    # Equation of Motion: Fm*{cos(theta), sin(theta)} = gamma_s*v_i - alpha*dfW2[t]/dt
                    force_term_dx = Fm*np.cos(theta_array[t])/gamma_s
                    stoch_term_dx = alpha*dfW2_x[t]/gamma_s
                    force_term_dy = Fm*np.sin(theta_array[t])/gamma_s
                    stoch_term_dy = alpha*dfW2_y[t]/gamma_s
                    # Update of position
                    x_array[t] = x_array[t-1] + force_term_dx*dt + stoch_term_dx
                    y_array[t] = y_array[t-1] + force_term_dy*dt + stoch_term_dy
                    fmpi_x_array[t] = force_term_dx*dt
                    fmpi_y_array[t] = force_term_dy*dt
                    xi_x_array[t] = stoch_term_dx
                    xi_y_array[t] = stoch_term_dy
                
                # For saving the data
                Dr_str = str.replace(str(Dr), '.', '_')
                H_str = str.replace(str(H1), '.', '_')
                Alpha_str = str.replace(str(alpha), '.', '_')
                
                # Define the save folder
                alpha_path = f'DATA/Alpha_{Alpha_str}/'
                main_path = f'{alpha_path}H_{H_str}/'
                folder_name = f'data_Alpha_{Alpha_str}_Dr_{Dr_str}_H_{H_str}'
                filename = f'Sim_{i}_Dr_{Dr_str}_H_{H_str}.csv'
                path_save = main_path + folder_name + '/' + filename
                
                # Create folders if they do not exist
                # For Alpha
                if not os.path.exists(alpha_path):
                    os.makedirs(alpha_path)
                # For H
                if not os.path.exists(main_path):
                    os.makedirs(main_path)
                # For Dr
                if not os.path.exists(main_path + folder_name):
                    os.makedirs(main_path + folder_name)

                # Create a table with the data and variable names
                data = pd.DataFrame({
                    "X": x_array,
                    "Y": y_array,
                    "Theta": theta_array,
                    "Fmpi_X": fmpi_x_array,
                    "Fmpi_Y": fmpi_y_array,
                    "Xi_X": xi_x_array,
                    "Xi_Y": xi_y_array
                })
                # Save the table to a CSV file
                data.to_csv(path_save, index=False)
                delta_time = time.time() - start_time_sim
                print(f"Completed Sim in {delta_time:.2f} seconds.")
    delta_time_model = time.time() - start_time_model
    print(f"Completed ALL Drs in {delta_time_model:.2f} seconds.")
delta_time_total = time.time() - start_time
print(f"Completed ALL in {delta_time_total:.2f} seconds.")
