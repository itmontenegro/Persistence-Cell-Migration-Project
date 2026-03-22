import os
import time
import numpy as np
import fbmutil as fbm

start_time = time.time()

# SET UP
# Model parameters
Dr_values = [1, 2, 5] # Diffusion coefficients
H_values = [0.5, 0.7, 0.9] # Hurst exponent values
Fm = 1
gamma_s = 1
alpha = 0.25
dt = 0.1
T = 100
Nts = int(T/dt) # Number of time steps
ftax = [0.1, 0.2, 0.5] # External force magnitudes

Nts2 = np.linspace(0, T, Nts) # Time vector for the simulation
# Position of the ORGANIZER CENTER (particle_x, particle_y)
particle_x = -100/np.sqrt(2) # x position of the organizer
particle_y = -100/np.sqrt(2) # y position of the organizer

# PARAMETER VALUES
# Number of simulations per parameter combination
num_simulations = 10 # Editable value for how many simulations per parameter set

# Iterate over all combinations of Dr_values, H_values, and ftax
for Dr in Dr_values:
    for H in H_values:
        for f in ftax:
            # Run the specified number of simulations with the current parameter set
            for sim_num in range(num_simulations):

                # Initialize variables for each simulation
                theta_array = np.zeros(Nts) # Angle array
                x_array = np.zeros(Nts) # x position array
                y_array = np.zeros(Nts) # y position array
                theta_org = np.zeros(Nts) # Angle to the organizer (target)
                fmpi_x_array = np.zeros(Nts) # Angle component in x direction
                fmpi_y_array = np.zeros(Nts) # Angle component in y direction
                xi_x_array = np.zeros(Nts) # Stochastic noise in x direction
                xi_y_array = np.zeros(Nts) # Stochastic noise in y direction
                theta_array[0] = np.random.uniform(0, 2*np.pi) # Random initial angle
                # theta_array[0] = np.random.unif # Initial angle set to 0 for consistency

                # Definition for Fractional Brownian Motion 1 -> White Noise Angle
                dfW1 = np.zeros(Nts)
                fbm_noise1 = fbm.fbm(Nts2, 0.5) # H=0.5 for white noise
                dfW1[1:] = np.diff(fbm_noise1)

                # Definition for Fractional Brownian Motion 2 -> Correlated Noise
                dfW2_x = np.zeros(Nts)
                fbm_noise2_x = fbm.fbm(Nts2, H)
                dfW2_x[1:] = np.diff(fbm_noise2_x)
                dfW2_y = np.zeros(Nts)
                fbm_noise2_y = fbm.fbm(Nts2, H)
                dfW2_y[1:] = np.diff(fbm_noise2_y)

                # Simulate cellular movement
                for t in range(1, Nts):
                    # Calculate the position differences (delta_x and delta_y)
                    delta_x = particle_x - x_array[t-1] # Difference in x
                    delta_y = particle_y - y_array[t-1] # Difference in y

                    # Compute target bearing and shortest signed turning error.
                    theta_target = np.arctan2(delta_y, delta_x)
                    theta_org[t] = theta_target
                    angle_error = np.arctan2(
                        np.sin(theta_target - theta_array[t-1]),
                        np.cos(theta_target - theta_array[t-1]),
                    )

                    # Calculate the variation in the angle (dtheta)
                    dtheta = f * angle_error + np.sqrt(2*Dr) * dfW1[t]
                    # Update the angle and keep it bounded in [-pi, pi]
                    theta_next = theta_array[t-1] + dtheta * dt
                    theta_array[t] = np.arctan2(np.sin(theta_next), np.cos(theta_next))

                    force_term_dx = Fm * np.cos(theta_array[t]) / gamma_s
                    stoch_term_dx = alpha * dfW2_x[t] / gamma_s
                    force_term_dy = Fm * np.sin(theta_array[t]) / gamma_s
                    stoch_term_dy = alpha * dfW2_y[t] / gamma_s
                    # Update the position
                    x_array[t] = x_array[t-1] + force_term_dx * dt + stoch_term_dx
                    y_array[t] = y_array[t-1] + force_term_dy * dt + stoch_term_dy
                    fmpi_x_array[t] = force_term_dx * dt
                    fmpi_y_array[t] = force_term_dy * dt
                    xi_x_array[t] = stoch_term_dx
                    xi_y_array[t] = stoch_term_dy
                
                # Store the data for the current simulation
                f_tax_str = str.replace(str(f), '.', '_')
                H_str = str.replace(str(H), '.', '_')
                Dr_str = str.replace(str(Dr), '.', '_')
                alpha_str = str.replace(str(alpha), '.', '_')
                simulation_data = np.column_stack((theta_array, theta_org, x_array, y_array, xi_x_array, xi_y_array, fmpi_x_array, fmpi_y_array))
                main_path = f'DATA/Alpha_{alpha_str}_f_tax_{f_tax_str}/H_{H_str}/Dr_{Dr_str}'
                os.makedirs(main_path, exist_ok=True)
                filename = f'Sim_{sim_num}_Dr_{Dr_str}_H_{H_str}_f_{f_tax_str}.csv'
                path_save = os.path.join(main_path, filename)
                np.savetxt(path_save, simulation_data, delimiter=',', header='theta,theta_org,x,y,xi_x,xi_y,fmpi_x,fmpi_y')

delta_time = time.time() - start_time
print(f"Total simulation time: {delta_time:.2f} seconds")