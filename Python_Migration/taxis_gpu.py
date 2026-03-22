import os
import time
import numpy as np
import cupy as cp 
import fbmfast_gpu as fbm

start_time = time.time()

# SET UP
# Model parameters
Dr_values = [0.1, 1, 10] # Diffusion coefficients
H_values = [0.5, 0.75, 0.99] # Hurst exponent values
Fm = 1
gamma_s = 1
alpha = 0.25
dt = 0.1
T = 100
Nts = int(T/dt) # Number of time steps
ftax = [0, 0.2, 0.5] # External force magnitudes

Nts2 = np.linspace(0, T, Nts) # Time vector for the simulation
# Position of the ORGANIZER CENTER (particle_x, particle_y)
particle_x = -100/np.sqrt(2) # x position of the organizer
particle_y = -100/np.sqrt(2) # y position of the organizer

# PARAMETER VALUES
# Number of simulations per parameter combination
num_simulations = 1000 # Editable value for how many simulations per parameter set

# Iterate over all combinations of Dr_values, H_values, and ftax
for Dr in Dr_values:
    for H in H_values:
        for f in ftax:
            print(f"Running batch: Dr={Dr}, H={H}, f={f}")
            start_time_batch = time.time()

            # Initialize variables for each batch in 2D arrays on GPU
            theta_array = cp.zeros((num_simulations, Nts)) # Angles array
            x_array = cp.zeros((num_simulations, Nts)) # x positions array
            y_array = cp.zeros((num_simulations, Nts)) # y positions array
            theta_org = cp.zeros((num_simulations, Nts)) # Angles to the organizer (target)
            fmpi_x_array = cp.zeros((num_simulations, Nts)) # Angles component in x direction
            fmpi_y_array = cp.zeros((num_simulations, Nts)) # Angles component in y direction
            xi_x_array = cp.zeros((num_simulations, Nts)) # Stochastic noises in x direction
            xi_y_array = cp.zeros((num_simulations, Nts)) # Stochastic noises in y direction
            theta_array[:, 0] = cp.random.uniform(0, 2*cp.pi, size=num_simulations) # Random initial angles

            # Pregenerate all noise for the batch on GPU
            # Definition for Fractional Brownian Motion 1 -> White Noise Angle
            dfW1 = cp.zeros((num_simulations, Nts))
            fbm_noise1 = fbm.fbm_batch(Nts, 0.5, dt, num_simulations) # H=0.5 for white noise
            dfW1[:, 1:] = cp.diff(fbm_noise1, axis=1)


            # Definition for Fractional Brownian Motion 2 -> Correlated Noise
            dfW2_x = cp.zeros((num_simulations, Nts))
            fbm_noise2_x = fbm.fbm_batch(Nts, H, dt, num_simulations)
            dfW2_x[:, 1:] = cp.diff(fbm_noise2_x, axis=1)

            dfW2_y = cp.zeros((num_simulations, Nts))
            fbm_noise2_y = fbm.fbm_batch(Nts, H, dt, num_simulations)
            dfW2_y[:, 1:] = cp.diff(fbm_noise2_y, axis=1)

            # Simulate cellular movement
            for t in range(1, Nts):
                # Calculate the position differences (delta_x and delta_y)
                delta_x = particle_x - x_array[:, t-1] # Difference in x
                delta_y = particle_y - y_array[:, t-1] # Difference in y

                # Compute target bearing and shortest signed turning error.
                theta_target = cp.arctan2(delta_y, delta_x)
                theta_org[:, t] = theta_target
                
                angle_error = cp.arctan2(
                    cp.sin(theta_target - theta_array[:, t-1]),
                    cp.cos(theta_target - theta_array[:, t-1]),
                )

                # Calculate the variation in the angle (dtheta)
                dtheta = f * angle_error * dt + cp.sqrt(2*Dr) * dfW1[:, t]
                theta_next = theta_array[:, t-1] + dtheta
                theta_array[:, t] = cp.arctan2(cp.sin(theta_next), cp.cos(theta_next))

                force_term_dx = Fm * cp.cos(theta_array[:, t]) / gamma_s
                stoch_term_dx = alpha * dfW2_x[:, t] / gamma_s
                force_term_dy = Fm * cp.sin(theta_array[:, t]) / gamma_s
                stoch_term_dy = alpha * dfW2_y[:, t] / gamma_s
                
                # Update the position
                x_array[:, t] = x_array[:, t-1] + force_term_dx * dt + stoch_term_dx
                y_array[:, t] = y_array[:, t-1] + force_term_dy * dt + stoch_term_dy
                
                fmpi_x_array[:, t] = force_term_dx * dt
                fmpi_y_array[:, t] = force_term_dy * dt
                xi_x_array[:, t] = stoch_term_dx
                xi_y_array[:, t] = stoch_term_dy
            
            # Store the data for the current simulation
            # Move data back to CPU for saving
            f_tax_str = str.replace(str(f), '.', '_')
            H_str = str.replace(str(H), '.', '_')
            Dr_str = str.replace(str(Dr), '.', '_')
            alpha_str = str.replace(str(alpha), '.', '_')

            main_path = f'DATA/Alpha_{alpha_str}_f_tax_{f_tax_str}/H_{H_str}/Dr_{Dr_str}'
            os.makedirs(main_path, exist_ok=True)

            # Name the file for the entire batch
            filename = f'Batch_Dr_{Dr_str}_H_{H_str}_f_{f_tax_str}.npz'
            path_save = os.path.join(main_path, filename)

            np.savez(path_save, 
                theta=theta_array.get(),
                theta_org=theta_org.get(),
                x=x_array.get(),
                y=y_array.get(),
                xi_x=xi_x_array.get(),
                xi_y=xi_y_array.get(),
                fmpi_x=fmpi_x_array.get(),
                fmpi_y=fmpi_y_array.get()
            )
            delta_time_batch = time.time() - start_time_batch
            print(f"Batch completed in {delta_time_batch:.2f} seconds")

delta_time = time.time() - start_time
print(f"Total simulation time: {delta_time:.2f} seconds")   