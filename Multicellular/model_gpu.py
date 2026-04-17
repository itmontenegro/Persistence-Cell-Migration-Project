import numpy as np
import cupy as cp
import fbmfast_gpu as fbm
import time
import os

start_time = time.time()

boundary = 'hard' # Boundaries: 'periodic', 'hard', 'none'

# SET UP
replicates = 1     # Number of replicates for each parameter set
N_cells = 50    # Number of cells to simulate per replicate

# Multicellular parameters
R = 1.0             # Cell radius
W_s = 1.0           # Energy of adhesion cell-substrate
W_c = 1.0          # Energy of adhesion cell-cell
f_cil = 0.1         # Repolarization rate for CIL

distribution = 4

Dr_values = [0.1] 
H1_values = [0.5]  # H for Equation of Angle
H2_values = [0.5]  # H for Equation of Motion --> 2D (x and y)
Fm = 1
gamma_s = 1
gamma_c = 0 # Friction coefficient for cell-cell interactinos
alpha = 0
dt = 0.1
T = 100
Nts = int(T/dt)
Nts2 = np.linspace(0, T, Nts)

for Hval1 in range(len(H1_values)):
    start_time_model = time.time()
    for Hval2 in range(len(H2_values)):
        for Drval in range(len(Dr_values)):
            print(Drval, Hval1, Hval2)
            print(f"Running batch: Dr={Dr_values[Drval]}, H1={H1_values[Hval1]}, H2={H2_values[Hval2]}")
            start_time_batch = time.time()
            
            # Values of Hurst Index for Correlations
            H1 = H1_values[Hval1]
            H2 = H2_values[Hval2]
            Dr = Dr_values[Drval]

            # For saving the data
            Dr_str = str.replace(str(Dr_values[Drval]), '.', '_')
            H1_str = str.replace(str(H1), '.', '_')
            H2_str = str.replace(str(H2), '.', '_')
            Alpha_str = str.replace(str(alpha), '.', '_')

            main_path = f'DATA/Alpha_{Alpha_str}/H1_{H1_str}_H2_{H2_str}/Dr_{Dr_str}/'
            os.makedirs(main_path, exist_ok=True)

            print(f"Initializing set: Dr={Dr_values[Drval]}, H1={H1}, H2={H2}, Alpha={alpha}")

            for sim in range(replicates):
                filename = f'Sim_{sim}_Dr_{Dr_str}_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}.npz'
                path_save = os.path.join(main_path, filename)
                if os.path.exists(path_save):
                    print(f"File {filename} already exists. Skipping simulation {sim}.")
                    continue

                print(f"Running simulation {sim}")
                start_time_batch = time.time()

                # Generation of fBm noise
                dfW1 = cp.zeros((N_cells, Nts))
                fbm_noise1 = fbm.fbm_batch(Nts, H1, dt, N_cells)
                dfW1[:, 1:] = cp.diff(fbm_noise1)

                dfW2_x = cp.zeros((N_cells, Nts))
                fbm_noise2_x = fbm.fbm_batch(Nts, H2, dt, N_cells)
                dfW2_x[:, 1:] = cp.diff(fbm_noise2_x)

                dfW2_y = cp.zeros((N_cells, Nts))
                fbm_noise2_y = fbm.fbm_batch(Nts, H2, dt, N_cells)
                dfW2_y[:, 1:] = cp.diff(fbm_noise2_y)

                # Initial conditions
                x_array = cp.zeros((N_cells, Nts))
                y_array = cp.zeros((N_cells, Nts))
                theta_array = cp.zeros((N_cells, Nts))
                
                # Distribute the cells randomly in a square domain to avoid initial overlaps
                L_box = R * N_cells / distribution  # Size of the box to distribute cells
                x_array[:, 0] = cp.random.uniform(-L_box, L_box, N_cells)
                y_array[:, 0] = cp.random.uniform(-L_box, L_box, N_cells)
                theta_array[:, 0] = cp.random.uniform(0, 2*cp.pi, N_cells)
                velocity_x_array = cp.zeros((N_cells, Nts))
                velocity_y_array = cp.zeros((N_cells, Nts))

                fmpi_x_array = cp.zeros((N_cells, Nts))
                fmpi_y_array = cp.zeros((N_cells, Nts))
                xi_x_array = cp.zeros((N_cells, Nts))
                xi_y_array = cp.zeros((N_cells, Nts))

                for t in range(1, Nts):
                    # Positions and angles at the actual time step
                    x_curr = x_array[:, t-1]
                    y_curr = y_array[:, t-1]
                    theta_curr = theta_array[:, t-1]

                    # Calculate velocities for the current time step
                    if t > 1:
                        velocity_x_array[:, t-1] = (x_curr - x_array[:, t-2]) / dt
                        velocity_y_array[:, t-1] = (y_curr - y_array[:, t-2]) / dt

                    # Distance matrices and normal vectors
                    # Broadcasting to compute pairwise distances and angles
                    dX = x_curr[:, None] - x_curr[None, :]
                    dY = y_curr[:, None] - y_curr[None, :]

                    # Distance matrix
                    dist = cp.sqrt(dX**2 + dY**2)
                    
                    # Avoid self-interaction by setting diagonal to infinity
                    cp.fill_diagonal(dist, cp.inf)

                    # Interaction masking: only consider interactions if R <= dist <= 2R (dist <= 2*R due to physical limitations)
                    mask = (dist <= 2*R)

                    # Velocity differences
                    dVx = velocity_x_array[:, t-1][:, None] - velocity_x_array[:, t-1][None, :]
                    dVy = velocity_y_array[:, t-1][:, None] - velocity_y_array[:, t-1][None, :]

                    # Normal vectors
                    nX = cp.zeros((N_cells, N_cells))
                    nY = cp.zeros((N_cells, N_cells))
                    nX[mask] = dX[mask] / dist[mask]
                    nY[mask] = dY[mask] / dist[mask]

                    # Friction forces
                    F_friction_x = cp.zeros((N_cells, N_cells))
                    F_friction_y = cp.zeros((N_cells, N_cells))
                    F_friction_x[mask] = -gamma_c * dVx[mask]
                    F_friction_y[mask] = -gamma_c * dVy[mask]

                    # Cell-cell forces
                    F_cc = cp.zeros((N_cells, N_cells))
                    F_cc[mask] = (2.0/R) * (W_s - (W_s + W_c) / R * (dist[mask] - R))

                    # Sum of forces for the neighboring cells
                    sum_F_x = cp.sum(F_cc * nX, axis=1) + cp.sum(F_friction_x, axis=1)
                    sum_F_y = cp.sum(F_cc * nY, axis=1) + cp.sum(F_friction_y, axis=1)

                    # CIL (Contact Inhibition of Locomotion)
                    N_contacts = cp.sum(mask, axis=1) # How many cells are in contact with each cell
                    has_neighbors = N_contacts > 0    # Boolean mask

                    dtheta_cil = cp.zeros(N_cells)

                    if cp.any(has_neighbors):
                        # Average of the positions of the neighboring cells
                        sum_X_neighbors = cp.sum(x_curr[None, :] * mask, axis=1)
                        sum_Y_neighbors = cp.sum(y_curr[None, :] * mask, axis=1)

                        mean_X = sum_X_neighbors[has_neighbors] / N_contacts[has_neighbors]
                        mean_Y = sum_Y_neighbors[has_neighbors] / N_contacts[has_neighbors]

                        # Vector that "escapes" from the center of mass of the contact neighbors
                        dir_X = x_curr[has_neighbors] - mean_X
                        dir_Y = y_curr[has_neighbors] - mean_Y

                        # Objective angle
                        theta_f = cp.arctan2(dir_Y, dir_X)

                        # Angular difference adjusted to be in the range [-pi, pi] to ensure correct direction of rotation
                        dtheta = (theta_curr[has_neighbors] - theta_f + cp.pi) % (2 * cp.pi) - cp.pi

                        # Turning rate
                        dtheta_cil[has_neighbors] = -f_cil * dtheta * dt
                    
                    # Update of stochastic equations
                    # fBm noise
                    dtheta_noise = cp.sqrt(2*Dr) * dfW1[:, t]

                    # Update angle
                    theta_array[:, t] = theta_curr + dtheta_cil + dtheta_noise

                    stoch_term_dx = alpha * dfW2_x[:, t] / gamma_s
                    stoch_term_dy = alpha * dfW2_y[:, t] / gamma_s

                    # Intrinsic motor
                    motor_x = Fm * cp.cos(theta_curr) / gamma_s
                    motor_y = Fm * cp.sin(theta_curr) / gamma_s

                    # Update positions
                    x_array[:, t] = x_curr + (motor_x + sum_F_x / gamma_s) * dt + stoch_term_dx
                    y_array[:, t] = y_curr + (motor_y + sum_F_y / gamma_s) * dt + stoch_term_dy
                    
                    # Apply boundary conditions
                    if boundary == 'periodic':
                        if L_box > 0:  # Only apply if L_box is defined
                            x_array[:, t] = (x_array[:, t] + L_box) % (2 * L_box) - L_box
                            y_array[:, t] = (y_array[:, t] + L_box) % (2 * L_box) - L_box
                    elif boundary == 'hard':
                        x_array[:, t] = cp.clip(x_array[:, t], -L_box, L_box)
                        y_array[:, t] = cp.clip(y_array[:, t], -L_box, L_box)

                    # Save values for analysis
                    fmpi_x_array[:, t] = motor_x * dt
                    fmpi_y_array[:, t] = motor_y * dt
                    xi_x_array[:, t] = stoch_term_dx
                    xi_y_array[:, t] = stoch_term_dy    
                
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
                print(f"Completed simulation {sim} in {delta_time:.2f} seconds.")
    delta_time_model = time.time() - start_time_model
    print(f"Completed set H1_{H1_str}_H2_{H2_str} in {delta_time_model:.2f} seconds.")
delta_time_total = time.time() - start_time
print(f"Completed ALL in {delta_time_total:.2f} seconds.")
