import cluster_cpu
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.csgraph as cpx_csgraph
import fbmbatch_gpu as fbm
import time
import os

start_time = time.time()

boundary_values = ['none', 'periodic']       # Boundaries: 'periodic', 'hard', 'none'
init_position_values = ['random', 'cluster']   # Initial positions: 'cluster', 'random', 'grid'
f_cil_values = [0.5]     # Repolarization rate for CIL

DTYPE = cp.float32

# SET UP
replicates = 10              # Number of replicates for each parameter set
max_batch_size = 10         # Maximum number of replicates to run simultaneously in VRAM
N_cells = 50                # Number of cells to simulate per replicate

# Multicellular parameters
R = 1.0                     # Cell radius
W_s = 1.0                   # Energy of adhesion cell-substrate
W_c = 1.0                   # Energy of adhesion cell-cell

L_box = 20                  # Size of side of the square boundary

Dr_values = [0.1] 
H1_values = [0.5]           # H for Equation of Angle
H2_values = [0.5, 0.99]     # H for Equation of Motion --> 2D (x and y)
Fm = 1
gamma_s = 1
gamma_c = 0                 # Friction coefficient for cell-cell interactinos
alpha = 0.25
dt = 0.1
T = 100
Nts = int(T/dt)
Nts2 = cp.linspace(0, T, Nts)

for boundary in boundary_values:
    start_time_model = time.time()
    for init_position in init_position_values:
        for f_cil in f_cil_values:
            for Hval1 in range(len(H1_values)):
                for Hval2 in range(len(H2_values)):
                    for Drval in range(len(Dr_values)):
                        print(Drval, Hval1, Hval2)
                        print(f"Running set: Dr={Dr_values[Drval]}, H1={H1_values[Hval1]}, H2={H2_values[Hval2]}, fcil = {f_cil}")
                        
                        # Values of Hurst Index for Correlations
                        H1 = H1_values[Hval1]
                        H2 = H2_values[Hval2]
                        Dr = Dr_values[Drval]

                        # For saving the data
                        Dr_str = str.replace(str(Dr_values[Drval]), '.', '_')
                        H1_str = str.replace(str(H1), '.', '_')
                        H2_str = str.replace(str(H2), '.', '_')
                        Alpha_str = str.replace(str(alpha), '.', '_')
                        Boundary_str = boundary.capitalize()
                        fcil_str = str.replace(str(f_cil), '.', '_')
                        Start_str = str.replace(str(init_position), '.', '_')

                        main_path = f'DATA/Alpha_{Alpha_str}/H1_{H1_str}_H2_{H2_str}/Dr_{Dr_str}/Boundary_{Boundary_str}/fcil_{fcil_str}/Start_{Start_str}'
                        os.makedirs(main_path, exist_ok=True)

                        print(f"Initializing set: Dr={Dr_values[Drval]}, H1={H1}, H2={H2}, Alpha={alpha}, Boundary={boundary}")

                        # Pre-check to determine which replicates actually need to be run
                        sims_to_run = []
                        for sim in range(replicates):
                            filename = f'Sim_{sim}_Dr_{Dr_str}_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}_Boundary_{Boundary_str}_fcil_{fcil_str}_Start{Start_str}.npz'
                            path_save = os.path.join(main_path, filename)
                            if not os.path.exists(path_save):
                                sims_to_run.append((sim, path_save))

                        if not sims_to_run:
                            print(f"All replicates for H1={H1} H2={H2} already exist. Skipping.")
                            continue

                        # Preallocation of arrays to mitigate GPU memory allocation overhead
                        x_array = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)
                        y_array = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)
                        theta_array = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)

                        # Array to save the angle between three positions.
                        angle_array = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)

                        fmpi_x_array = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)
                        fmpi_y_array = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)
                        xi_x_array = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)
                        xi_y_array = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)

                        dfW1 = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)
                        dfW2_x = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)
                        dfW2_y = cp.zeros((max_batch_size, N_cells, Nts), dtype=DTYPE)

                        diag_indices = cp.arange(N_cells)

                        # Chunk the runs into VRAM-safe batches
                        for chunk_start in range(0, len(sims_to_run), max_batch_size):
                            batch_chunk = sims_to_run[chunk_start:chunk_start + max_batch_size]
                            c = len(batch_chunk)
                            
                            print(f"Running chunk of {c} replicates in parallel...")
                            start_time_batch = time.time()

                            for arr in (x_array, y_array, theta_array, angle_array,
                                        fmpi_x_array, fmpi_y_array, xi_x_array, xi_y_array,
                                        dfW1, dfW2_x, dfW2_y):
                                arr.fill(0)  # Clear arrays in-place to reuse memory on the GPU

                            # fGn increments directly (leading zero when t=0)
                            dfW1[:c] = fbm.fgn_increments(Nts, H1, dt, c, N_cells, dtype=DTYPE)
                            dfW2_x[:c] = fbm.fgn_increments(Nts, H2, dt, c, N_cells, dtype=DTYPE)
                            dfW2_y[:c] = fbm.fgn_increments(Nts, H2, dt, c, N_cells, dtype=DTYPE)

                            if init_position == 'random':
                                x_array[:c, :, 0] = cp.random.uniform(-L_box, L_box, (c, N_cells))
                                y_array[:c, :, 0] = cp.random.uniform(-L_box, L_box, (c, N_cells))
                            elif init_position == 'cluster':
                                # Densest-packed hexagonal core, N_cells nearest the origin
                                d_sep = 2.0 * R
                                grid_size = int(cp.ceil(cp.sqrt(N_cells))) + 3
                                cols, rows = cp.meshgrid(cp.arange(-grid_size, grid_size),
                                                        cp.arange(-grid_size, grid_size))
                                cols_flat = cols.flatten()
                                rows_flat = rows.flatten()
                                x_flat = (cols_flat + 0.5 * (rows_flat % 2)) * d_sep
                                y_flat = rows_flat * (d_sep * cp.sqrt(3) / 2)
                                sort_idx = cp.argsort(x_flat ** 2 + y_flat ** 2)
                                x_init = x_flat[sort_idx[:N_cells]]
                                y_init = y_flat[sort_idx[:N_cells]]
                                x_init = x_init - cp.mean(x_init)
                                y_init = y_init - cp.mean(y_init)
                                x_array[:c, :, 0] = cp.tile(x_init, (c, 1))
                                y_array[:c, :, 0] = cp.tile(y_init, (c, 1))
                            elif init_position == 'grid':
                                d_sep = 2.0 * R
                                side = int(cp.ceil(cp.sqrt(N_cells)))
                                gcol, grow = cp.meshgrid(cp.arange(side), cp.arange(side))
                                gx = gcol.flatten()[:N_cells].astype(DTYPE) * d_sep
                                gy = grow.flatten()[:N_cells].astype(DTYPE) * d_sep
                                x_array[:c, :, 0] = cp.tile(gx - cp.mean(gx), (c, 1))
                                y_array[:c, :, 0] = cp.tile(gy - cp.mean(gy), (c, 1))
                            else:
                                raise ValueError(f"Unknown init_position: {init_position}")

                            for t in range(1, Nts):
                                # Positions and angles at the actual time step
                                x_curr = x_array[:c, :, t-1]
                                y_curr = y_array[:c, :, t-1]
                                theta_curr = theta_array[:c, :, t-1]

                                # Distance matrices and normal vectors
                                # Broadcasting to compute pairwise distances and angles
                                dX = x_curr[:, :, None] - x_curr[:, None, :]
                                dY = y_curr[:, :, None] - y_curr[:, None, :]

                                if boundary == 'periodic':
                                    dX = dX - 2 * L_box * cp.round(dX / (2 * L_box))
                                    dY = dY - 2 * L_box * cp.round(dY / (2 * L_box))
                                
                                # Distance matrix
                                dist = cp.sqrt(dX**2 + dY**2)
                                
                                # Avoid self-interaction by setting diagonal to infinity
                                dist[:, diag_indices, diag_indices] = cp.inf

                                # Interaction masking: only consider interactions if R <= dist <= 2R (dist <= 2*R due to physical limitations)
                                mask = (dist <= 2*R)
                                mask_f32 = mask.astype(cp.float32)

                                # Avoid division by zero for normal vector calculation
                                dist_safe = cp.maximum(dist, 1e-10)  
                                
                                # Normal vectors
                                nX = cp.where(mask, dX / dist_safe, 0.0)
                                nY = cp.where(mask, dY / dist_safe, 0.0)

                                # Cell-cell forces evaluated densely over the 3D tensor to eliminate boolean slicing
                                F_cc = cp.where(mask, (2.0/R) * (W_s - (W_s + W_c) / R * (dist - R)), 0.0)
                                # Sum of forces for the neighboring cells
                                sum_F_x = cp.sum(F_cc * nX, axis=2)
                                sum_F_y = cp.sum(F_cc * nY, axis=2)

                                # CIL (Contact Inhibition of Locomotion)
                                N_contacts = cp.sum(mask, axis=2) # How many cells are in contact with each cell
                                has_neighbors = N_contacts > 0    # Boolean mask
                                N_contact_safe = cp.maximum(N_contacts, 1)  # Avoid division by zero

                                # Escape direction from neighbors
                                dir_X = cp.sum(dX * mask, axis=2) / N_contact_safe
                                dir_Y = cp.sum(dY * mask, axis=2) / N_contact_safe

                                # Objective angle
                                theta_f = cp.arctan2(dir_Y, dir_X)

                                # Angular difference adjusted to be in the range [-pi, pi] to ensure correct direction of rotation
                                dtheta = (theta_curr - theta_f + cp.pi) % (2 * cp.pi) - cp.pi

                                # Turning rate evaluated algebraically with cp.where to circumvent GPU-CPU pipelines
                                dtheta_cil = cp.where(has_neighbors, -f_cil * dtheta * dt, 0.0)
                                
                                # Update of stochastic equations
                                # fBm noise
                                dtheta_noise = cp.sqrt(2*Dr) * dfW1[:c, :, t]

                                # Update angle
                                theta_new = theta_curr + dtheta_cil + dtheta_noise
                                theta_array[:c, :, t] = theta_new

                                # Intrinsic motor
                                motor_x = Fm * cp.cos(theta_new) / gamma_s
                                motor_y = Fm * cp.sin(theta_new) / gamma_s

                                # Update positions
                                x_new = x_curr + (motor_x + sum_F_x / gamma_s) * dt + alpha * dfW2_x[:c, :, t] / gamma_s
                                y_new = y_curr + (motor_y + sum_F_y / gamma_s) * dt + alpha * dfW2_y[:c, :, t] / gamma_s

                                # Apply boundary conditions
                                if boundary == 'periodic':
                                    if L_box > 0:  # Only apply if L_box is defined
                                        x_new = (x_new + L_box) % (2 * L_box) - L_box
                                        y_new = (y_new + L_box) % (2 * L_box) - L_box
                                elif boundary == 'hard':
                                    x_new = cp.clip(x_new, -L_box, L_box)
                                    y_new = cp.clip(y_new, -L_box, L_box)
                                
                                x_array[:c, :, t] = x_new
                                y_array[:c, :, t] = y_new

                            # Save values for analysis
                            fmpi_x_array[:c, :, t] = motor_x * dt
                            fmpi_y_array[:c, :, t] = motor_y * dt
                            xi_x_array[:c] = alpha * dfW2_x[:c] / gamma_s
                            xi_y_array[:c] = alpha * dfW2_y[:c] / gamma_s

                            # Relative turning angle between consecutive time steps
                            vx = cp.diff(x_array[:c], axis=2)
                            vy = cp.diff(y_array[:c], axis=2)
                            v1x, v2x = vx[:, :, :-1], vx[:, :, 1:]
                            v1y, v2y = vy[:, :, :-1], vy[:, :, 1:]
                            dot_product = v1x * v2x + v1y * v2y
                            norm_v1 = cp.sqrt(v1x**2 + v1y**2)
                            norm_v2 = cp.sqrt(v2x**2 + v2y**2)
                            cos_angle = cp.clip(dot_product / (norm_v1 * norm_v2 + 1e-10), -1.0, 1.0)  # Avoid division by zero
                            angle_array[:c, :, 2:] = cp.arccos(cos_angle)

                            # Save the npz file with all the arrays mapped back to individual files per simulation index
                            # And calculate cluster sizes and number of clusters for each replicate using the CPU function
                            for batch_idx, (sim_id, path_save) in enumerate(batch_chunk):
                                x_np = cp.asnumpy(x_array[batch_idx])
                                y_np = cp.asnumpy(y_array[batch_idx])
                                cluster_size_array, n_clusters_array = cluster_cpu.cluster_arrays(x_np, y_np, R, boundary, L_box)
                                np.savez(
                                    path_save,
                                    x_array=x_np,
                                    y_array=y_np,
                                    theta_array=cp.asnumpy(theta_array[batch_idx]),
                                    angle_array=cp.asnumpy(angle_array[batch_idx]),
                                    cluster_size_array=cluster_size_array,
                                    n_clusters_array=n_clusters_array,
                                    fmpi_x_array=cp.asnumpy(fmpi_x_array[batch_idx]),
                                    fmpi_y_array=cp.asnumpy(fmpi_y_array[batch_idx]),
                                    xi_x_array=cp.asnumpy(xi_x_array[batch_idx]),
                                    xi_y_array=cp.asnumpy(xi_y_array[batch_idx])
                                )
                            delta_time = time.time() - start_time_batch
                            print(f"Completed batch chunk of {c} simulations in {delta_time:.2f} seconds.")

    delta_time_model = time.time() - start_time_model
    print(f"Completed set H1_{H1_str}_H2_{H2_str} in {delta_time_model:.2f} seconds.")
delta_time_total = time.time() - start_time
print(f"Completed ALL in {delta_time_total:.2f} seconds.")