import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.csgraph as cpx_csgraph
import fbmfast_gpu as fbm
import time
import os

start_time = time.time()

boundary = 'none'       # Boundaries: 'periodic', 'hard', 'none'
init_position = 'random'   # Initial positions: 'cluster', 'random', 'grid'

# SET UP
replicates = 10             # Number of replicates for each parameter set
N_cells = 50                # Number of cells to simulate per replicate

DTYPE = cp.float32


# Multicellular parameters
R = 1.0                     # Cell radius
W_s = 1.0                   # Energy of adhesion cell-substrate
W_c = 1.0                   # Energy of adhesion cell-cell
f_cil = 0.5                   # Repolarization rate for CIL

L_box = 20                  # Size of side of the square boundary

Dr_values = [0.1] 
H1_values = [0.5]           # H for Equation of Angle
H2_values = [0.5, 0.99]     # H for Equation of Motion --> 2D (x and y)
Fm = 1
gamma_s = 1
gamma_c = 0 # Friction coefficient for cell-cell interactinos
alpha = 0.25
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
            Boundary_str = boundary.capitalize()
            fcil_str = str.replace(str(f_cil), '.', '_')

            main_path = f'DATA/Alpha_{Alpha_str}/H1_{H1_str}_H2_{H2_str}/Dr_{Dr_str}/Boundary_{Boundary_str}/fcil_{fcil_str}/'
            os.makedirs(main_path, exist_ok=True)

            print(f"Initializing set: Dr={Dr_values[Drval]}, H1={H1}, H2={H2}, Alpha={alpha}, Boundary={boundary}")

            # Initial conditions
            x_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            y_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            theta_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            angle_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            cluster_size_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            n_clusters_array = cp.zeros(Nts, dtype=DTYPE)
            fmpi_x_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            fmpi_y_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            xi_x_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            xi_y_array = cp.zeros((N_cells, Nts), dtype=DTYPE)
            dfW1 = cp.zeros((N_cells, Nts), dtype=DTYPE)
            dfW2_x = cp.zeros((N_cells, Nts), dtype=DTYPE)
            dfW2_y = cp.zeros((N_cells, Nts), dtype=DTYPE)

            for sim in range(replicates):
                filename = f'Sim_{sim}_Dr_{Dr_str}_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}_Boundary_{Boundary_str}_fcil_{fcil_str}.npz'
                path_save = os.path.join(main_path, filename)
                if os.path.exists(path_save):
                    print(f"File {filename} already exists. Skipping simulation {sim}.")
                    continue

                print(f"Running simulation {sim}")
                start_time_batch = time.time()

                # We reuse the arrays to avoid reallocation in each simulation
                for arr in (x_array, y_array, theta_array, angle_array, cluster_size_array,
                            n_clusters_array, fmpi_x_array, fmpi_y_array,
                            xi_x_array, xi_y_array, dfW1, dfW2_x, dfW2_y):
                    arr.fill(0)

                # fGn increments directly
                dfW1[:] = fbm.fgn_increments(Nts, H1, dt, N_cells, dtype=DTYPE)
                dfW2_x[:] = fbm.fgn_increments(Nts, H2, dt, N_cells, dtype=DTYPE)
                dfW2_y[:] = fbm.fgn_increments(Nts, H2, dt, N_cells, dtype=DTYPE)
                
                # Distribute the cells randomly in a square domain to avoid initial overlaps
                if init_position == 'random':
                    x_array[:, 0] = cp.random.uniform(-L_box, L_box, N_cells)
                    y_array[:, 0] = cp.random.uniform(-L_box, L_box, N_cells)
                elif init_position == 'cluster':
                    # Minimum separation distance to avoid overlaps
                    d_sep = 2.0 * R  
    
                    # Generate a slightly oversized grid to guarantee we have enough points 
                    # to slice a circular radius out of it safely
                    grid_size = int(cp.ceil(cp.sqrt(N_cells))) + 3
                    
                    # Generate 2D coordinate matrices natively on the GPU
                    cols, rows = cp.meshgrid(cp.arange(-grid_size, grid_size), cp.arange(-grid_size, grid_size))
                    cols_flat = cols.flatten()
                    rows_flat = rows.flatten()
                    
                    # Vectorized Hexagonal Equations
                    x_flat = (cols_flat + 0.5 * (rows_flat % 2)) * d_sep
                    y_flat = rows_flat * (d_sep * cp.sqrt(3) / 2)
                    
                    # Sort the indices based on proximity to the center (closest to furthest)
                    sort_indices = cp.argsort(x_flat**2 + y_flat**2)
                    
                    # Select the N_cells that are closest to the center
                    x_init = x_flat[sort_indices[:N_cells]]
                    y_init = y_flat[sort_indices[:N_cells]]
                    
                    # Re-center around the cluster's collective mass center
                    x_array[:, 0] = x_init - cp.mean(x_init)
                    y_array[:, 0] = y_init - cp.mean(y_init)
                elif init_position == 'grid':
                     # Square lattice, spacing 2R, centred on the origin
                    d_sep = 2.0 * R
                    side = int(cp.ceil(cp.sqrt(N_cells)))
                    gcol, grow = cp.meshgrid(cp.arange(side), cp.arange(side))
                    gx = gcol.flatten()[:N_cells].astype(cp.float64) * d_sep
                    gy = grow.flatten()[:N_cells].astype(cp.float64) * d_sep
                    x_array[:, 0] = gx - cp.mean(gx)
                    y_array[:, 0] = gy - cp.mean(gy)
                else:
                    raise ValueError(f"Unknown init_position: {init_position}")

                theta_array[:, 0] = cp.random.uniform(0, 2*cp.pi, N_cells)
                #velocity_x_array = cp.zeros((N_cells, Nts))
                #velocity_y_array = cp.zeros((N_cells, Nts))

                for t in range(1, Nts):
                    # Positions and angles at the actual time step
                    x_curr = x_array[:, t-1]
                    y_curr = y_array[:, t-1]
                    theta_curr = theta_array[:, t-1]

                    # Calculate velocities for the current time step
                    #if t > 1:
                        #velocity_x_array[:, t-1] = (x_curr - x_array[:, t-2]) / dt
                        #velocity_y_array[:, t-1] = (y_curr - y_array[:, t-2]) / dt

                    # Distance matrices and normal vectors
                    # Broadcasting to compute pairwise distances and angles
                    dX = x_curr[:, None] - x_curr[None, :]
                    dY = y_curr[:, None] - y_curr[None, :]

                    if boundary == 'periodic':
                        dX = dX - 2 * L_box * cp.round(dX / (2 * L_box))
                        dY = dY - 2 * L_box * cp.round(dY / (2 * L_box))
                    
                    # Distance matrix
                    dist = cp.sqrt(dX**2 + dY**2)
                    
                    # Avoid self-interaction by setting diagonal to infinity
                    diag_indices = cp.arange(N_cells)
                    dist[diag_indices, diag_indices] = cp.inf

                    # Interaction masking: only consider interactions if R <= dist <= 2R (dist <= 2*R due to physical limitations)
                    mask = (dist <= 2*R)

                    # Convert the mask to a sparse matrix
                    sparse_adj = cpx_sparse.csr_matrix(mask.astype(cp.float32))

                    # Get the number of clusters and labels mapping each cell to a cluster ID
                    n_clusters, labels = cpx_csgraph.connected_components(
                        sparse_adj, directed=False, connection='weak'
                    )

                    # Count how many cells are in each cluster ID
                    counts = cp.bincount(labels)
                    # Map the size of the cluster back to each cell
                    cluster_size_array[:, t] = counts[labels]
                    n_clusters_array[t] = n_clusters

                    dist_safe = cp.maximum(dist, 1e-10)  # Avoid division by zero for normal vector calculation

                    # Velocity differences
                    #dVx = velocity_x_array[:, t-1][:, None] - velocity_x_array[:, t-1][None, :]
                    #dVy = velocity_y_array[:, t-1][:, None] - velocity_y_array[:, t-1][None, :]

                    # Normal vectors
                    nX = cp.where(mask, dX / dist_safe, 0.0)
                    nY = cp.where(mask, dY / dist_safe, 0.0)

                    # Friction forces
                    #F_friction_x = cp.zeros((N_cells, N_cells))
                    #F_friction_y = cp.zeros((N_cells, N_cells))
                    #F_friction_x[mask] = -gamma_c * dVx[mask]
                    #F_friction_y[mask] = -gamma_c * dVy[mask]

                    # Cell-cell forces
                    F_cc = cp.where(mask, (2.0/R) * (W_s - (W_s + W_c) / R * (dist - R)), 0.0)

                    # Sum of forces for the neighboring cells
                    sum_F_x = cp.sum(F_cc * nX, axis=1) # + cp.sum(F_friction_x, axis=1)
                    sum_F_y = cp.sum(F_cc * nY, axis=1) # + cp.sum(F_friction_y, axis=1)

                    # CIL (Contact Inhibition of Locomotion)
                    N_contacts = cp.sum(mask, axis=1) # How many cells are in contact with each cell
                    has_neighbors = N_contacts > 0    # Boolean mask
                    N_contact_safe = cp.maximum(N_contacts, 1)  # Avoid division by zero

                    dir_X = cp.sum(dX * mask, axis=1) / N_contact_safe
                    dir_Y = cp.sum(dY * mask, axis=1) / N_contact_safe
                    theta_f = cp.arctan2(dir_Y, dir_X)  # Objective angle based on neighbors

                    # Angular difference adjusted to be in the range [-pi, pi] to ensure correct direction of rotation
                    dtheta = (theta_curr - theta_f + cp.pi) % (2 * cp.pi) - cp.pi

                    # Turning rate
                    dtheta_cil = cp.where(has_neighbors, -f_cil * dtheta * dt, 0.0)
                    
                    # Update of stochastic equations
                    # fBm noise
                    dtheta_noise = cp.sqrt(2*Dr) * dfW1[:, t]

                    # Update angle
                    theta_new = theta_curr + dtheta_cil + dtheta_noise
                    theta_array[:, t] = theta_new

                    # Intrinsic motor
                    motor_x = Fm * cp.cos(theta_new) / gamma_s
                    motor_y = Fm * cp.sin(theta_new) / gamma_s

                    # Update positions
                    x_new = x_curr + (motor_x + sum_F_x / gamma_s) * dt + alpha * dfW2_x[:, t] / gamma_s
                    y_new = y_curr + (motor_y + sum_F_y / gamma_s) * dt + alpha * dfW2_y[:, t] / gamma_s
                    
                    # Apply boundary conditions
                    if boundary == 'periodic':
                        if L_box > 0:  # Only apply if L_box is defined
                            x_new = (x_new + L_box) % (2 * L_box) - L_box
                            y_new = (y_new + L_box) % (2 * L_box) - L_box
                    elif boundary == 'hard':
                        x_new = cp.clip(x_new, -L_box, L_box)
                        y_new = cp.clip(y_new, -L_box, L_box)

                    x_array[:, t] = x_new
                    y_array[:, t] = y_new
                
                fmpi_x_array[:, 1:] = Fm * cp.cos(theta_array[:, 1:]) * dt / gamma_s
                fmpi_y_array[:, 1:] = Fm * cp.sin(theta_array[:, 1:]) * dt / gamma_s
                xi_x_array[:] = alpha * dfW2_x / gamma_s
                xi_y_array[:] = alpha * dfW2_y / gamma_s
                
                vx = cp.diff(x_array, axis=1)
                vy = cp.diff(y_array, axis=1)
                v1x, v2x = vx[:, :-1], vx[:, 1:]
                v1y, v2y = vy[:, :-1], vy[:, 1:]
                dot_product = v1x * v2x + v1y * v2y
                n1 = cp.sqrt(v1x**2 + v1y**2)
                n2 = cp.sqrt(v2x**2 + v2y**2)
                cos_angle = cp.clip(dot_product / (n1 * n2), -1.0, 1.0)
                angle_array[:, 2:] = cp.arccos(cos_angle)

                # Save the npz file with all the arrays
                np.savez(
                    path_save,
                    x_array=cp.asnumpy(x_array),
                    y_array=cp.asnumpy(y_array),
                    theta_array=cp.asnumpy(theta_array),
                    angle_array=cp.asnumpy(angle_array),
                    cluster_size_array=cp.asnumpy(cluster_size_array),
                    n_clusters_array=cp.asnumpy(n_clusters_array),
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
