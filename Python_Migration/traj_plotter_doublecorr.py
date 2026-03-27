import os
import numpy as np
import matplotlib.pyplot as plt

base_path = 'DATA'
save_path = 'DATA_2025_DoubleCorr_Traj_Na0'
save_path_zoom = 'DATA_2025_DoubleCorr_ZoomTraj_Na0'

# Parametros de los datos a graficar

H1_values = [0.5]
H2_values = [0.75, 0.99]
alpha = 0.25
Dr = 1

# Numero de trayectorias a graficar por cada combinación de H1 y H2
n_traj = 6

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_zoom):
    os.makedirs(save_path_zoom)

for h1 in range(len(H1_values)):
    for h2 in range(len(H2_values)):
        # Strings para nombres
        H1 = H1_values[h1]
        H2 = H2_values[h2]
        H1_str = str.replace(str(H1), '.', '_')
        H2_str = str.replace(str(H2), '.', '_')
        Alpha_str = str.replace(str(alpha), '.', '_')
        Dr_str = str.replace(str(Dr), '.', '_')

        # Carpeta de cada combinación de H1 y H2
        folder_path = f'{base_path}/Alpha_{Alpha_str}/H1_{H1_str}_H2_{H2_str}/Dr_{Dr_str}/'

        batch = np.load(f'{folder_path}Batch_Dr_{Dr_str}_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}.npz')
        
        # Revisar que hayan suficientes trayectorias para graficar
        if batch['x_array'].shape[0] < n_traj:
            print(f"Warning: Not enough trajectories in batch for H1={H1}, H2={H2}. Found {batch['x_array'].shape[0]}, expected {n_traj}. Skipping.")
            continue
        
        # Seleccionar n_traj trayectorias aleatorias
        selected_indices = np.random.choice(batch['x_array'].shape[0], n_traj, replace=False)

        # Graficar las trayectorias seleccionadas
        x_selected = batch['x_array'][selected_indices, :]
        y_selected = batch['y_array'][selected_indices, :]

        # Use autumn colormap based on time
        time_array = np.arange(x_selected.shape[1])
        cmap = plt.cm.get_cmap('autumn_r')
        # Transparency
        ataxis = 0.5
        
        # Figure size is in inches, 800px at 300 dpi is 2.67 inches
        fig, ax = plt.subplots(figsize=(800/300, 800/300), dpi=300)

        for i in range(n_traj):
            ax.scatter(x_selected[i, :], y_selected[i, :], c=time_array, cmap=cmap, s=1, alpha=ataxis, edgecolors='none')
        
        # EDITING General plot
        ax.axis([-75, 75, -75, 75])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0) # TickLength = [0 0]

        # Thicken the border box (LineWidth = 3)
        for spine in ax.spines.values():
            spine.set_linewidth(3)

        # Save the general plot
        plt.savefig(f'{save_path}/Traj_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}_Dr_{Dr_str}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved trajectory plot for H1={H1}, H2={H2} at {save_path}/Traj_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}_Dr_{Dr_str}.png")

        # Zoomed plot
        fig, ax = plt.subplots(figsize=(800/300, 800/300), dpi=300)

        color_map = plt.get_cmap('gist_rainbow')

        for i in range(n_traj):
            color_map_index = color_map(i / n_traj)  # Cycle through the first 20 colors
            ax.scatter(x_selected[i, :], y_selected[i, :], c=[color_map_index], s=1.2, edgecolors='none', zorder=2)
            ax.plot(x_selected[i, :], y_selected[i, :], color=color_map_index, linewidth=0.8,zorder=1)  # Add lines between points
        
        ax.axis([-1.5, 1.5, -1.5, 1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0) # TickLength = [0 0]

        for spine in ax.spines.values():
            spine.set_linewidth(3)
        
        plt.savefig(f'{save_path_zoom}/ZoomTraj_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}_Dr_{Dr_str}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved zoomed trajectory plot for H1={H1}, H2={H2} at {save_path_zoom}/ZoomTraj_H1_{H1_str}_H2_{H2_str}_Alpha_{Alpha_str}_Dr_{Dr_str}.png")

         