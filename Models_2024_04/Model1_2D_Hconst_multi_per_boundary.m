close all; clear all; clc;
start_time_full = tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Euler-Maruyama method on Smeets Model for multiple cells             %
%                  MULTI-CELL MODEL WITH PERIODIC BOUNDARY                       %
%                                                                                %
%   1) Eq. of Motion: Fm*{cos(θ),sin(θ)} = gamma_s*v_i + dfW2(t)                 %
%   2) Eq. Repolarization SDE: d(θ)/dt = sqrt(2*Dr) * dfW1(t)                    %
%   3) Eq. Repolarization direction: p_i_f = -Σ[(2*a_ij^2/d_ij^3)*(c_ij-x_ij)]   %
%   4) Eq. direction vector: p_i_f = (cosθ*_i , sinθ*_i)^T                       %
%   5) Eq. Repolarization CIL: d(θ)/dt = -fCIL(θi - θ*_i) + ξ * sqrt(2*Dr)       %
%   6) Eq. of Intercellular Force: 2/R * [Ws - (Ws + Wc)/R * (dij - R)]          %
%   7) Reflective Boundary Conditions                                            %
%                                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SET UP
Dr_values = [1];
num_cells = 1;  % Number of cells
colors = lines(num_cells);  % Colors for each cell
Wc = 1; % Cell-cell adhesion energy 
Ws = 1; % Cell-substrate adhesion energy 
R = 1;  % Cell radius
interaction_threshold = 2 * R;  % cells touching

% Simulation domain boundaries
L = 30;  % Simulation square domain [0, L] x [0, L]

for dr_val = 1:length(Dr_values)
    start_time_model = tic; 
    % Model Parameters
    Fm = 1; 
    gamma_s = 1; 
    Dr = Dr_values(dr_val); 
    alpha = 0;
    % Simulation Parameters
    dt = 0.1; 
    T = 100; 
    Nts = floor(T/dt); 
    Nts2 = linspace(0, T, Nts); 
    
    % Pre-allocate matrices for all cells
    x_array = zeros(Nts, num_cells); 
    y_array = zeros(Nts, num_cells); 
    theta_array = zeros(Nts, num_cells);

    for cell_idx = 1:num_cells    
        % Definition for Fractional Brownian Motion for orientation (theta)
        dfW1 = zeros(Nts,1); 
        fbm_noise1 = fbm(Nts2, 0.5); % fBm with constant H = 0.5
        dfW1(2:end) = diff(fbm_noise1);

        % Definition for Fractional Brownian Motion for X and Y positions
        dfW2_x = zeros(Nts, 1); 
        dfW2_y = zeros(Nts, 1);
        fbm_noise2_x = fbm(Nts2, 0.5); % fBm with constant H = 0.5
        fbm_noise2_y = fbm(Nts2, 0.5);
        dfW2_x(2:end) = diff(fbm_noise2_x);
        dfW2_y(2:end) = diff(fbm_noise2_y);

        % Initial conditions for each cell
        x_array(1, cell_idx) = (floor(cell_idx/8.1)+1)*10;  % Random initial X position within [0, L]
        y_array(1, cell_idx) = 10 * cell_idx - 80* floor(cell_idx/8.1);  % Random initial Y position within [0, L]
        theta_array(1, cell_idx) = rand * 2 * pi;  % Random initial orientation
    end

    %% SIMULATION FOR ALL CELLS
    for t = 2:Nts
        for cell_idx = 1:num_cells
            p_i_f_x = cos(theta_array(1, cell_idx));  % direction vector in x
            p_i_f_y = sin(theta_array(1, cell_idx));  % direction vector in y
            num_contacts = 0;

            % CIL
            for other_cell = 1:num_cells
                if other_cell ~= cell_idx
                    % Distance between cells (considering periodic boundary)
                    dx = x_array(t-1, cell_idx) - x_array(t-1, other_cell);
                    dy = y_array(t-1, cell_idx) - y_array(t-1, other_cell);

                    % Apply periodic boundary conditions to distances
                    dx = dx - L * round(dx / L);
                    dy = dy - L * round(dy / L);

                    d_ij = sqrt(dx^2 + dy^2);

                    if d_ij <= interaction_threshold
                        % a_ij = dx + 2R
                        a_ij = dx + 2 * R;

                        % c_ij = midpoint (considering periodic boundaries)
                        c_ij_x = x_array(t-1, cell_idx) + dx / 2;
                        c_ij_y = y_array(t-1, cell_idx) + dy / 2;

                        % Eq. Repolarization direction: p_i_f = -Σ[(2*a_ij^2/d_ij^3)*(c_ij-x_ij)] 
                        p_i_f_x = p_i_f_x - (2 * a_ij^2 / d_ij^3) * (c_ij_x - x_array(t-1, cell_idx));
                        p_i_f_y = p_i_f_y - (2 * a_ij^2 / d_ij^3) * (c_ij_y - y_array(t-1, cell_idx));
                        num_contacts = num_contacts + 1;
                    end
                end
            end

            % Eq. Repolarization CIL or SDE
            if num_contacts > 0
                % direction vector: p_i_f = (cosθ*_i , sinθ*_i)^T  ->  Free angle: θ*_i = tan^-1(p_i_f_y, p_i_f_x)
                theta_star_i = atan2(p_i_f_y, p_i_f_x);

                %  Eq. Repolarization CIL (with contact)
                fCIL = 0.1;
                dtheta = -fCIL * (theta_array(t-1, cell_idx) - theta_star_i) + dfW1(t) * sqrt(2*Dr);
            else
                % Eq. Repolarization SDE (without contact)
                dtheta = sqrt(2*Dr) * dfW1(t);
            end
            
            % Update angle
            theta_array(t, cell_idx) = theta_array(t-1, cell_idx) + dtheta * dt;

            % Eq. of Motion for X and Y
            dx_dt = (Fm * cos(theta_array(t, cell_idx)) - alpha * dfW2_x(t)) / gamma_s;
            dy_dt = (Fm * sin(theta_array(t, cell_idx)) - alpha * dfW2_y(t)) / gamma_s;
            
            %% Interaction with other cells
            for other_cell = 1:num_cells
                if other_cell ~= cell_idx
                    % Distance between cells
                    d_ij = sqrt((x_array(t-1, cell_idx) - x_array(t-1, other_cell))^2 + ...
                                (y_array(t-1, cell_idx) - y_array(t-1, other_cell))^2);
                    
                    if d_ij <= interaction_threshold
                        % Eq. of Intercellular Force
                        F_int = (2/R) * (Ws - ((Ws + Wc)/R) * (d_ij - R));

                        % Avoid division by zero
                        if d_ij ~= 0
                            % Components of the force
                            dx_dt = dx_dt + F_int * (x_array(t-1, cell_idx) - x_array(t-1, other_cell)) / d_ij;
                            dy_dt = dy_dt + F_int * (y_array(t-1, cell_idx) - y_array(t-1, other_cell)) / d_ij;
                        else
                            % If cells are at the exact same position, apply a random small force to separate them
                            theta_rand = rand * 2 * pi;
                            dx_dt = dx_dt + F_int * cos(theta_rand);
                            dy_dt = dy_dt + F_int * sin(theta_rand);
                        end
                    end
                end
            end
            
            % Update position
            new_x = x_array(t-1, cell_idx) + dx_dt * dt;
            new_y = y_array(t-1, cell_idx) + dy_dt * dt;
            
            %% Periodic Boundary Conditions
            new_x = mod(new_x, L);
            new_y = mod(new_y, L);
            
            % Update position
            x_array(t, cell_idx) = new_x;
            y_array(t, cell_idx) = new_y;
        end
    end

    % ------------------------  OUTPUTS  ------------------------ %

    % Path for 'Output_Trajectories'
    current_folder = fileparts(mfilename('fullpath'));
    output_folder = fullfile(current_folder, 'Output_Trajectories_Periodic_Boundary');

    % Create folder if it doesn't exist
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    %% Plot final trajectories for all cells with distinct colors
    figure;
    hold on;

    % Crear un arreglo para manejar objetos de líneas (para la leyenda)
    line_handles = gobjects(num_cells, 1);

    for cell_idx = 1:num_cells
        for t = 2:Nts
            % Detect crossing boundaries and adjust visualization
            x_prev = x_array(t-1, cell_idx);
            y_prev = y_array(t-1, cell_idx);
            x_curr = x_array(t, cell_idx);
            y_curr = y_array(t, cell_idx);

            dx = x_curr - x_prev;
            dy = y_curr - y_prev;

            % Adjust for periodic boundary crossing in X
            if abs(dx) > L/2
                if dx > 0
                    x_curr = x_curr - L;
                else
                    x_curr = x_curr + L;
                end
            end

            % Adjust for periodic boundary crossing in Y
            if abs(dy) > L/2
                if dy > 0
                    y_curr = y_curr - L;
                else
                    y_curr = y_curr + L;
                end
            end

            % Plot the adjusted segment
            if t == 2
                % Guardar el primer objeto de línea para la leyenda
                line_handles(cell_idx) = plot([x_prev, x_curr], [y_prev, y_curr], ...
                                            'Color', colors(cell_idx, :), 'LineWidth', 2);
            else
                % Agregar segmentos sin modificar la leyenda
                plot([x_prev, x_curr], [y_prev, y_curr], 'Color', colors(cell_idx, :), 'LineWidth', 2);
            end
        end
    end

    % Draw the boundary for visualization
    rectangle('Position', [0, 0, L, L], 'EdgeColor', 'k', 'LineWidth', 2);
    axis([-0.1*L 1.1*L -0.1*L 1.1*L]);
    xlabel('X position');
    ylabel('Y position');
    title(['Cell Movement (Final Trajectories) for Dr = ', num2str(Dr)]);

    % Actualizar la leyenda con los objetos de línea correctos
    legend(line_handles, arrayfun(@(x) ['Cell ', num2str(x)], 1:num_cells, 'UniformOutput', false), 'Location', 'BestOutside');
    grid on;
    hold off;
    
    % Name the file according to the Dr value
    file_name = fullfile(output_folder, sprintf('Trajectory_Simulation_Boundary_Dr=%d.png', Dr));
    
    % Save the graph as PNG, overwrite if necessary
    saveas(gcf, file_name);
    
    % Show the ejecution time for each simulation
    end_time = toc(start_time_model);
    fprintf('Completed Model for Dr=%.2f in: %.2f seconds\n', Dr_values(dr_val), end_time);
end

% Show the total ejecution time
end_time_full = toc(start_time_full);
fprintf('Completed ALL simulations in: %.2f seconds\n', end_time_full);
