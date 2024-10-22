close all; clear all; clc;
start_time_full = tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Euler-Maruyama method on Smeets Model for multiple cells             %
%                        MULTI-CELL MODEL WITH BOUNDARY                          %
%                                                                                %
%   1) Eq. of Motion: Fm*{cos(theta),sin(theta)} = gamma_s*v_i + dfW2(t)         %
%   2) Eq. Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)                  %
%   3) Eq. of Intercellular Force: 2/R * [Ws - (Ws + Wc)/R * (dij - R)]          %
%   4) Reflective Boundary Conditions                                            %
%                                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SET UP
Dr_values = [0.1, 0.5, 1, 5, 100];
num_cells = 5;  % Number of cells
colors = lines(num_cells);  % Colors for each cell
Wc = 1; % Cell-cell adhesion energy 
Ws = 1; % Cell-substrate adhesion energy 
R = 1;  % Cell radius
interaction_threshold = 2 * R;  % Set to 2R as per the model

% Simulation domain boundaries
L = 10;  % Simulation square domain [0, L] x [0, L]

for dr_val = 1:length(Dr_values)
    start_time_model = tic;
    
    % Model Parameters
    Fm = 1; 
    gamma_s = 1; 
    Dr = Dr_values(dr_val); 
    alpha = 2.5;
    
    % Simulation Parameters
    dt = 0.1; 
    T = 100; 
    Nts = floor(T/dt); 
    Nts2 = linspace(0, T, Nts); 
    
    % Pre-allocate matrices for all cells
    x_array = zeros(Nts, num_cells); 
    y_array = zeros(Nts, num_cells); 
    theta_array = zeros(Nts, num_cells);
    
    % Pre-allocate velocity arrays (optional, useful for boundary reflections)
    vx_array = zeros(Nts, num_cells);
    vy_array = zeros(Nts, num_cells);
    
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
        x_array(1, cell_idx) = rand * L;  % Random initial X position within [0, L]
        y_array(1, cell_idx) = rand * L;  % Random initial Y position within [0, L]
        theta_array(1, cell_idx) = rand * 2 * pi;  % Random initial orientation
    end

    %% SIMULATION FOR ALL CELLS
    for t = 2:Nts
        for cell_idx = 1:num_cells
            % Eq. Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)
            dtheta = sqrt(2*Dr) * dfW1(t);
            theta_array(t, cell_idx) = theta_array(t-1, cell_idx) + dtheta * dt;

            % Eq. of Motion for X and Y without interactions
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
            
            %% Reflective Boundary Conditions
            % Check and enforce boundaries for X
            if new_x < 0
                new_x = -new_x;  % Reflect position
                dx_dt = -dx_dt;  % Reverse velocity component
            elseif new_x > L
                new_x = 2*L - new_x;
                dx_dt = -dx_dt;
            end

            % Check and enforce boundaries for Y
            if new_y < 0
                new_y = -new_y;
                dy_dt = -dy_dt;
            elseif new_y > L
                new_y = 2*L - new_y;
                dy_dt = -dy_dt;
            end
            
            % Optionally, store velocities if needed for analysis
            vx_array(t, cell_idx) = dx_dt;
            vy_array(t, cell_idx) = dy_dt;
            
            % Update position
            x_array(t, cell_idx) = new_x;
            y_array(t, cell_idx) = new_y;
        end
    end

    % ------------------------  OUTPUTS  ------------------------ %

    % Path for 'Output_Trajectories'
    current_folder = fileparts(mfilename('fullpath'));
    output_folder = fullfile(current_folder, 'Output_Trajectories_Boundary');

    % Create folder if it doesn't exist
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    %% Plot final trajectories for all cells with distinct colors
    figure;
    hold on;
    for cell_idx = 1:num_cells
        plot(x_array(:, cell_idx), y_array(:, cell_idx), 'Color', colors(cell_idx,:), 'LineWidth', 2);
    end
    % Draw the boundary for visualization
    rectangle('Position', [0, 0, L, L], 'EdgeColor', 'k', 'LineWidth', 2);
    xlabel('X position');
    ylabel('Y position');
    title(['Cell Movement (Final Trajectories) for Dr = ', num2str(Dr)]);
    legend(arrayfun(@(x) ['Cell ', num2str(x)], 1:num_cells, 'UniformOutput', false));
    grid on;
    hold off;
    
    % Name the file according to the Dr value
    file_name = fullfile(output_folder, sprintf('Trajectory_Simulation_Boundary_Dr=%d.png', Dr));
    
    % Save the graph as PNG, overwrite if necessary
    saveas(gcf, file_name);

    end_time = toc(start_time_model);
    fprintf('Completed Model for Dr=%.2f in: %.2f seconds\n', Dr_values(dr_val), end_time);
end

% Mostrar el tiempo total de ejecución para todas las simulaciones
end_time_full = toc(start_time_full);
fprintf('Completed ALL simulations in: %.2f seconds\n', end_time_full);
