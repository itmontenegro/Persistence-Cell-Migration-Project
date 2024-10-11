close all; clear all; clc;
start_time_full = tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Euler-Maruyama method on Smeets Model for multiple cells             %
%                              MULTI-CELL MODEL                                  %
%                                                                                %
%   1) Eq. of Motion: Fm*{cos(theta),sin(theta)} = gamma_s*v_i + dfW2(t)         %
%   2) Eq. Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)                  %
%   3) Eq. of Intercellular Force: 2/R * [Ws - (Ws + Wc)/R * (dij - R)]          %
%                                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SET UP
Dr_values = [0.1, 0.5, 1, 5, 100];
num_cells = 5;  % Number of cells
colors = lines(num_cells);  % Colors for each cell
Wc = 1; % cell-cell adhesion energy 
Ws = 1; % cell-substract adhesion energy 
R = 1; % cell radius
interaction_threshold = 2*R;  % Max distance for interactions to take place

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
    Nts = T/dt; 
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
        x_array(1, cell_idx) = rand * 10;  % Random initial X position
        y_array(1, cell_idx) = rand * 10;  % Random initial Y position
        theta_array(1, cell_idx) = rand * 2 * pi;  % Random initial orientation

        %% SIMULATION FOR ALL CELL
        for t = 2:Nts
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
            x_array(t, cell_idx) = x_array(t-1, cell_idx) + dx_dt * dt;
            y_array(t, cell_idx) = y_array(t-1, cell_idx) + dy_dt * dt;
        end
    end

    % ------------------------  OUTPUTS  ------------------------ %

    % Path for 'Output_Trajectories'
    current_folder = fileparts(mfilename('fullpath'));
    output_folder = fullfile(current_folder, 'Output_Trajectories');

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
    xlabel('X position');
    ylabel('Y position');
    title(['Cell Movement (Final Trajectories) for Dr = ', num2str(Dr)]);
    legend(arrayfun(@(x) ['Cell ', num2str(x)], 1:num_cells, 'UniformOutput', false));
    grid on;
    hold off;
    
    % Name the file according to the Dr value
    file_name = fullfile(output_folder, sprintf('Trajectory_Simulation_Dr=%d.png', Dr));
    
    % Save the graph as PNG, overwrite if necessary
    saveas(gcf, file_name);

    % Show the ejecution time for each simulation
    end_time = toc(start_time_model);
    fprintf('Completed Model for Dr=%.2f in: %.2f seconds\n', Dr, end_time);

end

% Show the total ejecution time
end_time_full = toc(start_time_full);
fprintf('Completed ALL simulations in: %.2f seconds\n', end_time_full);
end_time_full = toc(start_time_full);
fprintf('Completed ALL simulations in: %.2f seconds\n', end_time_full);