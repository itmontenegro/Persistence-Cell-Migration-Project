close all; % Close all figures
clear all; % Clear all variables
clc; % Clear the command window
start_time = tic; % Start the timer for simulation duration

%% SET UP
% Model parameters
Dr_values = [0, 0.1, 1, 10]; % Diffusion coefficients
H_values = [0.5, 0.75, 0.99]; % Hurst exponent values
Fm = 1; % Force applied to the particle
gamma_s = 1; % Some constant for the system
alpha = 0; % Friction coefficient
dt = 0.1; % Time step
T = 100; % Total time for simulation
Nts = T / dt; % Number of time steps
ftax = [0.01, 0.1]; % External force frequency
Nts2 = linspace(0, T, Nts); % Time vector for the simulation

% Position of the ORGANIZER CENTER (particle_x, particle_y)
particle_x = -50; % x position of the organizer
particle_y = -50; % y position of the organizer

% Initialize an array to store the data
data = [];

%% PARAMETER VALUES
% Number of simulations per parameter combination
num_simulations = 3; % Editable value for how many simulations per parameter set

% File name for the output Excel file
output_file = 'simulation_134.xlsx'; 

% Iterate over all combinations of Dr_values, H_values, and ftax
for Dr = Dr_values
    for H = H_values
        for f = ftax
            % Define the sheet name for Excel output based only on the parameters
            sheet_name = sprintf('Dr_%.2f_H_%.2f_f_%.2f', Dr, H, f);
            
            % Asegúrate de que el nombre de la hoja no contenga caracteres no permitidos
            sheet_name = sheet_name(1:min(end, 31)); % Limitar la longitud a 31 caracteres
            sheet_name = strrep(sheet_name, '\', '_');
            sheet_name = strrep(sheet_name, '/', '_');
            sheet_name = strrep(sheet_name, '*', '_');
            sheet_name = strrep(sheet_name, '?', '_');
            sheet_name = strrep(sheet_name, ':', '_');
            sheet_name = strrep(sheet_name, '[', '_');
            sheet_name = strrep(sheet_name, ']', '_');
            
            % Initialize a table to store all simulations for this parameter set
            all_simulation_data = table();
            
            % Run the specified number of simulations with the current parameter set
            for sim_num = 1:num_simulations
                % Initialize variables for each simulation
                theta_array = zeros(Nts, 1); % Angle array
                x_array = zeros(Nts, 1); % X position array
                y_array = zeros(Nts, 1); % Y position array
                theta_org = zeros(Nts, 1); % Angle of the organizer (target)
                xi_x_array = zeros(Nts, 1); % Stochastic noise in x direction
                xi_y_array = zeros(Nts, 1); % Stochastic noise in y direction
                
                theta_array(1) = rand * 2 * pi; % Random initial angle
                
                % Simulate cellular movement
                for t = 2:Nts
                    % Calculate the position differences (delta_x and delta_y)
                    delta_x = particle_x - x_array(t - 1); % Difference in x
                    delta_y = particle_y - y_array(t - 1); % Difference in y
                    normalized_dx = delta_x / (delta_x^2 + delta_y^2); % Normalize x difference
                    normalized_dy = delta_y / (delta_x^2 + delta_y^2); % Normalize y difference
                    
                    % Calculate the angle of the organizer (theta_org) with alignment
                    theta_org_alignment = normalized_dx * cos(theta_array(t-1)) + normalized_dy * sin(theta_array(t-1));
                    theta_org(t) = acos(theta_org_alignment); % Angle calculation
                    
                    % Calculate the variation in the angle (dtheta)
                    cross_angle = normalized_dx * sin(theta_array(t-1)) - normalized_dy * cos(theta_array(t-1));
                    sign_angle = sign(cross_angle); % Keep the sign of the angle
                    dtheta = -f * (theta_array(t-1) - theta_org(t)) * sign_angle + sqrt(2*Dr) * randn; % Angle change
                    
                    % Update the angle
                    theta_array(t) = theta_array(t - 1) + dtheta * dt;
                    
                    % Update the position with stochastic noise
                    xi_x_array(t) = randn;  % Stochastic noise in x direction
                    xi_y_array(t) = randn;  % Stochastic noise in y direction
                    
                    % Calculate the movement in x and y directions
                    dx_dt = (Fm * cos(theta_array(t)) - alpha * xi_x_array(t)) / gamma_s;
                    dy_dt = (Fm * sin(theta_array(t)) - alpha * xi_y_array(t)) / gamma_s;
                    
                    % Update positions (x and y)
                    x_array(t) = x_array(t - 1) + dx_dt * dt;
                    y_array(t) = y_array(t - 1) + dy_dt * dt;
                end
                
                % Store the data for the current simulation
                simulation_data = table(theta_array, theta_org, x_array, y_array, xi_x_array, xi_y_array);
                
                % Append the data to the all_simulation_data table
                all_simulation_data = [all_simulation_data; simulation_data];
            end
            
            % Save all simulations for the current parameter combination in one sheet
            writetable(all_simulation_data, output_file, 'Sheet', sheet_name);
        end
    end
end

end_time = toc(start_time); % End the timer
fprintf('Total simulation time: %.2f seconds\n', end_time); % Display the total time
