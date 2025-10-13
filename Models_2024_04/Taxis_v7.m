close all; clear all;
clc; % Clear the command window
start_time = tic; % Start the timer for simulation duration

%% SET UP
% Model parameters
Dr_values = [0,0.1,1,10]; % Diffusion coefficients
H_values = [0.5]; % Hurst exponent values
Fm = 1; gamma_s = 1; alpha = 0; dt = 0.1; T = 100; Nts = T / dt; % Number of time steps
ftax = [1]; % External force frequency

Nts2 = linspace(0, T, Nts); % Time vector for the simulation
% Position of the ORGANIZER CENTER (particle_x, particle_y)
particle_x = -100/sqrt(2); % x position of the organizer
particle_y = -100/sqrt(2); % y position of the organizer

%% PARAMETER VALUES
% Number of simulations per parameter combination
num_simulations = 100; % Editable value for how many simulations per parameter set

% Iterate over all combinations of Dr_values, H_values, and ftax
for Dr = Dr_values
    for H = H_values
        for f = ftax 
            % Run the specified number of simulations with the current parameter set
            for sim_num = 1:num_simulations
                % Initialize variables for each simulation
                theta_array = zeros(Nts, 1); % Angle array
                x_array = zeros(Nts, 1); % X position array
                y_array = zeros(Nts, 1); % Y position array
                theta_org = zeros(Nts, 1); % Angle of the organizer (target)
                fmpi_x_array = zeros(Nts, 1); % Angle component in x direction
                fmpi_y_array = zeros(Nts, 1); % Angle component in y direction
                xi_x_array = zeros(Nts, 1); % Stochastic noise in x direction
                xi_y_array = zeros(Nts, 1); % Stochastic noise in y direction
                theta_array(1) = rand * 2 * pi; % Random initial angle
                
                % Definition for Fractional Brownian Motion 1 -> White Noise Angle
                dfW1 = zeros(Nts,1); 
                fbm_noise1 = fbm(Nts2,0.5);
                dfW1(2:end) = diff(fbm_noise1);
                % Definition for Fractional Brownian Motion 2 -> Correlated Noise
                dfW2_x = zeros(Nts,1); 
                fbm_noise2_x = fbm(Nts2,H);
                dfW2_x(2:end) = diff(fbm_noise2_x);            
                dfW2_y = zeros(Nts,1); 
                fbm_noise2_y = fbm(Nts2,H);
                dfW2_y(2:end) = diff(fbm_noise2_y);

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
                    dtheta = -f * (theta_array(t-1) - theta_org(t)) * sign_angle + sqrt(2*Dr) * dfW1(t); % Angle change
                    % Update the angle
                    theta_array(t) = theta_array(t - 1) + dtheta * dt;
                    
                    force_term_dx = Fm*cos(theta_array(t))/gamma_s;
                    stoch_term_dx = alpha*dfW2_x(t)/gamma_s;
                    force_term_dy = Fm*sin(theta_array(t))/gamma_s;
                    stoch_term_dy = alpha*dfW2_y(t)/gamma_s;
                    % Update of Position|||||||||||||||||||||||
                    x_array(t) = x_array(t-1) + force_term_dx*dt + stoch_term_dx;
                    y_array(t) = y_array(t-1) + force_term_dy*dt + stoch_term_dy;
                    fmpi_x_array(t) = force_term_dx*dt;
                    fmpi_y_array(t) = force_term_dy*dt;
                    xi_x_array(t) = stoch_term_dx;
                    xi_y_array(t) = stoch_term_dy;
                end
                
                % Store the data for the current simulation
                f_tax_str = strrep(num2str(f), '.', '_'); H_str = strrep(num2str(H), '.', '_'); Dr_str = strrep(num2str(Dr), '.', '_');
                simulation_data = table(theta_array, theta_org, x_array, y_array, xi_x_array, xi_y_array,fmpi_x_array,fmpi_y_array);
                main_path = ['DATA_202555/alpha_0_f_tax_',f_tax_str,'/H_',H_str,'/'...
                    'Dr_',Dr_str];
                filename = ['Sim_', num2str(sim_num), '_Dr_', Dr_str, '_H_', H_str, '_f_', f_tax_str,'.csv'];
                path_save = fullfile(main_path,filename);
                if ~isfolder(main_path)
                    mkdir(main_path);
                end
                % Save all simulations for the current parameter combination in one sheet
                writetable(simulation_data, path_save);
            end
            
        end
    end
end

end_time = toc(start_time); % End the timer
fprintf('Total simulation time: %.2f seconds\n', end_time); % Display the total time