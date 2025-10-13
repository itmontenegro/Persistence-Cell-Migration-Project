close all;clear all;clc;
start_time = tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Euler-Maruyama method on Smeets Model                                   %         
%                         SINGLE CELL MODEL                                               %
%                                                                                         %
%       1) Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + alpha*dfW2(t)/dt       %
%       2) Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)/dt                        %
%                                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SET UP
Dr_values = [0,0.1,1,10];
H_values = [0.5];
start_time_H = tic;
for Hval = 1:length(H_values)
    start_time_model = tic;
    for Drval = 1:length(Dr_values)
        % Model Parameters
        Fm = 1; gamma_s = 1; Dr = Dr_values(Drval); alpha = 0;
        % Simulation Parameters
        dt = 0.1; T = 100; Nts = T/dt; 
        Nts2 = linspace(0,T,Nts);     
        % 200 Sims per model
        for i = 1:25
            start_time_sim = tic;
            % Definition for Fractional Brownian Motion 1 -> White Noise Angle
            dfW1 = zeros(Nts,1); 
            fbm_noise1 = fbm(Nts2,0.5);
            dfW1(2:end) = diff(fbm_noise1);
            % Definition for Fractional Brownian Motion 2 -> Correlated Noise
            H = H_values(Hval);
            dfW2_x = zeros(Nts,1); 
            fbm_noise2_x = fbm(Nts2,H);
            dfW2_x(2:end) = diff(fbm_noise2_x);            
            dfW2_y = zeros(Nts,1); 
            fbm_noise2_y = fbm(Nts2,H);
            dfW2_y(2:end) = diff(fbm_noise2_y);        
            % Initial Conditions
            x_array = zeros(Nts,1); y_array = zeros(Nts,1); theta_array = zeros(Nts,1);
            theta_array(1) = rand*2*pi;        
            % NEW TEST FOR SAVING VALUES FOR QUANTIFICATION OF ALPHA
            fmpi_x_array = zeros(Nts,1); fmpi_y_array = zeros(Nts,1);
            xi_x_array = zeros(Nts,1); xi_y_array = zeros(Nts,1);        
            for t=2:Nts
                % AS OF THEORY CORRECTIONS MAY 2024 --> Xi(t) = dfW(t)/dt
                % Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)/dt
                dtheta = sqrt(2*Dr)*dfW1(t);
                theta_array(t) = theta_array(t-1)+dtheta;
                % Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i -
                % alpha*dfW2(t)/dt
                force_term_dx = Fm*cos(theta_array(t))/gamma_s;
                stoch_term_dx = alpha*dfW2_x(t)/gamma_s;
                force_term_dy = Fm*sin(theta_array(t))/gamma_s;
                stoch_term_dy = alpha*dfW2_y(t)/gamma_s;
                % Update of Position
                x_array(t) = x_array(t-1) + force_term_dx*dt + stoch_term_dx;
                y_array(t) = y_array(t-1) + force_term_dy*dt + stoch_term_dy;
                fmpi_x_array(t) = force_term_dx*dt;
                fmpi_y_array(t) = force_term_dy*dt;
                xi_x_array(t) = stoch_term_dx;
                xi_y_array(t) = stoch_term_dy;
            end            
            %%% For saving the data
            Dr_str = strrep(num2str(Dr), '.', '_');
            H_str = strrep(num2str(H), '.', '_');
            Alpha_str = strrep(num2str(alpha), '.', '_');
            alpha_path = ['../DATA_2025/Batch25_InConTest/Alpha_',Alpha_str,'/'];
            main_path = ['../DATA_2025/Batch25_InConTest/Alpha_',Alpha_str,'/H_',H_str,'/'];
            folder_name = ['data__Alpha_',Alpha_str,'__Dr_', Dr_str, '__H_', H_str];
            filename = ['Sim_', num2str(i), '__Dr_', Dr_str, '__H_', H_str, '.csv'];
            path_save = fullfile(main_path,folder_name, filename);
            % Create the folders if they doesn't exist
            % FOR ALPHA
            if ~isfolder(alpha_path)
                mkdir(alpha_path);
            end
            % FOR H
            if ~isfolder(main_path)
                mkdir(main_path);
            end
            % FOR Dr
            if ~isfolder(fullfile(main_path,folder_name))
                mkdir(fullfile(main_path,folder_name));
            end
            % Create a table with the data and variable names
            data = table(x_array, y_array,theta_array,fmpi_x_array,fmpi_y_array,xi_x_array,xi_y_array, 'VariableNames', {'X', 'Y', 'Theta', 'FmPi_x','FmPi_y','Xi_x','Xi_y'});
            % Save the table to a CSV file
            writetable(data, path_save);
            end_time = toc(start_time_sim);
            fprintf('Completed Sim in: %.2f seconds\n', end_time);
        end        
    end
    end_time = toc(start_time_model);
    fprintf('Completed ALL Drs in: %.2f seconds\n', end_time);    
end
end_time = toc(start_time_H);
fprintf('Completed ALL Hs in: %.2f seconds\n', end_time);
end_time = toc(start_time);
    fprintf('Completed ALL Alphas in: %.2f seconds\n', end_time);
