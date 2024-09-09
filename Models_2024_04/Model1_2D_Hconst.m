close all;clear all;clc;
start_time = tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Euler-Maruyama method on Smeets Model                                   %         
%                         SINGLE CELL MODEL                                               %
%                                                                                         %
%       1) Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + alpha*dfW2(t)          %
%       2) Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)                           %
%                                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SET UP PARAMETERS
Dr_values = [0,0.1,0.5,1,5,10,100];
H_values = [0.5];
for Hval = 1:length(H_values)
    start_time_model = tic;
    for Drval = 1:length(Dr_values)
        % Model Parameters
        Fm = 1; gamma_s = 1; Dr = Dr_values(Drval); alpha = 0;
        % Simulation Parameters
        dt = 0.1; T = 100; Nts = T/dt; 
        Nts2 = linspace(0,T,Nts);     
        % 200 Sims per model
        for i = 1:200
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
                % Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)
                dtheta = sqrt(2*Dr)*dfW1(t);
                theta_array(t) = theta_array(t-1)+dtheta*dt;
                % Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + alpha*dfW2(t)
                dx_dt = (Fm*cos(theta_array(t)) - alpha*dfW2_x(t))/(gamma_s);
                dy_dt = (Fm*sin(theta_array(t)) - alpha*dfW2_y(t))/(gamma_s);
                % Update of Position
                x_array(t) = x_array(t-1) + dx_dt*dt;
                y_array(t) = y_array(t-1) + dy_dt*dt;
                fmpi_x_array(t) = Fm*cos(theta_array(t));
                fmpi_y_array(t) = Fm*sin(theta_array(t));
                xi_x_array(t) = alpha*dfW2_x(t);
                xi_y_array(t) = alpha*dfW2_y(t);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % IDEA FOR "LIVE" PLOTTING (DANIELA MEETING)
                % UNSURE IF IT WORKS, BECAUSE IT IS INSIDE A LOOP, BUT MAYBE TRY(?)
                figure;
                plot(x_array,y_array, 'b-')
                hold on;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end            
            %%% For saving the data
            Dr_str = strrep(num2str(Dr), '.', '_');
            H_str = strrep(num2str(H), '.', '_');
            Alpha_str = strrep(num2str(alpha), '.', '_');
            main_path = ['../DATA_2024/Model1_H_cnst/Model1_2D_Batch200/Alpha_',Alpha_str,'/H_',H_str,'/'];
            folder_name = ['data__Alpha_',Alpha_str,'__Dr_', Dr_str, '__H_', H_str];
            filename = ['Sim_', num2str(i), '__Dr_', Dr_str, '__H_', H_str, '.csv'];
            path_save = fullfile(main_path,folder_name, filename);
            % Create the folder if it doesn't exist
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
    fprintf('Completed Model in: %.2f seconds\n', end_time);    
end
end_time = toc(start_time);
fprintf('Completed ALL in: %.2f seconds\n', end_time);
