close all;clear all;clc;
start_time_full = tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Euler-Maruyama method on Smeets Model                                   %         
%                         SINGLE CELL MODEL                                               %
%                                                                                         %
%       1) Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + dfW2(t)                %
%       2) Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)                           %
%                                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SET UP
Dr_values = [0.1,0.5,1,5,100];
for dr_val=1:length(Dr_values)
    start_time_model = tic;
    % Model Parameters
    Fm = 1; gamma_s = 1; Dr = Dr_values(dr_val); alpha = 2.5;
    % Simulation Parameters
    dt = 0.1; T = 100; Nts =T/dt; Nts2 = linspace(0,T,Nts); 
    % Set up number of simulations per batch (i)
    for i = 1:200
        start_time_sim = tic;
        % Definition for Fractional Brownian Motion 1 -> White Noise in Angle
        dfW1 = zeros(Nts,1); 
        fbm_noise1 = fbm(Nts2,0.5);
        dfW1(2:end) = diff(fbm_noise1);        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% ONLY IF WE TEST A MODEL WITH CORRELATED NOISE ONLY ON X-AXIS; 
        % Definition for Fractional Brownian Motion 2 -> White Noise in Y-Axis
        dfW2_y = zeros(Nts,1); 
        fbm_noise2_y = fbm(Nts2,0.5);
        dfW2_y(2:end) = diff(fbm_noise2_y);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        % Initial Conditions (Position and Orientation)
        x_array = zeros(Nts,1); y_array = zeros(Nts,1); theta_array = zeros(Nts,1);
        theta_array(1) = rand * 2 * pi;
        % NEW TEST FOR SAVING VALUES FOR QUANTIFICATION OF ALPHA
        fmpi_x_array = zeros(Nts,1); fmpi_y_array = zeros(Nts,1);
        xi_x_array = zeros(Nts,1); xi_y_array = zeros(Nts,1);
        %% SIMULATION
        for t=2:Nts
            %% DEFINITIONS FOR CORRELATION AS FUNCTION OF SPACE (grid of -100 to 100)
            %1) LinearPos: Hx from 1 to 0.5 to 1 (+0+)
            H_x = abs(x_array(t-1)) * 0.005 + 0.5;
            H_max = 1;
            % 1.1) LinearAntiPos: Hx from 0.5 to 1 to 0.5 (0+0)
            % H_x = -abs(x_array(t-1))*0.005 + 1;
            % H_max = 0.5;
            % 2) LinearNeg: Hx from 0 to 0.5 to 0 (-0-)
            % H_x = -abs(x_array(t-1)) * 0.005 + 0.5;
            % H_max = 0;
            % 2.1) LinearAntiNeg: Hx from 0.5 to 0 to 0.5 (--)
            % H_x = abs(x_array(t-1))*0.005;
            % H_max = 0.5;
            %% MODELS 3, 4, and 5 ONLY FOR FUTURE WORK            
            % 3) Gradient in R(x,y) from 0.5 to 1;
            % R = sqrt(x_array(t-1)^2 + y_array(t-1)^2);
            % H_x = R*0.005 + 0.5;
            % H_y = R*0.005 + 0.5;
            % H_max = 1;
            % 4) Gradient in R(x,y) from 0 to 0.5
            % R = sqrt(x_array(t-1)^2 + y_array(t-1)^2);
            % H_x = R*0.005 + 0.5;
            % H_y = R*0.005 + 0.5;
            % H_max = 0.5;
            % 5) Linear Gradient in both axes (0.5 to 1, can change)
            % H_x = abs(x_array(t-1)) * 0.005 + 0.5;
            % H_y = abs(y_array(t-1)) * 0.005 + 0.5;
            % H_max = 1;
            %% LIMIT CASE FOR H, WHEN X >= 100
            if x_array(t) >= 100
                H_x = H_max;
            end            
            %% fBm DEFINITION AT EACH TIMESTEP FOR VARIABLE H
            Ntsx = linspace(0,t*dt,t);
            fbm_noise2_x = fbm(Ntsx,H_x);
            dfW2_x = zeros(t,1);
            dfW2_x(2:end) = diff(fbm_noise2_x);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% ONLY IF MODEL INCLUDES Y COMPONENT FOR CORRELATED NOISE
            % Ntsy = linspace(0,t*dt,t);
            % if y_array(t) >= 100
            %     H_y = H_max;
            % end
            % fbm_noise2_y = fbm(Ntsy,H_y);
            % dfW2_y = zeros(t,1);
            % dfW2_y(2:end) = diff(fbm_noise2_y);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            %% MODEL EQUATIONS
            % Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)
            dtheta = sqrt(2*Dr)*dfW1(t);
            theta_array(t) = theta_array(t-1)+dtheta*dt;
            % Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + dfW2(t)
            dx_dt = (Fm*cos(theta_array(t)) - alpha*dfW2_x(t))/(gamma_s);
            dy_dt = (Fm*sin(theta_array(t)) - alpha*dfW2_y(t))/(gamma_s);
            % Update of Position
            x_array(t) = x_array(t-1) + dx_dt*dt;
            y_array(t) = y_array(t-1) + dy_dt*dt;
            fmpi_x_array(t) = Fm*cos(theta_array(t));
            fmpi_y_array(t) = Fm*sin(theta_array(t));
            xi_x_array(t) = alpha*dfW2_x(t);
            xi_y_array(t) = alpha*dfW2_y(t);
        end        
        %%% For saving the data
        Dr_str = strrep(num2str(Dr), '.', '_');
        Alpha_str = strrep(num2str(alpha), '.', '_');
        folder_name = ['data__Alpha_',Alpha_str,'__Dr_', Dr_str, '__LinearPosX'];
        filename = ['Sim_', num2str(i), '__Dr_', Dr_str, '__LinearPosX.csv'];
        path_save = fullfile(folder_name, filename);
        % Create the folder if it doesn't exist
        if ~isfolder(folder_name)
            mkdir(folder_name);
        end
        % Create a table with the data and variable names
        data = table(x_array, y_array,theta_array,fmpi_x_array,fmpi_y_array,xi_x_array,xi_y_array, 'VariableNames', {'X', 'Y', 'Theta', 'FmPi_x','FmPi_y','Xi_x','Xi_y'});
        % Save the table to a CSV file
        writetable(data, path_save);
        end_time = toc(start_time_sim);
        fprintf('Completed Sim in: %.2f seconds\n', end_time);
    end 
    end_time = toc(start_time_model);
    fprintf('Completed Model in: %.2f seconds\n', end_time);
end
end_time = toc(start_time_full);
fprintf('Completed ALL in: %.2f seconds\n', end_time);
