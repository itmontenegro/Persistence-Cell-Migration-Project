close all;clear all;clc;
start_time = tic;
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
for Drval = 1:length(Dr_values)
    start_time_model = tic;
    % Model Parameters
    Fm = 1; gamma_s = 1; Dr = Dr_values(Drval); alpha = 2.5;
    for i = 11:100
        start_time_sim = tic;
        % Simulation Parameters
        dt = 0.1; T = 100; Nts = T/dt; Nts2 = linspace(0,T,Nts); 
        % Definition for Fractional Brownian Motion 1 -> White Noise in Angle
        dfW1 = zeros(Nts,1); 
        fbm_noise1 = fbm(Nts2,0.5);
        dfW1(2:end) = diff(fbm_noise1);
        % Initial Conditions
        x_array = zeros(Nts,1); y_array = zeros(Nts,1); theta_array = zeros(Nts,1);
        % Initial Position and direction of Cell
        x_array(1) = 0;
        theta_array(1) = rand * 2 * pi;
        fmpi_x_array = zeros(Nts,1);xi_x_array = zeros(Nts,1);
        %% SIMULATION
        for t=2:Nts
            % Definition for fBm Noise as Function of the Position
            % 1) Gradient in x from 0 to 1 (- +)
            %H = x_array(t) * 0.005 + 0.5;
            % 1.1) Gradient in x from 1 to 0.5 to 1 (+ +)
            H = abs(x_array(t)) * 0.005 + 0.5;
            % 2.2) Gradient in x from 0.5 to 1 to 0.5 (+ +)
            %H = -abs(x_array(t)) * 0.005 + 1;
            % 3.1) Gradient in x from 0 to 0.5 to 0 (- -)
            %H = -abs(x_array(t)) * 0.005 + 0.5;
            % 3.2) Gradient in x from 0.5 to 0 to 0.5 (- -)
            %H = abs(x_array(t)) * 0.005 + 0;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Nts2 = linspace(0,t*dt,t);
            fbm_noise2 = fbm(Nts2,H);
            dfW2 = diff(fbm_noise2);
            % Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)
            dtheta = sqrt(2*Dr)*dfW1(t);
            theta_array(t) = theta_array(t-1)+dtheta*dt;
            % Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + dfW2(t)
            dx_dt = (Fm*cos(theta_array(t)) - alpha*dfW2(t-1))/(gamma_s);
            % Update of Position
            x_array(t) = x_array(t-1) + dx_dt*dt;
            fmpi_x_array(t) = Fm*cos(theta_array(t));
            xi_x_array(t) = alpha*dfW2(t-1);
        end
        %%% For saving the data
        Dr_str = strrep(num2str(Dr), '.', '_');
        folder_name = ['data__Model2_1D__Dr_', Dr_str];
        filename = ['Sim_', num2str(i), '__Dr_', Dr_str,'.csv'];
        path_save = fullfile(folder_name, filename);
        % Create the folder if it doesn't exist
        if ~isfolder(folder_name)
            mkdir(folder_name);
        end
        % Create a table with the data and variable names
        data = table(x_array, theta_array,fmpi_x_array, xi_x_array, 'VariableNames', {'X', 'Theta', 'FmPi_x','Xi_x'});
        % Save the table to a CSV file
        writetable(data, path_save);
        end_time = toc(start_time_sim);
        fprintf('Completed Sim in: %.2f seconds\n', end_time);
    end
    end_time = toc(start_time_model);
    fprintf('Completed Model in: %.2f seconds\n', end_time);   
end
end_time = toc(start_time);
fprintf('Completed ALL in: %.2f seconds\n', end_time);

% %% RESULTS
% close all;
% data_array = readtable(['Model2_HasX/Model2_1D/Alpha_2_5/LinearAntiPos/' ...
%     'data__Model2_1D__Dr_0_5/Sim_3__Dr_0_5.csv']);
% x_array = data_array.X; y_array = zeros(1000,1); Nts = 1000;
% 
% % Create a colormap based on time
% colormapTime = jet(Nts);  % You can choose a different colormap if you prefer
% alphaValue = 0.5;  % Adjust as needed (transparency)
% 
% % Plotting Trajectory with color-coded x_array
% figure;
% hold on;
% 
% % Plot using a colormap with transparency
% scatter(x_array, y_array, 100, 1:Nts, 'filled', 'MarkerFaceAlpha', alphaValue);
% 
% hold off;
% xlabel('X');
% ylabel('Y');
% title('Dr = 0.5')
% subtitle(['Initial position of cell at X = ', num2str(x_array(1))])
% grid on;
% axis([-100, 100, -10, 10]);
% 
% % Adjust the aspect ratio to create a rectangular plot
% aspectRatio = 8;
% pbaspect([aspectRatio, 1, 1]);  % The first value controls the x-to-y ratio
% 
% colorbar;  % Add a colorbar to indicate time
% colormap(colormapTime);  % Set the colormap
% 
% % Optionally, you can set the colorbar label to represent time
% c = colorbar;
% c.Label.String = 'Time Step';
% % SAVING PNG FIGURE
% filename = ['LinearAntiPos__Dr_0_5_3.png'];
% saveas(gcf, filename);