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

%% SET UP (CHANGE DR HERE)
% Model Parameters
Fm = 1; gamma_s = 1; Dr = 100;
% Simulation Parameters
dt = 0.1; T = 100; Nts = T/dt;

% 50 Sims per model
for i = 1:50
    % Definition for Fractional Brownian Motion 1 -> White Noise
    dfW1 = zeros(Nts,1); 
    fbm_noise1 = fbm(Nts,0.5);
    dfW1(2:end) = diff(fbm_noise1);
    %% CHANGE H HERE
    % Definition for Fractional Brownian Motion 2 -> Correlated Noise
    H = 0.99;
    dfW2_x = zeros(Nts,1); 
    fbm_noise2_x = fbm(Nts,H);
    dfW2_x(2:end) = diff(fbm_noise2_x);

    % Initial Conditions
    x_array = zeros(Nts,1); y_array = zeros(Nts,1); theta_array = zeros(Nts,1);
    theta_array(1) = rand*2*pi;

    for t=2:Nts
        % Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)
        dtheta = sqrt(2*Dr)*dfW1(t);
        theta_array(t) = theta_array(t-1)+dtheta*dt;
        % Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + dfW2(t)
        dx_dt = (Fm*cos(theta_array(t)) - dfW2_x(t))/(gamma_s);
        % Update of Position
        x_array(t) = x_array(t-1) + dx_dt*dt;
    end
    
    %%% For saving the data
    Dr_str = strrep(num2str(Dr), '.', '_');
    H_str = strrep(num2str(H), '.', '_');
    folder_name = ['data__Dr_', Dr_str, '__H_', H_str];
    filename = ['Sim_', num2str(i), '__Dr_', Dr_str, '__H_', H_str, '.csv'];
    path_save = fullfile(folder_name, filename);
    % Create the folder if it doesn't exist
    if ~isfolder(folder_name)
        mkdir(folder_name);
    end
    % Create a table with the data and variable names
    data = table(x_array, theta_array, 'VariableNames', {'X', 'Theta'});
    % Save the table to a CSV file
    writetable(data, path_save);
end

%% RESULTS
% Create a colormap based on time
colormapTime = jet(Nts);  % You can choose a different colormap if you prefer
alphaValue = 0.5;  % Adjust as needed (transparency)

% Plotting Trajectory with color-coded x_array
figure;
hold on;

% Plot using a colormap with transparency
scatter(x_array, y_array, 100, 1:Nts, 'filled', 'MarkerFaceAlpha', alphaValue);

hold off;
xlabel('X');
ylabel('Y');
title(['Dr = ', num2str(Dr), ' H = ', num2str(H)])
subtitle(['Initial position of cell at X = ', num2str(x_array(1))])
grid on;
axis([-100, 100, -10, 10]);

% Adjust the aspect ratio to create a rectangular plot
aspectRatio = 8;
pbaspect([aspectRatio, 1, 1]);  % The first value controls the x-to-y ratio

colorbar;  % Add a colorbar to indicate time
colormap(colormapTime);  % Set the colormap

% Optionally, you can set the colorbar label to represent time
c = colorbar;
c.Label.String = 'Time Step';

% Optionally, save the plot
saveas(gcf, ['1DTraj__Dr_',Dr_str,'__H_',H_str,'.png']);
close(gcf); 


end_time = toc(start_time);
fprintf('Elapsed Time: %.2f seconds\n', end_time);
