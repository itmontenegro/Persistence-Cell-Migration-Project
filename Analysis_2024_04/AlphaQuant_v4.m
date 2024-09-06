clear all; close all;
% SCRIPT TO CALCULATE NORM OF THE NOISES; FOR THIS WE COMPARE THE CONTROL
% CASE WHERE H=0.5 and Dr=1; AS TO DEFINE OUR ALPHA
%% LOAD PROCESSED CSVs
mainFolderPath = 'Model1_H_cnst/Model1_2D_Batch20_AlphaTest_v6/';% Get a list of subfolders in the main folder
subfolders = dir(mainFolderPath);
subfolders = subfolders([subfolders.isdir]);  % Filter out non-directories 
% NOISE MAGNITUDE MATRIX STORE
noise_ratio_matrix = zeros(1000, length(subfolders)-2);
xi_noise_matrix = zeros(1000, length(subfolders)-2);
fmpi_noise_matrix = zeros(1000, length(subfolders)-2);
% Array to store the subfolder names (header for mean_displacement)
subfolderNames = {};
% Iterate through each model (combination of H and Dr)
for i = 3:length(subfolders)  % Start from 3 to skip '.' and '..' entries
    subfolder = subfolders(i).name;
    subfolderNames = [subfolderNames,subfolder];
    subfolderPath = fullfile(mainFolderPath,subfolder)
    norm_fmpi_data = readtable([subfolderPath,'/H_0_5/norm_fmpi.csv']);
    fmpi_array = table2array(norm_fmpi_data);
    norm_xi_data = readtable([subfolderPath,'/H_0_5/norm_xi.csv']);
    xi_array = table2array(norm_xi_data, 'VariableNames', subfolder);
    % WITH THIS ONE WE GET THE RATIO OF NOISE MAGNITUDES
    % noise_ratio_array = xi_array./fmpi_array;
    % WITH THIS ONE WE GET THE ABSOLUTE DIFFERENCE IN NOISE MAGNITUDES
    noise_ratio_array = abs(xi_array - fmpi_array);

    noise_ratio_array(isnan(noise_ratio_array))=0;
    noise_ratio_matrix(:, i-2) = noise_ratio_array; 
    xi_noise_matrix(:,i-2) = xi_array;
    fmpi_noise_matrix(:,i-2) = fmpi_array;
end
noise_ratio_table = array2table(noise_ratio_matrix, 'VariableNames',subfolderNames);
noise_mean_ratio_matrix = mean(noise_ratio_matrix);
noise_sd_ratio_matrix = std(noise_ratio_matrix);
%% Surface Plot of Noise Magnitude Ratio
% Extract the time vector (assuming time is the row number)
time = (1:height(noise_ratio_table))';
% Extract the condition names
conditions = noise_ratio_table.Properties.VariableNames;
for alpha_index = 1:length(conditions)
    conditions(alpha_index) = strrep(conditions(alpha_index),'Alpha_0_','0.');
    conditions(alpha_index) = strrep(conditions(alpha_index),'Alpha_0','0');
    conditions(alpha_index) = strrep(conditions(alpha_index),'Alpha_1','1');
end
% Create a meshgrid for the surface plot
[TimeGrid, ConditionGrid] = meshgrid(time, 1:width(noise_ratio_table));
% Convert the table to an array for plotting
dataArray = table2array(noise_ratio_table);
% Create the surface plot
figure;
surf(TimeGrid, ConditionGrid, dataArray');
% Label the axes
xlabel('Time');
ylabel('Alpha Value');
zlabel('Ratio');
title('Noise Magnitude Ratio: (F_mp/\gamma \cdot \Delta t)/(\alpha dW^H)');
% Customize the y-axis to show condition names
set(gca, 'YTick', 1:width(noise_ratio_table), 'YTickLabel', conditions);
% Add a color bar to show the scale of values
colorbar;
axis([2 1000 0 21 0 0.3])
% Adjust view angle for better visualization
view(135, 30);
grid on;
%% 2D Plot of Mean Noise Magnitude Ratio
figure;
plot(1:21, noise_mean_ratio_matrix, 'b-', 'LineWidth',2);
hold on
scatter(1:21, noise_mean_ratio_matrix, 30, 'b','filled');
hold off
% Label the axes
xlabel('Alpha Value');
ylabel('Magnitude Difference');
% title('Mean Noise Magnitude Ratio: (F_mp/\gamma \cdot \Delta t)/(\alpha dW^H)');
title('Mean Noise Magnitude Difference: (F_mp/\gamma \cdot \Delta t)-(\alpha dW^H)');
% Customize the y-axis to show condition names
set(gca, 'XTick', 1:21, 'XTickLabel', conditions);
axis([1 21 0 0.3]);% Add a horizontal red line at y = 0.1
yline(1, 'r-', 'LineWidth', 2);
% Optionally, add a label to the red line
yline(1, 'r-', 'LineWidth', 2, 'Label', '|F_mp/\gamma \cdot \Delta t| = 0.1', 'LabelHorizontalAlignment', 'right');
