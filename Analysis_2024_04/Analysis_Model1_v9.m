clear all; close all;

%% LOAD PROCESSED CSVs
mainFolderPath = 'Model1_H_cnst/Model1_2D_Batch200/Alpha_2_5/';

disp_data_1 = readtable([mainFolderPath,'H_0_01/displacement_data.csv']);
dist_data_1 = readtable([mainFolderPath,'H_0_01/distance_data.csv']);
msd_data_1 = readtable([mainFolderPath,'H_0_01/MSD_data.csv']);
disp_sd_data_1 = readtable([mainFolderPath,'H_0_01/displacement_sd_data.csv']);
dist_sd_data_1 = readtable([mainFolderPath,'H_0_01/distance_sd_data.csv']);
msd_sd_data_1 = readtable([mainFolderPath,'H_0_01/MSD_sd_data.csv']);
pers_fac_1 = readtable([mainFolderPath, 'H_0_01/persistence_factor.csv']);
pers_fac_sd_data_1 = readtable([mainFolderPath, 'H_0_01/persistence_factor_sd.csv']);

disp_data_2 = readtable([mainFolderPath,'H_0_25/displacement_data.csv']);
dist_data_2 = readtable([mainFolderPath,'H_0_25/distance_data.csv']);
msd_data_2 = readtable([mainFolderPath,'H_0_25/MSD_data.csv']);
disp_sd_data_2 = readtable([mainFolderPath,'H_0_25/displacement_sd_data.csv']);
dist_sd_data_2 = readtable([mainFolderPath,'H_0_25/distance_sd_data.csv']);
msd_sd_data_2 = readtable([mainFolderPath,'H_0_25/MSD_sd_data.csv']);
pers_fac_2 = readtable([mainFolderPath, 'H_0_25/persistence_factor.csv']);
pers_fac_sd_data_2 = readtable([mainFolderPath, 'H_0_25/persistence_factor_sd.csv']);

disp_data_3 = readtable([mainFolderPath,'H_0_5/displacement_data.csv']);
dist_data_3 = readtable([mainFolderPath,'H_0_5/distance_data.csv']);
msd_data_3 = readtable([mainFolderPath,'H_0_5/MSD_data.csv']);
disp_sd_data_3 = readtable([mainFolderPath,'H_0_5/displacement_sd_data.csv']);
dist_sd_data_3 = readtable([mainFolderPath,'H_0_5/distance_sd_data.csv']);
msd_sd_data_3 = readtable([mainFolderPath,'H_0_5/MSD_sd_data.csv']);
pers_fac_3 = readtable([mainFolderPath, 'H_0_5/persistence_factor.csv']);
pers_fac_sd_data_3 = readtable([mainFolderPath, 'H_0_5/persistence_factor_sd.csv']);

disp_data_4 = readtable([mainFolderPath,'H_0_75/displacement_data.csv']);
dist_data_4 = readtable([mainFolderPath,'H_0_75/distance_data.csv']);
msd_data_4 = readtable([mainFolderPath,'H_0_75/MSD_data.csv']);
disp_sd_data_4 = readtable([mainFolderPath,'H_0_75/displacement_sd_data.csv']);
dist_sd_data_4 = readtable([mainFolderPath,'H_0_75/distance_sd_data.csv']);
msd_sd_data_4 = readtable([mainFolderPath,'H_0_75/MSD_sd_data.csv']);
pers_fac_4 = readtable([mainFolderPath, 'H_0_75/persistence_factor.csv']);
pers_fac_sd_data_4 = readtable([mainFolderPath, 'H_0_75/persistence_factor_sd.csv']);

disp_data_5 = readtable([mainFolderPath,'H_0_99/displacement_data.csv']);
dist_data_5 = readtable([mainFolderPath,'H_0_99/distance_data.csv']);
msd_data_5 = readtable([mainFolderPath,'H_0_99/MSD_data.csv']);
disp_sd_data_5 = readtable([mainFolderPath,'H_0_99/displacement_sd_data.csv']);
dist_sd_data_5 = readtable([mainFolderPath,'H_0_99/distance_sd_data.csv']);
msd_sd_data_5 = readtable([mainFolderPath,'H_0_99/MSD_sd_data.csv']);
pers_fac_5 = readtable([mainFolderPath, 'H_0_99/persistence_factor.csv']);
pers_fac_sd_data_5 = readtable([mainFolderPath, 'H_0_99/persistence_factor_sd.csv']);

% Concatenate tables to have one big dataset for each value:
merged_disp = [disp_data_1,disp_data_2,disp_data_3,disp_data_4,disp_data_5];
merged_dist = [dist_data_1,dist_data_2,dist_data_3,dist_data_4,dist_data_5];
merged_MSD = [msd_data_1,msd_data_2,msd_data_3,msd_data_4,msd_data_5];
merged_disp_sd = [disp_sd_data_1,disp_sd_data_2,disp_sd_data_3,disp_sd_data_4,disp_sd_data_5];
merged_dist_sd = [dist_sd_data_1,dist_sd_data_2,dist_sd_data_3,dist_sd_data_4,dist_sd_data_5];
merged_MSD_sd = [msd_sd_data_1,msd_sd_data_2,msd_sd_data_3,msd_sd_data_4,msd_sd_data_5];
merged_persfac = [pers_fac_1,pers_fac_2,pers_fac_3,pers_fac_4,pers_fac_5];
merged_persfac_sd = [pers_fac_sd_data_1,pers_fac_sd_data_2,pers_fac_sd_data_3,pers_fac_sd_data_4,pers_fac_sd_data_5];

%% 3D PLOT FOR PERSISTENCE FACTOR
models = merged_persfac.Properties.VariableNames;
% Extract different Dr values
Dr_values = zeros(1, numel(models));
for i = 1:numel(models)/5
    params = strsplit(models{i},'__');
    Dr_named = params{2}
    Model_named = params{3};
    Dr_named = strrep(Dr_named,'_','.');
    H_named_1 = strrep(Model_named,'_','.');
    H_named = strrep(H_named_1,'H.','');
    % Extract numerical values from "Dr" and "H" parts
    Dr_values(i:7:numel(models)) = str2double(strrep(Dr_named, 'Dr.', ''));
end

% Extract different H values
H_values = zeros(1, numel(models));
for i = 1:numel(models)
    params = strsplit(models{i},'__');
    H_named = params{3};
    H_named = strrep(H_named,'_','.');
    % Extract numerical values from "Dr" and "H" parts
    H_values(i) = str2double(strrep(H_named, 'H.', ''));
end
% Extract different Persistence Factor values
persistence_values = table2array(merged_persfac);

%FOR FILTERING THE DATA AFTER SIMULATIONS WERE MADE WIITH DR=0
% Define the range of Dr values you want to plot
min_Dr = 0;  % Define your minimum Dr value
max_Dr = 100; % Define your maximum Dr value
% Find the indices of Dr values within the specified range
valid_indices = (Dr_values >= min_Dr) & (Dr_values <= max_Dr);
% Filter the data based on the valid indices
filtered_Dr_values = Dr_values(valid_indices);
filtered_H_values = H_values(valid_indices);
filtered_persistence_values = persistence_values(valid_indices);

% Create a grid for interpolation
[X, Y] = meshgrid(unique(filtered_Dr_values), unique(filtered_H_values));
% Interpolate the persistence_values
Z = griddata(filtered_Dr_values, filtered_H_values, filtered_persistence_values, X, Y, 'cubic');
% Apply a logarithmic scale to the x-axis (Dr values)
X = log10(X);
% Create a 3D surface plot
figure;
surf(X, Y, Z);
% imagesc(X(1,:), Y(:,1), Z);
% Add labels and a colorbar
xlabel('Dr (log 10)');
ylabel('H');
zlabel('Persistence Factor Values');
title('Persistence Factor 3D Plot');
subtitle('Alpha = 2.5');
% Customize the colormap if needed
colormap(jet);
% Add a colorbar for the z-axis (Persistence Values)
colorbar;

%% 3D PLOT FOR STANDARD DEVIATION OF SQUARE DISPLACEMENT
msd_sd_vals = table2array(mean(merged_MSD_sd));
% Create a grid for interpolation
[X, Y] = meshgrid(unique(Dr_values), unique(H_values));
% Interpolate the persistence_values
Z = griddata(Dr_values, H_values, msd_sd_vals, X, Y, 'cubic');
% Apply a logarithmic scale to the x-axis (Dr values)
X = log10(X);
% Create a 3D surface plot
figure;
surf(X, Y, Z);
% Add labels and a colorbar
xlabel('Dr(log 10)');
ylabel('H');
zlabel('Std. Dev. MSD');
title('Standar Deviation of Square Displacement');
subtitle('Alpha = 2.5, with Mean of Time Intervals');
% Customize the colormap if needed
colormap(jet);
% Add a colorbar for the z-axis (Persistence Values)
colorbar;
%% "GRAVEYARD" PLOT
close all;
% Specify the path to the main folder with data
mainFolderPath_H = 'Model1_H_cnst/Model1_2D_Batch200/Alpha_2_5/H_0_01/';
% Get a list of subfolders in the main folder
subfolders = dir(mainFolderPath_H);
subfolders = subfolders([subfolders.isdir]);  % Filter out non-directories

final_pos = [];
% Array to store the subfolder names (header for mean_displacement)
subfolderNames = {};
% Iterate through each model (combination of H and Dr)
for i = 3:length(subfolders)  % Start from 3 to skip '.' and '..' entries
    subfolder = subfolders(i).name;
    subfolderPath = fullfile(mainFolderPath_H,subfolder);
    csvFiles = dir(fullfile(subfolderPath,'*.csv'));
    model_name = strrep(subfolder, 'data__', '')
    params = strsplit(model_name,'__');
    params{1} = strrep(params{1},'2_','2.');
    params{2} = strrep(params{2},'0_','0.');
    params{3} = strrep(params{3},'0_','0.');
    % Plot final position (X,Y)
    figure;
    axis ([-200,200,-200,200]);

    hold on;
    for j = 1:length(csvFiles)
        csvFileName = csvFiles(j).name;        
        csvFilePath = fullfile(subfolderPath,csvFileName);    
        data_sim = readtable(csvFilePath);
        final_position = data_sim(end,:);
        scatter(final_position.X,final_position.Y, 20, 'filled','b')
    end
    hold off;
    box on;
    % EDITING
    axis ([-200,200,-200,200]);
    axis on;
    set(gca,'LineWidth',3)
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca,'TickLength',[0 0])
    % SAVING PNG FIGURE
    filename = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_0/Figures/GY/',model_name,'_GY.png'];
    % exportgraphics(gcf,filename,'Resolution',600);
    close(gcf);
end

%% MSD PLOT FOR EACH H VALUE (7 Drs)
for msd_model_index = 1:width(merged_MSD)
    msd_model_name = merged_MSD.Properties.VariableNames{msd_model_index}
    msd_modelData = table2array(merged_MSD(:, msd_model_index));
    msd_modelData(msd_modelData == 0) = NaN;
%% MSD SLOPE REGRESSION ANALYSIS
close all;
coeff_list = zeros(1,numel(models));

for msd_model_index = 1:width(merged_MSD)
    msd_model_name = merged_MSD.Properties.VariableNames{msd_model_index}
    msd_modelData = table2array(merged_MSD(:, msd_model_index));
    msd_modelData(msd_modelData == 0) = NaN;
    % Extract only the last third of the data
    total_length = length(msd_modelData);
    last_third = msd_modelData(round(2 * total_length / 3) - 1:end-1);
    % Create time vector for the last third
    time_vector_last_third = (round(2 * total_length / 3) - 1:total_length-1)';  
    % Lineal Fit Section: Fit a linear model using fitlm
    lm_model = fitlm(time_vector_last_third, last_third, 'linear');
    % Extract coefficients and fitted values for the linear model
    lm_coefficients = lm_model.Coefficients.Estimate;
    lm_fitted_values = predict(lm_model, time_vector_last_third);
    % Calculate R-squared
    lm_r_squared = lm_model.Rsquared.Ordinary; 
    coeff_list(msd_model_index) = lm_coefficients(2);
end

%% 3D PLOT FOR THE BETA COEFFICIENT OF THE MSD REGRESSION
% Create a grid for interpolation
[X, Y] = meshgrid(unique(Dr_values), unique(H_values));
% Interpolate the persistence_values
Z = griddata(Dr_values, H_values, coeff_list, X, Y, 'cubic');
% Apply a logarithmic scale to the x-axis (Dr values)
X = log10(X);
% Create a 3D surface plot
figure;
surf(X, Y, Z);
% Add labels and a colorbar
xlabel('Dr(log 10)');
ylabel('H');
zlabel('Beta Coefficient');
title('3D Plot for Beta Coefficient');
subtitle('Alpha = 2.5');
% Customize the colormap if needed
colormap(jet);
% Add a colorbar for the z-axis (Persistence Values)
colorbar;