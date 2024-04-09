close all;clear all;clc;

start_time = tic;

%% Specify the path to the main folder with data
H_values = [0.5];
Alpha_values = [0];
for Alphaval = 1:length(Alpha_values)
    Alpha = Alpha_values(Alphaval);
    Alpha_str = strrep(num2str(Alpha), '.', '_');
    for Hval = 1:length(H_values)
        H = H_values(Hval);
        H_str = strrep(num2str(H), '.', '_');
        mainFolderPath = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_',Alpha_str,'/H_',H_str,'/']
        % Get a list of subfolders in the main folder
        subfolders = dir(mainFolderPath);
        subfolders = subfolders([subfolders.isdir]);  % Filter out non-directories
        
        %% DISPLACEMENT, DISTANCE, MSD ADN THETA QUANTIFICATION
        % Arrays to store displacements, distances, MSDs, and Thetas
        model_displacements = [];
        model_distances = [];
        model_MSD = [];
        model_theta = [];
        % Arrays to store standar deviations of models
        model_dist_sds = [];
        model_disp_sds = [];
        model_msd_sds = [];
        model_theta_sds = [];
        % Arrays for Noise Norms
        model_fmpi_x = [];
        model_fmpi_y = [];
        model_xi_x = [];
        model_xi_y = [];
        model_norm_fmpi = [];
        model_norm_xi = [];
        mean_time_norm_xi = [];
        % Array for Persistence Factor
        pers_fac = [];
        pers_fac_sd = [];
        
        % Array to store the subfolder names (header for mean_displacement)
        subfolderNames = {};
        
        % Iterate through each model (combination of H and Dr)
        for i = 3:length(subfolders)  % Start from 3 to skip '.' and '..' entries
            subfolder = subfolders(i).name;
            subfolderNames = [subfolderNames,subfolder];
            subfolderPath = fullfile(mainFolderPath,subfolder);
            csvFiles = dir(fullfile(subfolderPath,'*.csv'));
            
            % Array for each model that will have all values
            file_dist_full = zeros(1000,length(csvFiles));
            file_disp_full = zeros(1000,length(csvFiles));
            file_msd_full = zeros(1000,length(csvFiles));
            file_theta_full = zeros(1000,length(csvFiles));
            file_fmpi_x_full = zeros(1000,length(csvFiles));
            file_fmpi_y_full = zeros(1000,length(csvFiles));
            file_xi_x_full = zeros(1000,length(csvFiles));
            file_xi_y_full = zeros(1000,length(csvFiles));
            file_norm_xi_full = zeros(1000,length(csvFiles));
            file_norm_fmpi_full = zeros(1000,length(csvFiles));
            file_mean_time_norm_xi = zeros(1, length(csvFiles));
            file_pers_fac = zeros(1,length(csvFiles));
        
            % Open each csv inside the subfolder
            for j = 1:length(csvFiles)
                csvFileName = csvFiles(j).name;        
                csvFilePath =fullfile(subfolderPath,csvFileName);    
                data_sim = readtable(csvFilePath);
                % Start values t calculate the full distance and distance^2
                total_distance = 0; total_norm_time_xi = 0;
                %% FOR DISTANCE, DISPLACEMENT,THETA AND NORMS OF FmPi and Xi
                for t = 2:height(data_sim)
                    % Get position at each timestep
                    position = sqrt(data_sim.X(t)^2 + data_sim.Y(t)^2);
                    % Get distance travelled between each timestep
                    travel = sqrt((data_sim.X(t)-data_sim.X(t-1))^2 + (data_sim.Y(t)-data_sim.Y(t-1))^2);
                    total_distance = total_distance + travel;
                    theta_change = abs(data_sim.Theta(t)-data_sim.Theta(t-1));
                    norm_fmpi = sqrt(data_sim.FmPi_x(t)^2 + data_sim.FmPi_y(t)^2);
                    norm_xi = sqrt(data_sim.Xi_x(t)^2 + data_sim.Xi_y(t)^2);
                    total_norm_time_xi = total_norm_time_xi + norm_xi;
                    % Store position and travel values for this timestep in the current file's arrays
                    file_disp_full(t,j) = position;    
                    file_dist_full(t,j) = total_distance;
                    file_theta_full(t,j) = theta_change;
                    file_fmpi_x_full(t,j) = data_sim.FmPi_x(t);
                    file_fmpi_y_full(t,j) = data_sim.FmPi_y(t);
                    file_xi_x_full(t,j) = data_sim.Xi_x(t);
                    file_xi_y_full(t,j) = data_sim.Xi_y(t);
                    file_norm_fmpi_full(t,j) = norm_fmpi;
                    file_norm_xi_full(t,j) = norm_xi; 
                end
                file_mean_time_norm_xi(1,j) = total_norm_time_xi/t;
                %% FOR PERSISTENCE FACTOR
                file_pers_fac(1,j) = position/total_distance;
                %% FOR MSD WITH DIFFERENT TIME INTERVALS (WANG et al., 2014)
                for time_interval = 1:999 % Intervals of time
                    sum_term = 0;
                    for t = 1:(height(data_sim)-time_interval)
                        x_sqr = (data_sim.X(t+time_interval)-data_sim.X(t))^2;
                        y_sqr = (data_sim.Y(t+time_interval)-data_sim.Y(t))^2;
                        dist_sqr = sqrt(x_sqr+y_sqr)^2;
                        sum_term = sum_term + dist_sqr;
                    end
                    MSD = sum_term/(t);
                    % Results will be in a file that has ROWS = Time Intervals;
                    % COLUMNS = # Sim Models (50)
                    file_msd_full(time_interval,j) = MSD;
                end
            end
            mean_model_dist = mean(file_dist_full,2);
            mean_model_disp = mean(file_disp_full,2);    
            mean_model_theta = mean(file_theta_full,2);
            sd_model_dist = std(file_dist_full,0,2);
            sd_model_disp = std(file_disp_full,0,2);    
            sd_model_theta = std(file_theta_full,0,2);
            
            %% Persistence Factor
            mean_pers_fac = mean(file_pers_fac,2);
            sd_pers_fac = std(file_pers_fac,0,2);
            
            %% MSD TABLES (COMMENT IF NOT CALCULATED)
            mean_model_MSD = mean(file_msd_full,2);
            sd_model_MSD = std(file_msd_full,0,2);
            model_MSD = [model_MSD,mean_model_MSD];
            model_msd_sds = [model_msd_sds,sd_model_MSD];
            
            model_displacements = [model_displacements,mean_model_disp];
            model_distances = [model_distances,mean_model_dist];    
            model_theta = [model_theta,mean_model_theta];
            model_dist_sds = [model_dist_sds,sd_model_dist];
            model_disp_sds = [model_disp_sds,sd_model_disp];    
            model_theta_sds = [model_theta_sds,sd_model_theta];
            %% Persistence Factor Table
            pers_fac = [pers_fac, mean_pers_fac];
            pers_fac_sd = [pers_fac_sd, sd_pers_fac];
            %% TABLES FOR ALPHA (NOISE NORM) CALCULATION (COMMENT IF NOT CALCULATED)
            mean_fmpi_x = mean(file_fmpi_x_full,2);
            mean_fmpi_y = mean(file_fmpi_y_full,2);
            mean_xi_x = mean(file_xi_x_full,2);
            mean_xi_y = mean(file_xi_y_full,2);
            mean_norm_fmpi = mean(file_norm_fmpi_full,2);
            mean_norm_xi = mean(file_norm_xi_full,2);
            mean_norm_time_xi = mean(file_mean_time_norm_xi,2);
            model_fmpi_x = [model_fmpi_x,mean_fmpi_x];
            model_fmpi_y = [model_fmpi_y,mean_fmpi_y];
            model_xi_x = [model_xi_x,mean_xi_x];
            model_xi_y = [model_xi_y,mean_xi_y];
            model_norm_xi = [model_norm_xi,mean_norm_xi];
            model_norm_fmpi = [model_norm_fmpi,mean_norm_fmpi];
            mean_time_norm_xi = [mean_time_norm_xi, mean_norm_time_xi];
        end
        
        %% SAVING THE DATA (CHECK SAVING FOLDER BELOW)
        
        % Iterate through subfolderNames and remove "data__" from each name
        for i = 1:length(subfolderNames)
            subfolderNames{i} = strrep(subfolderNames{i}, 'data__', '');
        end
        
        % Create tables from mean arrays with column names
        displacementTable = array2table(model_displacements, 'VariableNames', subfolderNames);
        distanceTable = array2table(model_distances, 'VariableNames', subfolderNames);
        thetaTable = array2table(model_theta, 'VariableNames', subfolderNames);
        % Create tables from standar deviations arrays with column names
        displacementsdTable = array2table(model_disp_sds, 'VariableNames', subfolderNames);
        distancesdTable = array2table(model_dist_sds, 'VariableNames', subfolderNames);
        thetasdTable = array2table(model_theta_sds, 'VariableNames', subfolderNames);
        % Persistence Factor
        persfacTable = array2table(pers_fac,'VariableNames', subfolderNames);
        persfacsdTable = array2table(pers_fac_sd,'VariableNames', subfolderNames);
        %% MSD VALUES (COMMENT IF NOT CALCULATED)
        MSDTable = array2table(model_MSD, 'VariableNames', subfolderNames);
        MSDsdTable = array2table(model_msd_sds, 'VariableNames', subfolderNames);
    
        %% ALPHA VALUES (COMMENT IF NOT CALCULATED)
        fmpi_x_table = array2table(model_fmpi_x, 'VariableNames',subfolderNames);
        fmpi_y_table = array2table(model_fmpi_y, 'VariableNames',subfolderNames);
        xi_x_table = array2table(model_xi_x, 'VariableNames', subfolderNames);
        xi_y_table = array2table(model_xi_y, 'VariableNames', subfolderNames);
        norm_fmpi_table = array2table(model_norm_fmpi, 'VariableNames', subfolderNames);
        norm_xi_table = array2table(model_norm_xi, 'VariableNames', subfolderNames);
        mean_time_norm_xi_table = array2table(mean_time_norm_xi,'VariableNames', subfolderNames);
        
        % Save the tables to CSV files
        writetable(displacementTable, ([mainFolderPath,'displacement_data.csv']));
        writetable(distanceTable, ([mainFolderPath,'distance_data.csv']));
        writetable(thetaTable, ([mainFolderPath,'theta_data.csv']));
        writetable(displacementsdTable, ([mainFolderPath,'displacement_sd_data.csv']));
        writetable(distancesdTable, ([mainFolderPath,'distance_sd_data.csv']));
        writetable(thetasdTable, ([mainFolderPath,'theta_sd_data.csv']));
        writetable(persfacTable, ([mainFolderPath,'persistence_factor.csv']));
        writetable(persfacsdTable, ([mainFolderPath,'persistence_factor_sd.csv']));
        %% MSD csvs (COMMENT IF NOT CALCULATED)
        writetable(MSDTable,([mainFolderPath,'MSD_data.csv']))
        writetable(MSDsdTable,([mainFolderPath,'MSD_sd_data.csv']))
    
        %% ALPHA QUANT csvs (COMMENT IF NOT CACULATED)
        writetable(fmpi_x_table, ([mainFolderPath,'fmpi_x.csv']));
        writetable(fmpi_y_table, ([mainFolderPath,'fmpi_y.csv']));
        writetable(xi_x_table, ([mainFolderPath,'xi_x.csv']));
        writetable(xi_y_table, ([mainFolderPath,'xi_y.csv']));
        writetable(norm_fmpi_table, ([mainFolderPath,'norm_fmpi.csv']));
        writetable(norm_xi_table,([mainFolderPath,'norm_xi.csv']));
        writetable(mean_time_norm_xi_table, ([mainFolderPath,'mean_time_norm_xi.csv'])); 
        end_time = toc(start_time);
        fprintf('Elapsed Time: %.2f seconds\n', end_time);
    end
end