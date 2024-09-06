% close all; clear all;
%% angle MEAN, StdDev
H_values = [0.5,0.75,0.99];
% Dr_values = [0,0.1,0.5,1,5,10];
% H_values = [0.5];
Dr_values = [0];
mean_angle_matrix = zeros(length(H_values),length(Dr_values));
stddev_angle_matrix = zeros(length(H_values),length(Dr_values));
colors_per_dr = {'m','b','c','g','y','r'};

% For each H value
for Hval = 1:length(H_values)
    H_value = H_values(Hval);
    H_str = strrep(num2str(H_values(Hval)), '.', '_');
    mainFolderPath = ['Model1_H_cnst/Model1_2D_Batch200_v5/Alpha_0_25/H_',H_str,'/'];  
    % For each Dr value in order of reading Dr_values
    for Drval = 1:length(Dr_values)  
        Dr_value = Dr_values(Drval);
        Dr_str = strrep(num2str(Dr_value), '.', '_');
        subfolder = ['/data__Alpha_0_25__Dr_',Dr_str,'__H_',H_str];
        subfolderPath = fullfile(mainFolderPath,subfolder);
        csvFiles = dir(fullfile(subfolderPath,'*.csv')); 
        % Array for each model that will have all values
        file_angle_full = zeros(999,length(csvFiles));
        % For each csv file
        for j = 1:length(csvFiles)
            csvFileName = csvFiles(j).name;        
            csvFilePath =fullfile(subfolderPath,csvFileName);
            data_sim = readtable(csvFilePath);
            % For each timestep
            for t = 1:(height(data_sim)-2)
            %for t = 2:height(data_sim)
                %diff_X = (data_sim.X(t) - data_sim.X(t-1))^2;
                %diff_Y = (data_sim.Y(t) - data_sim.Y(t-1))^2;
                %angle = sqrt(diff_X + diff_Y);
                %file_step_full(t-1,j) = angle;
                step_0 = sqrt((data_sim.X(t+1)-data_sim.X(t))^2 + (data_sim.Y(t+1)-data_sim.Y(t))^2);
                step_1 = sqrt((data_sim.X(t+2)-data_sim.X(t+1))^2 + (data_sim.Y(t+2)-data_sim.Y(t+1))^2);
                step_2 = sqrt((data_sim.X(t+2)-data_sim.X(t))^2 + (data_sim.Y(t+2)-data_sim.Y(t))^2);
                
                file_angle_full(t, j) = pi - acos((step_0^2+step_1^2-step_2^2)/(2*step_0*step_1));
            end
        end
        % At this point we have scanned all 200 sims and have one array
        % with ALL 999 (Nts-1) angles values

        %radians to degrees
        file_angle_full = rad2deg(file_angle_full);

        mean_angle_per_sim = mean(file_angle_full);
        mean_angle_per_model = mean(mean_angle_per_sim);
        mean_angle_matrix(Hval,Drval) = mean_angle_per_model;

        stddev_angle_per_sim = std(file_angle_full);
        mean_stddev_angle_per_model = mean(stddev_angle_per_sim);
        stddev_angle_matrix(Hval,Drval) = mean_stddev_angle_per_model;

        %% Scatter Plot
        model_name = ['H=',num2str(H_value),' Dr=',num2str(Dr_value)]
        model_average_angle = deg2rad(mean_angle_per_model)
        model_deviation_angle = deg2rad(mean_stddev_angle_per_model)
        scatter(mean_angle_per_sim,stddev_angle_per_sim,20,[0.5 0.2 0.55],...
            'filled','DisplayName',['H=',num2str(H_value),' Dr=',num2str(Dr_value)]);
        axis ([0,90,0,60])
        axis on;
        box on;
        set(gca,'LineWidth',3)
        set(gca, 'XTickLabel', []);
        set(gca, 'YTickLabel', []);
        set(gca,'TickLength',[0 0])
        % Saving
        filename = ['Model1_H_cnst/FIGURES_082024/' ...
            'ScatterAnglesize/Alpha_0_25_Dr_', Dr_str, '__H_',H_str,'__ScatterAnglesize.png'];
        exportgraphics(gcf,filename,'Resolution',300);
        % close(gcf);
    end
end
