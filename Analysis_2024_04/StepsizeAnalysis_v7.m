close all; clear all;
%% STEPSIZE MEAN, StdDev
% H_values = [0.5,0.75,0.99];
H_values = [0.5];
% Dr_values = [0,0.1,0.5,1,5,10];
Dr_values = [0];
colors_per_dr = {'m','b','c','g','y','r'};

mean_stepsize_matrix = zeros(length(H_values),length(Dr_values));
stddev_stepsize_matrix = zeros(length(H_values),length(Dr_values));
% For each H value
for Hval = 1:length(H_values)
    H_value = H_values(Hval);
    H_str = strrep(num2str(H_values(Hval)), '.', '_');
    mainFolderPath = ['Model1_H_cnst/Model1_2D_Batch200_v5/Alpha_0/H_',H_str,'/'];  
    % For each Dr value in order of reading Dr_values
    for Drval = 1:length(Dr_values)  
        Dr_value = Dr_values(Drval);
        Dr_str = strrep(num2str(Dr_value), '.', '_');
        subfolder = ['/data__Alpha_0__Dr_',Dr_str,'__H_',H_str]
        subfolderPath = fullfile(mainFolderPath,subfolder);
        csvFiles = dir(fullfile(subfolderPath,'*.csv')); 
        % Array for each model that will have all values
        file_step_full = zeros(999,length(csvFiles));
        figure;
        % For each csv file        
        for j = 1:length(csvFiles)
            csvFileName = csvFiles(j).name;        
            csvFilePath =fullfile(subfolderPath,csvFileName);
            data_sim = readtable(csvFilePath);
            % For each timestep
            for t = 2:height(data_sim)
                diff_X = (data_sim.X(t) - data_sim.X(t-1))^2;
                diff_Y = (data_sim.Y(t) - data_sim.Y(t-1))^2;
                stepsize = sqrt(diff_X + diff_Y);
                file_step_full(t-1,j) = stepsize;
            end
        end 
        sims_average_stepsize = mean(file_step_full);
        sims_deviation_stepsize = std(file_step_full);
        scatter(sims_average_stepsize,sims_deviation_stepsize,30,[0.5 0.2 0.55],...
            'filled','DisplayName',['H=',num2str(H_value),' Dr=',num2str(Dr_value)]);
        axis ([0,0.2,0,0.1])
        axis on;
        box on;
        set(gca,'LineWidth',3)
        set(gca, 'XTickLabel', []);
        set(gca, 'YTickLabel', []);
        set(gca,'TickLength',[0 0])
        % Saving
        filename = ['Model1_H_cnst/FIGURES_082024/' ...
            'ScatterStepsize/Alpha_0__Dr_', Dr_str, '__H_',H_str,'__ScatterStepsize.png'];
        legend_mean = mean(sims_average_stepsize)
        legend_deviation = mean(sims_deviation_stepsize)
        exportgraphics(gcf,filename,'Resolution',300);
        % close(gcf);
    end    
end