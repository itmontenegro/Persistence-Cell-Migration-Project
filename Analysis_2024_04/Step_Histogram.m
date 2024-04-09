close all; clear all;
H_values = [0.01,0.25,0.5,0.75,0.99];
for i = 1:length(H_values)
    H_str = strrep(num2str(H_values(i)), '.', '_');
    mainFolderPath = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_5/H_',H_str,'/'];
    % Get a list of subfolders in the main folder
    subfolders = dir(mainFolderPath);
    subfolders = subfolders([subfolders.isdir]);  % Filter out non-directories
    model_step = [];
    subfolderNames = {};
    for i = 3:length(subfolders)  % Start from 3 to skip '.' and '..' entries
        subfolder = subfolders(i).name;
        subfolderNames = [subfolderNames,subfolder];
        subfolderPath = fullfile(mainFolderPath,subfolder);
        csvFiles = dir(fullfile(subfolderPath,'*.csv'));
    
        % Array for each model that will have all values
        file_step_full = zeros(1000,length(csvFiles));
        % Open each csv inside the subfolder
        for j = 1:length(csvFiles)
            csvFileName = csvFiles(j).name;        
            csvFilePath =fullfile(subfolderPath,csvFileName);    
            data_sim = readtable(csvFilePath);
            % Start values t calculate the full distance and distance^2
            total_distance = 0;
            %% FOR STEP SIZE AT EVERY TIMESTEP
            for t = 2:height(data_sim)
                % Get position at each timestep
                step = sqrt((data_sim.X(t)-data_sim.X(t-1))^2 + (data_sim.Y(t)-data_sim.Y(t-1))^2);
                file_step_full(t,j) = step;
            end        
        end
        mean_steps = mean(file_step_full);
        %%
        figure;
        histogram(mean_steps,[0:0.05:1],'Normalization','probability');
        % EDITING
        axis ([0 1 0 1]);
        axis on;
        % box on;
        set(gca,'LineWidth',3)
        % set(gca, 'XTickLabel', []);
        % set(gca, 'YTickLabel', []);
        set(gca,'TickLength',[0 0])
        % SAVING PNG FIGURE
        filename = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_5/Figures/Histograms/',subfolder,'_Histogram.png']
        exportgraphics(gcf,filename,'Resolution',300);
        close(gcf);
    end
end