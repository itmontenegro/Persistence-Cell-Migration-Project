%% "GRAVEYARD" PLOT
close all;
% Specify the path to the main folder with data
mainFolderPath_H = 'Model1_H_cnst/Model1_2D_Batch200_v5/Alpha_0_25/H_0_99/';
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
    axis ([-150,150,-150,150]);

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
    axis ([-150,150,-150,150]);
    axis on;
    set(gca,'LineWidth',3)
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca,'TickLength',[0 0])
    % SAVING PNG FIGURE
    filename = ['Model1_H_cnst/Model1_2D_Batch200_v5/Alpha_0_25/Figures/GY/',model_name,'_GY.png'];
    exportgraphics(gcf,filename,'Resolution',600);
    close(gcf);
end