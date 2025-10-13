%% "GRAVEYARD" PLOT
close all;
% Specify the path to the main folder with data
mainFolderPath_H = '../DATA_2025/Batch25_InConTest/Alpha_0/H_0_5/';
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
    axis ([-100,100,-100,100]);
    axis on;
    set(gca,'LineWidth',3)
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca,'TickLength',[0 0])
    % SAVING PNG FIGURE
    filename = ['../DATA_2025/InConTest_GY_2025/',model_name,'_InConTest_GY.png'];
    exportgraphics(gcf,filename,'Resolution',300,'Width',800,'Height',800);
    % close(gcf);
end