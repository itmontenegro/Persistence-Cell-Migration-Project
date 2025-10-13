close all; clear; clc;

% === RUTAS ===
mainFolderPath_H = '/MATLAB Drive/MODELS_2025_/DATA_2025_DoubleCorr_Na2';
saveFolderPath_GY = '/MATLAB Drive/MODELS_2025_/DATA_2025_DoubleCorr_GY_Na2';

% Crear carpeta de guardado si no existe
if ~exist(saveFolderPath_GY, 'dir')
    mkdir(saveFolderPath_GY);
end

% Obtener subcarpetas
subfolders = dir(mainFolderPath_H);
subfolders = subfolders([subfolders.isdir]);  % Solo carpetas

% Iterar por cada subcarpeta
for i = 3:length(subfolders)  % Saltar '.' y '..'
    subfolder = subfolders(i).name;
    subfolderPath = fullfile(mainFolderPath_H, subfolder);
    csvFiles = dir(fullfile(subfolderPath, '*.csv'));

    % === Extraer nombre de modelo
    model_name = subfolder;  % ya está en formato HAngle_0_75__HMotion_0_99

    % === Plotear posiciones finales
    figure;
    hold on;
    for j = 1:length(csvFiles)
        csvFilePath = fullfile(subfolderPath, csvFiles(j).name);
        data_sim = readtable(csvFilePath);
        final_position = data_sim(end, :);
        scatter(final_position.X, final_position.Y, 20, [0 0.4470 0.7410], 'filled');
    end
    hold off;
    box on;

    % Estilo del gráfico
    axis([-100, 100, -100, 100]);
    axis on;
    set(gca, 'LineWidth', 3);
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca, 'TickLength', [0 0]);

    % === Guardar figura PNG
    save_filename = fullfile(saveFolderPath_GY, [model_name, '_GY.png']);
    exportgraphics(gcf, save_filename, 'Resolution', 300, 'ContentType', 'image');
    close(gcf);

    fprintf('Guardado: %s\n', save_filename);
end
