close all; clear; clc;

% === RUTAS ===
mainFolderPath_H = '/MATLAB Drive/MODELS_2025_/DATA_2025_DoubleCorr_Na2';
saveFolderPath_GY = '/MATLAB Drive/MODELS_2025_/DATA_2025_DoubleCorr_GY_Na2.1';

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

    model_name = subfolder;  % Nombre ya válido

    % === Crear figura
    figure;
    hold on;

    % === Graficar posiciones finales
    for j = 1:length(csvFiles)
        csvFilePath = fullfile(subfolderPath, csvFiles(j).name);
        data_sim = readtable(csvFilePath);

        % Validar nombres de columna
        data_sim.Properties.VariableNames = matlab.lang.makeValidName(data_sim.Properties.VariableNames);
        varNames = data_sim.Properties.VariableNames;

        idxX = find(strcmpi(varNames, 'X'));
        idxY = find(strcmpi(varNames, 'Y'));

        if isempty(idxX) || isempty(idxY)
            warning('Archivo %s no contiene columnas X e Y válidas.', csvFilePath);
            continue;
        end

        x_final = data_sim{end, idxX};
        y_final = data_sim{end, idxY};

        scatter(x_final, y_final, 20, [0 0.4470 0.7410], 'filled');
    end

    % === Posición del organizador (punto negro)
    x_org = -100 / sqrt(2);
    scatter(x_org, x_org, 60, 'k', 'filled');

    hold off;
    box on;

    % Estilo del gráfico
    axis([-100, 100, -100, 100]);
    axis on;
    set(gca, 'LineWidth', 3);
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca, 'TickLength', [0 0]);

    % === Guardar imagen
    save_filename = fullfile(saveFolderPath_GY, [model_name, '_GY.png']);
    exportgraphics(gcf, save_filename, 'Resolution', 300, 'ContentType', 'image');
    close(gcf);

    fprintf('Guardado: %s\n', save_filename);
end
