close all force; clear; clc;

% === RUTAS ===
base_path = '/MATLAB Drive/MODELS_2025_/DATA_2025_DoubleCorr_Na2';
save_path = '/MATLAB Drive/MODELS_2025_/DATA_2025_DoubleCorr_Traj_Na2';

% Crear carpeta de guardado si no existe
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

% === VALORES DE H ===
H1_values = [0.5, 0.75, 0.99];  % H del ángulo
H2_values = [0.5, 0.75, 0.99];  % H del movimiento

for h1 = 1:length(H1_values)
    for h2 = 1:length(H2_values)
        % Strings para nombres
        H1 = H1_values(h1); H2 = H2_values(h2);
        H1_str = strrep(num2str(H1), '.', '_');
        H2_str = strrep(num2str(H2), '.', '_');

        % Carpeta de simulaciones
        folder_path = fullfile(base_path, ['HAngle_', H1_str, '__HMotion_', H2_str]);

        % Listar archivos CSV
        file_list = dir(fullfile(folder_path, '*.csv'));
        if length(file_list) < 6
            warning('No hay suficientes simulaciones para H1=%.2f, H2=%.2f', H1, H2);
            continue;
        end

        % Seleccionar 6 archivos aleatorios
        idx = randperm(length(file_list), 6);
        selected_files = file_list(idx);

        % Cargar datos
        traj_data = cell(1, 6);
        for k = 1:6
            traj_data{k} = readtable(fullfile(folder_path, selected_files(k).name));
        end

        % === GRAFICAR ===
        colormapTime = autumn(1000);
        ataxis = 0.5;
        figure;
        hold on;

        for k = 1:6
            x = traj_data{k}.X;
            y = traj_data{k}.Y;
            if ~isempty(x) && ~isempty(y)
                scatter(x, y, 10, 1:length(x), 'filled', 'MarkerFaceAlpha', ataxis);
            else
                warning('Archivo %s no contiene columnas X e Y válidas.', selected_files(k).name);
            end
        end

        % Punto de referencia
        x_org = -100 / sqrt(2);
        scatter(x_org, x_org, 60, 'k', 'filled');

        % Estilo de la figura
        colormap(flipud(colormapTime));
        axis([-100, 100, -100, 100]);
        axis on; box on;
        set(gca, 'LineWidth', 3, 'XTickLabel', [], 'YTickLabel', [], 'TickLength', [0 0]);

        % === GUARDAR FIGURA ===
        filename = fullfile(save_path, ['Traj__HAngle_', H1_str, '__HMotion_', H2_str, '.png']);
        exportgraphics(gcf, filename, 'Resolution', 300, 'ContentType', 'image', ...
                       'BackgroundColor', 'white');
        close(gcf);
        fprintf('Guardado: %s\n', filename);
    end
end
