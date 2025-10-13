close all; clear all;

% Definir los valores de f_tax y H a iterar
f_tax_values = [0.01,0.1];  
H_values = [0.5];  

% Definir la ruta principal de datos y de guardado
main_taxis = 'DATA_202555/';
save_path = 'Prueba_1/'; % Ruta corregida

for f_tax = f_tax_values
    for H = H_values
        % Corregir formato de H y f_tax para coincidir con las carpetas
        H_str = strrep(sprintf('%.2f', H), '.', '_'); % Evita H_0_990
        f_tax_str = strrep(sprintf('%.3f', f_tax), '.', '_'); % Mantiene precisión en 3 decimales

        % Definir nombres de carpetas
        f_tax_folder = ['alpha_0_f_tax_', f_tax_str];
        H_folder = ['H_', H_str];

        % Ruta de la carpeta que contiene los valores de Dr
        dr_path = fullfile(main_taxis, f_tax_folder, H_folder);

        % Depuración: Mostrar la ruta generada
        fprintf('Verificando carpeta: %s\n', dr_path);

        % Verificar si la carpeta existe
        if ~isfolder(dr_path)
            warning(['No existe la carpeta: ', dr_path]);
            continue;
        end

        % Obtener todas las carpetas dentro de dr_path
        dr_folders = dir(dr_path);
        dr_folders = dr_folders([dr_folders.isdir]);  
        dr_folders = dr_folders(~ismember({dr_folders.name}, {'.', '..'}));

        for k = 1:length(dr_folders)
            Dr_folder = dr_folders(k).name;

            % Verificar que la carpeta siga el formato 'Dr_X'
            if startsWith(Dr_folder, 'Dr_')
                Dr_str = erase(Dr_folder, 'Dr_');  
            else
                warning(['Carpeta inesperada en ', dr_path, ': ', Dr_folder]);
                continue;
            end

            data_path = fullfile(dr_path, Dr_folder);
            csv_files = dir(fullfile(data_path, '*.csv'));

            num_files = length(csv_files);
            if num_files < 100
                warning(['No hay suficientes simulaciones en ', data_path]);
                continue;
            end

            % Seleccionar 99 archivos aleatorios
            permuted_indices = randperm(num_files, 100);
            data = cell(1, 100);

            for i = 1:100
                file_name = fullfile(data_path, csv_files(permuted_indices(i)).name);
                data{i} = readtable(file_name);
            end

            % Graficar solo las posiciones finales
            figure;
            hold on;
            ataxis = 0.5;

            % Posición del organizador
            x_org = -100 / sqrt(2);
            scatter(x_org, x_org, 40, 'filled', 'k'); % Punto organizador en negro

            for j = 1:100
                final_position = data{j}(end, :); % Última posición de cada archivo
                scatter(final_position.x_array, final_position.y_array, 20, 'filled', 'b');
            end

            axis([-100, 100, -100, 100]);
            axis on; box on;
            set(gca, 'LineWidth', 3);
            set(gca, 'XTickLabel', []);
            set(gca, 'YTickLabel', []);
            set(gca, 'TickLength', [0 0]);

            % Guardar imagen en la carpeta correspondiente
            save_dir = fullfile(save_path, 'Trajectories_Alpha_0');
            if ~exist(save_dir, 'dir')
                mkdir(save_dir);
            end

            filename = fullfile(save_dir, ['FinalPos__f_', f_tax_str, '_Dr_', Dr_str, '_H_', H_str, '.png']);
            exportgraphics(gcf, filename, 'Resolution', 300);
            close(gcf);

            % Depuración: Mostrar la ruta donde se guardó la imagen
            fprintf('Imagen guardada en: %s\n', filename);
        end
    end
end
