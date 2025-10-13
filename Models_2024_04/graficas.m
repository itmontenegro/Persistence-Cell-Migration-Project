close all; clear all;

% Definir los valores de f_tax y H a iterar
f_tax_values = [1];  
H_values = [0.5,0.75,0.99];  

for f_tax = f_tax_values
    for H = H_values
        % Corregir formato de H y f_tax para coincidir con las carpetas
        H_str =  strrep(num2str(H), '.', '_');
        f_tax_str = strrep(num2str(f_tax),'.','_');
        % Cargar el archivo CSV
        data_totdist = readmatrix(['TotDist_alpha_0_25_f_tax_',f_tax_str,'_H_',H_str,'.csv']);
        % Valores reales de Dr
        Dr_real = [0, 0.1, 1, 10];       
        % Valores X ajustados visualmente (espaciados uniformemente)
        x_pos = 1:length(Dr_real);
        % Colores para cada grupo
        colors = lines(length(Dr_real)); % paleta de colores distintivos
        % Crear la figura para DISTANCIA A X_ORG  
        figure;
        % Dr 0
        scatter(x_pos(1), data_totdist(1,:), 20, [0.5 0.2 0.55], 'filled')
        hold on
        violinplot(x_pos(1), real(data_totdist(1,:)), 'EdgeColor' ,[0.5 0.2 0.55], 'FaceColor',[0.5 0.2 0.55])
        hold on
        % Dr 0.1
        scatter(x_pos(2), data_totdist(2,:), 20, [0.1 0.35 0.7], 'filled')
        hold on
        violinplot(x_pos(2), real(data_totdist(2,:)), 'EdgeColor' ,[0.1 0.35 0.7], 'FaceColor', [0.1 0.35 0.7])
        hold on
        % Dr 1
        scatter(x_pos(3), data_totdist(3,:), 20, [0.45 0.75 0.35], 'filled')
        hold on
        violinplot(x_pos(3), real(data_totdist(3,:)), 'EdgeColor' ,[0.45 0.75 0.35], 'FaceColor',[0.45 0.75 0.35])
        hold on
        % Dr 10
        scatter(x_pos(4), data_totdist(4,:), 20, [0.65 0.1 0.2], 'filled')
        hold on
        violinplot(x_pos(4), real(data_totdist(4,:)),'EdgeColor' ,[0.65 0.1 0.2], 'FaceColor',[0.65 0.1 0.2])
        hold off

        % Personalizar la gráfica
        xlim([0, length(Dr_real) + 1]);
        ylim([0, 200]);
        % xlabel('Dr');
        % ylabel('Total Distance'); % Etiqueta del eje Y en inglés
        axis on; box on;
        set(gca, 'LineWidth', 3);
        set(gca, 'XTickLabel', []);
        set(gca, 'YTickLabel', []);
        set(gca, 'TickLength', [0 0]);

        save_dir = 'Taxis_Analys/';
        filename = fullfile(save_dir, ['TotDist_alpha_0_25_f_', f_tax_str, '_H_', H_str, '_violin.png']);
        exportgraphics(gcf, filename, 'Resolution', 300,'Width',800,'Height',800);
        close(gcf);

       
    end
end