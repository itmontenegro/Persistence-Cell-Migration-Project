close all;clear all;clc;
%%
Dr_values = [0,0.1,1,10];
H_values = [0.5];

for Hval = 1:length(H_values)
    for Drval = 1:length(Dr_values)
        %% Generating based on values of H and Dr
            Dr = Dr_values(Drval); H = H_values(Hval); f_tax= 0.1;%f-tax para modificar

        % Loading 6 random datasets of the 10 sims
        permuted_indices = randperm(100);r = permuted_indices(1:6);
        Dr_str = strrep(num2str(Dr), '.', '_');
        H_str = strrep(num2str(H), '.', '_');
        f_tax_str = strrep(num2str(f_tax),'.','_');

        main_taxis = ['DATA_202555/alpha_0_f_tax_',f_tax_str,'/H_',H_str,'/Dr_',Dr_str]; %ruta directa

        name_1 = [main_taxis,'/Sim_',num2str(r(1)),'_Dr_',Dr_str,'_H_',H_str,'_f_',f_tax_str,'.csv'];
        name_2 = [main_taxis,'/Sim_',num2str(r(2)),'_Dr_',Dr_str,'_H_',H_str,'_f_',f_tax_str,'.csv'];
        name_3 = [main_taxis,'/Sim_',num2str(r(3)),'_Dr_',Dr_str,'_H_',H_str,'_f_',f_tax_str,'.csv'];
        name_4 = [main_taxis,'/Sim_',num2str(r(4)),'_Dr_',Dr_str,'_H_',H_str,'_f_',f_tax_str,'.csv'];
        name_5 = [main_taxis,'/Sim_',num2str(r(5)),'_Dr_',Dr_str,'_H_',H_str,'_f_',f_tax_str,'.csv'];
        name_6 = [main_taxis,'/Sim_',num2str(r(6)),'_Dr_',Dr_str,'_H_',H_str,'_f_',f_tax_str,'.csv'];
 
        %% Loading Data
        data_1 = readtable(name_1);data_2 = readtable(name_2);data_3 = readtable(name_3);data_4 = readtable(name_4);data_5 = readtable(name_5);data_6 = readtable(name_6);
        
        %% Plotting
        x_data_1 = data_1.x_array; y_data_1 = data_1.y_array;x_data_2 = data_2.x_array; y_data_2 = data_2.y_array;x_data_3 = data_3.x_array; y_data_3 = data_3.y_array;
        x_data_4 = data_4.x_array; y_data_4 = data_4.y_array;x_data_5 = data_5.x_array; y_data_5 = data_5.y_array;x_data_6 = data_6.x_array; y_data_6 = data_6.y_array;
        x_org = -100/sqrt(2);% EL x org queda constante

        % Create a colormap based on time
        colormapTime = jet(1000);  % You can choose a different colormap if you prefer
        ataxis = 0.5;  % Adjust as needed (transparency)
        % Plotting Trajectory with color-coded X
        figure;
        % Plot using a colormap with transparency
        scatter(x_data_1, y_data_1, 10, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_2, y_data_2, 10, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_3, y_data_3, 10, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_4, y_data_4, 10, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_5, y_data_5, 10, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_6, y_data_6, 10, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        % hold on; 
        scatter(x_org,x_org, 40, 'filled','k'); %Grosor de punto 40
        hold off;

        % EDITING
        % colorbar;  % Add a colorbar to indicate time
        colormap(colormapTime);  % Set the colormap
        % Optionally, you can set the colorbar label to represent time
        % c = colorbar;
        axis ([-100,100,-100,100]); 
        axis on;
        box on;
        set(gca,'LineWidth',3)
        set(gca, 'XTickLabel', []);
        set(gca, 'YTickLabel', []);
        set(gca,'TickLength',[0 0])

        % SAVING PNG FIGURE
        filename = ['Prueba_2/Trajectories/' ...
            ,'alpha_0_Dr_', Dr_str, '__H_',H_str,'_f_',f_tax_str,'.png']; % Configurar ruta de fuardado
        exportgraphics(gcf,filename,'Resolution',300,'Width',800,'Height',800);
        close(gcf);        

    end
end
