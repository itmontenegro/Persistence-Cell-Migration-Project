close all;clear all;clc;
%%
Dr_values = [1];
H_values = [0.5];
Fm_values = [1]

for Fval = 1:length(Fm_values)
    for Drval = 1:length(Dr_values)
        %% Generating based on values of H and Dr
        Dr = Dr_values(Drval);  H = 0.5;f_tax = 1; Fm = Fm_values(Fval); %f-tax para modificar
        % Loading 6 random datasets of the 10 sims
        permuted_indices = randperm(5);r = permuted_indices(1:5);
        Dr_str = strrep(num2str(Dr), '.', '_');
        H_str = strrep(num2str(H), '.', '_');
        % f_tax_str = strrep(num2str(f_tax),'.','_');
        Fm_str = strrep(num2str(Fm),'.','_');

        main_taxis = ['DATA_2025_TestVic/data__Dr_',Dr_str,'__Fm_', Fm_str]; %ruta directa

        name_1 = [main_taxis,'/Sim_',num2str(r(1)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_2 = [main_taxis,'/Sim_',num2str(r(2)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_3 = [main_taxis,'/Sim_',num2str(r(3)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_4 = [main_taxis,'/Sim_',num2str(r(4)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_5 = [main_taxis,'/Sim_',num2str(r(5)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        % name_6 = [main_taxis,'/Sim_',num2str(r(6)),'_Dr_',Dr_str,'_H_',H_str,'_f_',f_tax_str,'.csv'];
 
        %% Loading Data
        data_1 = readtable(name_1);data_2 = readtable(name_2);data_3 = readtable(name_3);data_4 = readtable(name_4);data_5 = readtable(name_5);
        % data_6 = readtable(name_6);
        
        %% Plotting
        x_data_1 = data_1.X; y_data_1 = data_1.Y;x_data_2 = data_2.X; y_data_2 = data_2.Y;x_data_3 = data_3.X; y_data_3 = data_3.Y;
        x_data_4 = data_4.X; y_data_4 = data_4.Y;x_data_5 = data_5.X; y_data_5 = data_5.Y;
        % x_data_6 = data_6.X; y_data_6 = data_6.Y;
        x_org = -100/sqrt(2);% EL x org queda constante

        % Create a colormap based on time
        colormapTime = autumn(1000);  % You can choose a different colormap if you prefer
        ataxis = 0.5;  % Adjust as needed (transparency)
        % Plotting Trajectory with color-coded X
        figure;
        % Plot using a colormap with transparency
        scatter(x_data_1, y_data_1, 50, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_2, y_data_2, 50, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_3, y_data_3, 50, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_4, y_data_4, 50, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        scatter(x_data_5, y_data_5, 50, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        hold on;
        % scatter(x_data_6, y_data_6, 50, 1:1000, 'filled', 'MarkerFaceAlpha', ataxis);
        % hold on; 
        % scatter(x_org,x_org, 60, 'filled','k'); %Grosor de punto 40
        hold off;

        % EDITING
        % colorbar;  % Add a colorbar to indicate time
        colormap(flipud(colormapTime));  % Set the colormap
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
        filename = ['Traj_2026_Vic/' ...
            ,'SIZE_50__Dr_', Dr_str, '__H_',H_str,'_fm_',Fm_str,'.png']; % Configurar ruta de guardado
        exportgraphics(gcf,filename,'Resolution',300,'Width',800,'Height',800);
        % close(gcf);    

        %% Movie
        % close all;
        % mkdir (['VideoFrames_2026_Vic/Dr_',Dr_str,'_H_',H_str])
        % for t = 1:10:length(x_data_1)
        %     colormapTime = autumn(t);
        %     alphaValue = 0.5;
        %     figure;
        %     scatter(x_data_1(1:t), y_data_1(1:t), 10,1:t, 'filled', 'MarkerFaceAlpha', alphaValue);
        %     hold on
        %     scatter(x_data_2(1:t), y_data_2(1:t), 10,1:t, 'filled', 'MarkerFaceAlpha', alphaValue);
        %     hold on
        %     scatter(x_data_3(1:t), y_data_3(1:t), 10,1:t, 'filled', 'MarkerFaceAlpha', alphaValue);
        %     hold on
        %     scatter(x_data_4(1:t), y_data_4(1:t), 10,1:t, 'filled', 'MarkerFaceAlpha', alphaValue);
        %     hold on
        %     scatter(x_data_5(1:t), y_data_5(1:t), 10,1:t, 'filled', 'MarkerFaceAlpha', alphaValue);
        %     % hold on
        %     % scatter(x_data_6(1:t), y_data_6(1:t), 10,1:t, 'filled', 'MarkerFaceAlpha', alphaValue);
        %     % hold on;
        %     % scatter(x_org,x_org, 60, 'filled','k'); %Grosor de punto 60
        %     hold off
        %     colormap(flipud(colormapTime))
        %     axis ([-100 100 -100 100]);
        %     % axis equal
        %     % Show the axes without tick values and labels
        %     axis on;
        %     box on;
        %     set(gca,'LineWidth',3)
        %     set(gca, 'XTickLabel', []);
        %     set(gca, 'YTickLabel', []);
        %     set(gca,'TickLength',[0 0])
        %     % SAVING PNG FIGURE
        %     filename = ['VideoFrames_2026_Vic/Dr_',Dr_str,'_H_',H_str,'/Dr_', Dr_str, '__H_',H_str,'__Frame_',num2str(t),'.png'];
        %     exportgraphics(gcf,filename,'Resolution',300,'Height',800,'Width',800);
        %     disp(["SAVED FRAME NUMBER: ", t])
        %     close(gcf);
        % end

    end
end