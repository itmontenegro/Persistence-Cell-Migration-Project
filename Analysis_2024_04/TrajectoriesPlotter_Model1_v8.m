close all;clear all;clc;
%%
Dr_values = [1];
H_values = [0.99];
for Hval = 1:length(H_values)
    for Drval = 1:length(Dr_values)
        %% Generating based on values of H and Dr
        Dr = Dr_values(Drval); H = H_values(Hval); alpha = 0.25;
        % Loading 5 random datasets of the 50 sims
        permuted_indices = randperm(200);r = permuted_indices(1:10);
        Dr_str = strrep(num2str(Dr), '.', '_');
        H_str = strrep(num2str(H), '.', '_');
        alpha_str = strrep(num2str(alpha),'.','_');
        main_pre_alpha = ['../Model1_H_cnst/Model1_2D_Batch200_v5/Alpha_',alpha_str,'/H_',H_str];
        name_1 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(1)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_2 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(2)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_3 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(3)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_4 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(4)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_5 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(5)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_6 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(6)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
 
        %% Loading Data
        data_1 = readtable(name_1);data_2 = readtable(name_2);data_3 = readtable(name_3);data_4 = readtable(name_4);data_5 = readtable(name_5);data_6 = readtable(name_6);
        
        %% Plotting
        x_data_1 = data_1.X; y_data_1 = data_1.Y;x_data_2 = data_2.X; y_data_2 = data_2.Y;x_data_3 = data_3.X; y_data_3 = data_3.Y;
        x_data_4 = data_4.X; y_data_4 = data_4.Y;x_data_5 = data_5.X; y_data_5 = data_5.Y;x_data_6 = data_6.X; y_data_6 = data_6.Y;

        % Create a colormap based on time
        colormapTime = jet(1000);  % You can choose a different colormap if you prefer
        alphaValue = 0.5;  % Adjust as needed (transparency)
        % Plotting Trajectory with color-coded x_array
        figure;
        % Plot using a colormap with transparency
        scatter(x_data_1, y_data_1, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold on;
        scatter(x_data_2, y_data_2, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold on;
        scatter(x_data_3, y_data_3, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold on;
        scatter(x_data_4, y_data_4, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold on;
        scatter(x_data_5, y_data_5, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold on;
        scatter(x_data_6, y_data_6, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
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
        filename = ['../Model1_H_cnst/FIGURES_122024/Trajectories/' ...
            'Alpha_',alpha_str,'__Dr_', Dr_str, '__H_',H_str,'__v2.png'];
        % exportgraphics(gcf,filename,'Resolution',300,'Width',800,'Height',800);
        % close(gcf);        

        %% 30x30 ZOOMED IN PLOT
        % SET UP FIGURE
        % Plotting Trajectory with color-coded x_array
        figure;
        axis ([-3.5,3.5,-3.5,3.5]);
        % Plot using a colormap with transparency
        scatter(x_data_3, y_data_3, 10, 1:1000, 'c','filled');
        hold on;
        plot(x_data_3,y_data_3,'c','LineWidth',1);
        hold on;
        scatter(x_data_4, y_data_4, 10, 1:1000, 'm','filled');
        hold on;
        plot(x_data_4,y_data_4,'m','LineWidth',1);
        hold on;
        scatter(x_data_5, y_data_5, 10, 1:1000, 'k','filled');
        hold on;
        plot(x_data_5,y_data_5,'k','LineWidth',1);
        hold on;
        scatter(x_data_6, y_data_6, 10, [0.4660 0.6740 0.1880],'filled');
        hold on;
        plot(x_data_6,y_data_6,'Color',[0.4660 0.6740 0.1880],'LineWidth',1);
        hold on;
        scatter(x_data_2, y_data_2, 10, [0.98 0.69 0.18],'filled');
        hold on;
        plot(x_data_2,y_data_2,'Color',[0.98 0.69 0.18],'LineWidth',1);
        hold on;
        scatter(x_data_1, y_data_1, 10, [0.44 0.44 0.88],'filled');
        hold on;
        plot(x_data_1,y_data_1,'Color',[0.44 0.44 0.88],'LineWidth',1);
        hold off;

        % EDITING
        axis ([-3.5,3.5,-3.5,3.5]);
        axis on;
        box on;
        set(gca,'LineWidth',3)
        set(gca, 'XTickLabel', []);
        set(gca, 'YTickLabel', []);
        set(gca,'TickLength',[0 0])
        % SAVING PNG FIGURE
        filename = ['../Model1_H_cnst/FIGURES_122024/' ...
            'TrajectoriesZoomed/Alpha_',alpha_str,'__Dr_', Dr_str, '__H_',H_str,'__10Traj.png'];
        % exportgraphics(gcf,filename,'Resolution',300,'Width',800,'Height',800);
        % close(gcf);
        
        %% Movie
        % close all;
        % for t = 1:10:length(x_data_1)
        %     figure;
        % 
        %     scatter(x_data_1(1:t), y_data_1(1:t),10,[0.5 0.2 0.55],'filled');
        %     hold on; % Add this line to hold the current plot  
        %     plot(x_data_1(1:t),y_data_1(1:t),'Color',[0.5 0.2 0.55],'LineWidth', 2);
        %     % scatter(x_data_2(1:t), y_data_2(1:t),10,[0.45 0.75 0.35],'filled');
        %     % plot(x_data_2(1:t),y_data_2(1:t),'Color',[0.45 0.75 0.35],'LineWidth', 2);
        %     scatter(x_data_3(1:t), y_data_3(1:t),10,[0.1 0.6 1],'filled');
        %     plot(x_data_3(1:t),y_data_3(1:t),'Color',[0.1 0.6 1],'LineWidth', 2);
        %     % scatter(x_data_4(1:t), y_data_4(1:t),15,[0.1 0.35 0.7],'filled');
        %     % plot(x_data_4(1:t),y_data_4(1:t),'Color',[0.1 0.35 0.7],'LineWidth', 2);
        %     scatter(x_data_5(1:t), y_data_5(1:t),10,[1 0.4 0.1],'filled');
        %     plot(x_data_5(1:t),y_data_5(1:t),'Color',[1 0.4 0.1],'LineWidth', 2);
        %     % scatter(x_data_6(1:t), y_data_6(1:t),15,[0.65 0.1 0.2],'filled');
        %     % plot(x_data_6(1:t),y_data_6(1:t),'Color',[0.65 0.1 0.2],'LineWidth', 2);     
        %     hold off; % Add this line to release the current plot        
        %     % title(['Dr = ', num2str(Dr), ' H = ', num2str(H), ' with Alpha = ', num2str(alpha)])
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
        %     filename = ['../Model1_H_cnst/' ...
        %         'VideoFrames_Multi/Dr_',Dr_str,'_H_',H_str,'/Alpha_',alpha_str,'__Dr_', Dr_str, '__H_',H_str,'__Frame_',num2str(t),'.png'];
        %     exportgraphics(gcf,filename,'Resolution',150);
        %     close(gcf);
        % end
    end
end
