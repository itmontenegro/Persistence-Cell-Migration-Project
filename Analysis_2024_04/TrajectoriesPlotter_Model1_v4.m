close all;clear all;clc;

%%
Dr_values = [0,0.1,0.5,1,5,10,100];
H_values = [0.01,0.25,0.5,0.75,0.99];
% Dr_values=[0.1];
H_values=[0.5];
for Hval = 1:length(H_values)
    for Drval = 1:length(Dr_values)
        %% Generating based on values of H and Dr
        Dr = Dr_values(Drval); H = H_values(Hval); alpha = 0;
        % Loading 5 random datasets of the 50 sims
        permuted_indices = randperm(25);
        r = permuted_indices(1:10);
        Dr_str = strrep(num2str(Dr), '.', '_');
        H_str = strrep(num2str(H), '.', '_');
        alpha_str = strrep(num2str(alpha),'.','_');
        main_pre_alpha = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_',alpha_str,'/H_',H_str];
        name_1 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(1)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_2 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(2)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_3 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(3)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_4 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(4)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_5 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(5)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_6 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(6)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_7 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(7)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_8 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(8)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_9 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(9)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        name_10 = [main_pre_alpha,'/data__Alpha_',alpha_str,'__Dr_',Dr_str,'__H_',H_str,'/Sim_',num2str(r(10)),'__Dr_',Dr_str,'__H_',H_str,'.csv'];
        
        %% Loading Data
        data_1 = readtable(name_1);
        data_2 = readtable(name_2);
        data_3 = readtable(name_3);
        data_4 = readtable(name_4);
        data_5 = readtable(name_5);
        data_6 = readtable(name_6);
        data_7 = readtable(name_7);
        data_8 = readtable(name_8);
        data_9 = readtable(name_9);
        data_10 = readtable(name_10);
        
        %% Plotting
        x_data_1 = data_1.X; y_data_1 = data_1.Y;
        x_data_2 = data_2.X; y_data_2 = data_2.Y;
        x_data_3 = data_3.X; y_data_3 = data_3.Y;
        x_data_4 = data_4.X; y_data_4 = data_4.Y;
        x_data_5 = data_5.X; y_data_5 = data_5.Y;
        x_data_6 = data_6.X; y_data_6 = data_6.Y;
        x_data_7 = data_7.X; y_data_7 = data_7.Y;
        x_data_8 = data_8.X; y_data_8 = data_8.Y;
        x_data_9 = data_9.X; y_data_9 = data_9.Y;
        x_data_10 = data_10.X; y_data_10 = data_10.Y;
        
        % Create a colormap based on time
        colormapTime = jet(1000);  % You can choose a different colormap if you prefer
        alphaValue = 0.5;  % Adjust as needed (transparency)
        % Plotting Trajectory with color-coded x_array
        figure;
        axis ([-200,200,-200,200]);
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
        hold on;
        scatter(x_data_7, y_data_7, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold on;
        scatter(x_data_8, y_data_8, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold on;
        scatter(x_data_9, y_data_9, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold on;
        scatter(x_data_10, y_data_10, 10, 1:1000, 'filled', 'MarkerFaceAlpha', alphaValue);
        hold off;
        % EDITING
        % colorbar;  % Add a colorbar to indicate time
        colormap(colormapTime);  % Set the colormap
        % Optionally, you can set the colorbar label to represent time
        % c = colorbar;
        axis ([-150,150,-150,150]);
        axis on;
        box on;
        set(gca,'LineWidth',3)
        set(gca, 'XTickLabel', []);
        set(gca, 'YTickLabel', []);
        set(gca,'TickLength',[0 0])
        % SAVING PNG FIGURE
        filename = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_',alpha_str,'/Figures/' ...
            'Trajectories/Alpha_',alpha_str,'__Dr_', Dr_str, '__H_',H_str,'__10Traj.png'];
        exportgraphics(gcf,filename,'Resolution',300);
        close(gcf);
        

        %% 30x30 ZOOMED IN PLOT
        % SET UP FIGURE
        % Create a colormap based on time
        colormapTime = jet(1000);  % You can choose a different colormap if you prefer
        alphaValue = 0.5;  % Adjust as needed (transparency)
        % Plotting Trajectory with color-coded x_array
        figure;
        axis ([-3.5,3.5,-3.5,3.5]);
        % Plot using a colormap with transparency
        scatter(x_data_1, y_data_1, 10, 1:1000, 'b','filled');
        hold on;
        scatter(x_data_2, y_data_2, 10, 1:1000, 'r', 'filled');
        hold on;
        scatter(x_data_3, y_data_3, 10, 1:1000, 'c','filled');
        hold on;
        scatter(x_data_4, y_data_4, 10, 1:1000, 'm','filled');
        hold on;
        scatter(x_data_5, y_data_5, 10, 1:1000, 'k','filled');
        hold on;
        scatter(x_data_6, y_data_6, 10, [0.8 0.6 0.49]);
        hold on;
        scatter(x_data_7, y_data_7, 10, [0.4660 0.6740 0.1880],'filled');
        hold on;
        scatter(x_data_8, y_data_8, 10, [0.98 0.69 0.18],'filled');
        hold on;
        scatter(x_data_9, y_data_9, 10, [0 0.5 0.37],'filled');
        hold on;
        scatter(x_data_10, y_data_10, 10, [0.44 0.44 0.88],'filled');
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
        filename = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_',alpha_str,'/Figures/' ...
            'TrajectoriesZoomed/Alpha_',alpha_str,'__Dr_', Dr_str, '__H_',H_str,'__10Traj.png'];
        exportgraphics(gcf,filename,'Resolution',300);
        close(gcf);

        %% FOR ANGLE EVOLUTION
        % theta_data_1 = data_1.Theta;
        % norm_theta_1 = mod(theta_data_1, 2*pi);
        % theta_data_2 = data_2.Theta;
        % norm_theta_2 = mod(theta_data_2, 2*pi);
        % theta_data_3 = data_3.Theta;
        % norm_theta_3 = mod(theta_data_3, 2*pi);
        % time_vector = linspace(1,1000,height(data_1));
        % 
        % figure;
        % h_traj = plot(time_vector,theta_data_1, 'r-', ...
        %               time_vector,theta_data_2, 'b-', ...
        %               time_vector,theta_data_3, 'g-','LineWidth', 2);
        % % title('Angle Evolution');
        % % subtitle(['Dr = ', num2str(Dr), ' H = ', num2str(H), ' with Alpha = ', num2str(alpha)])
        % % xlabel('Time');
        % % ylabel('Angle Value (in radians)');
        % axis([0 1000 -10*pi 10*pi]);
        % set(gcf, 'PaperPosition', [0, 0, 900 / get(gcf, 'ScreenPixelsPerInch'), 900 / get(gcf, 'ScreenPixelsPerInch')]);
        % % EDITING
        % axis on;
        % set(gca,'LineWidth',3)
        % set(gca, 'XTickLabel', []);
        % set(gca, 'YTickLabel', []);
        % set(gca,'TickLength',[0 0])
        % % SAVING PNG FIGURE
        % filename = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_',alpha_str,'/Figures/' ...
        %     'AngleEvo/Alpha_',alpha_str,'__Dr_', Dr_str, '__H_',H_str,'__3AngleEvo.png'];
        % exportgraphics(gcf,filename,'Resolution',300);
        % close(gcf);
        % 
        % %% FOR dTheta Evolution
        % time_vector = linspace(1,1000,height(data_1));
        % dtheta_1 = zeros(1000,1); dtheta_2 = zeros(1000,1); dtheta_3 = zeros(1000,1);
        % for i = 2:height(data_1)
        %     dtheta_1(i,1) = theta_data_1(i) - theta_data_1(i-1);
        %     dtheta_2(i,1) = theta_data_2(i) - theta_data_2(i-1);
        %     dtheta_3(i,1) = theta_data_3(i) - theta_data_3(i-1);
        % end
        % figure;
        % h_traj = plot(time_vector,dtheta_1, 'r-', ...
        %               time_vector,dtheta_2, 'b-', ...
        %               time_vector,dtheta_3, 'g-','LineWidth', 1);
        % axis([0 1000 -pi pi]);
        % set(gcf, 'PaperPosition', [0, 0, 900 / get(gcf, 'ScreenPixelsPerInch'), 900 / get(gcf, 'ScreenPixelsPerInch')]);
        % axis on;
        % set(gca,'LineWidth',3)
        % set(gca, 'XTickLabel', []);
        % set(gca, 'YTickLabel', []);
        % set(gca,'TickLength',[0 0])
        % filename_angle = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_',alpha_str,'/Figures/' ...
        %     'dThetaEvo/Alpha_',alpha_str,'__Dr_', Dr_str, '__H_',H_str,'__3dThetaEvo.png'];
        % exportgraphics(gcf,filename_angle,'Resolution',300);
        % close(gcf);
        % 
        % 
        %% Movie
        % close all;
        % for t = 1:10:length(x_data_1)
        %     figure;
        %     scatter(x_data_1(t), y_data_1(t), 'r','filled');
        %     hold on; % Add this line to hold the current plot  
        %     plot(x_data_1(1:t),y_data_1(1:t),'r-','LineWidth', 2);
        %     scatter(x_data_2(t), y_data_2(t), 'b','filled');    
        %     plot(x_data_2(1:t),y_data_2(1:t),'b-','LineWidth', 2);
        %     scatter(x_data_3(t), y_data_3(t), 'g','filled');
        %     plot(x_data_3(1:t),y_data_3(1:t),'g-','LineWidth', 2);
        %     scatter(x_data_4(t), y_data_4(t), 'm','filled');
        %     plot(x_data_4(1:t),y_data_4(1:t),'m-','LineWidth', 2);
        %     scatter(x_data_5(t), y_data_5(t), 'k','filled');
        %     plot(x_data_5(1:t),y_data_5(1:t),'k-','LineWidth', 2);
        % 
        %     hold off; % Add this line to release the current plot        
        %     % title(['Dr = ', num2str(Dr), ' H = ', num2str(H), ' with Alpha = ', num2str(alpha)])
        %     axis ([-200,200,-200,200]);
        %     set(gcf, 'PaperPosition', [0, 0, 900 / get(gcf, 'ScreenPixelsPerInch'), 900 / get(gcf, 'ScreenPixelsPerInch')]);
        %     % Show the axes without tick values and labels
        %     axis on;
        %     set(gca, 'XTickLabel', []);
        %     set(gca, 'YTickLabel', []);
        %     % SAVING PNG FIGURE
        %     filename = ['Model1_H_cnst/Model1_2D_Batch200/Alpha_',alpha_str,'/Figures/' ...
        %         'VideoFrames/Dr_',Dr_str,'_H_',H_str,'/Alpha_',alpha_str,'__Dr_', Dr_str, '__H_',H_str,'__Frame_',num2str(t),'.png'];
        %     saveas(gcf, filename);
        %     close(gcf);
        % end
    end
end
