close all; clear all;
%% STEPSIZE MEAN, StdDev
H_values = [0.5];
Dr_values = [10];

mean_stepsize_matrix = zeros(length(Dr_values),200);
mean_angle_matrix = zeros(length(Dr_values),200);
mean_persfac_matrix = zeros(length(Dr_values),200);

noise_matrix_x = zeros(length(Dr_values),200);
noise_matrix_y = zeros(length(Dr_values),200);

% For each H value
for Hval = 1:length(H_values)
    H_value = H_values(Hval);
    H_str = strrep(num2str(H_values(Hval)), '.', '_');
    mainFolderPath = ['../Model1_H_cnst/Model1_2D_Batch200_v5/Alpha_0/H_',H_str,'/'];  
    % For each Dr value in order of reading Dr_values
    for Drval = 1:length(Dr_values)  
        Dr_value = Dr_values(Drval);
        Dr_str = strrep(num2str(Dr_value), '.', '_');
        subfolder = ['/data__Alpha_0__Dr_',Dr_str,'__H_',H_str]
        subfolderPath = fullfile(mainFolderPath,subfolder);
        csvFiles = dir(fullfile(subfolderPath,'*.csv')); 
        % Array for each model that will have all values
        % file_step_full = zeros(999,length(csvFiles));
        % file_angle_full = zeros(999,length(csvFiles));
        file_persfac_full = zeros(1,length(csvFiles));
        % For each csv file        
        for j = 1:length(csvFiles)
            csvFileName = csvFiles(j).name;        
            csvFilePath =fullfile(subfolderPath,csvFileName);
            data_sim = readtable(csvFilePath);
            % For each timestep
            total_distance = 0;
            for t = 1:(height(data_sim)-2)
                diff_X = (data_sim.X(t+1) - data_sim.X(t))^2;
                diff_Y = (data_sim.Y(t+1) - data_sim.Y(t))^2;
                stepsize = sqrt(diff_X + diff_Y);
                total_distance = total_distance + stepsize;
                % file_step_full(t,j) = stepsize;
                step_0 = sqrt((data_sim.X(t+1)-data_sim.X(t))^2 + (data_sim.Y(t+1)-data_sim.Y(t))^2);
                step_1 = sqrt((data_sim.X(t+2)-data_sim.X(t+1))^2 + (data_sim.Y(t+2)-data_sim.Y(t+1))^2);
                step_2 = sqrt((data_sim.X(t+2)-data_sim.X(t))^2 + (data_sim.Y(t+2)-data_sim.Y(t))^2);                
                % file_angle_full(t, j) = pi - acos((step_0^2+step_1^2-step_2^2)/(2*step_0*step_1));
            end
            displacement = sqrt(data_sim.X(t)^2 + data_sim.Y(t)^2);
            pers_fac_sim = displacement/total_distance;
            file_persfac_full(1,j) = pers_fac_sim;
            noise_matrix_x(Drval,j) = data_sim.Xi_x(2);
            noise_matrix_y(Drval,j) = data_sim.Xi_y(2);
        end 
        
        % Scatter for Persistence vs initial Conditions
        for sim = 1:200
            noise_norm = sqrt(noise_matrix_x(sim)^2 + noise_matrix_y(sim)^2);
            scatter(noise_norm,file_persfac_full(sim),20,[0.65 0.1 0.2],'filled','DisplayName',['H=',num2str(H_value),' Dr=',num2str(Dr_value)]);
            axis ([0,0.3,0,1])
            axis on;
            box on;
            set(gca,'LineWidth',3)
            set(gca, 'XTickLabel', []);
            set(gca, 'YTickLabel', []);
            set(gca,'TickLength',[0 0])
            % Saving
            filename = ['../Model1_H_cnst/FIGURES_082024/' ...
                'Scatter_PF_IC/Alpha_0_Dr_', Dr_str, '__H_',H_str,'__PF_IC.png'];            
            % close(gcf);
            hold on;
        end
        hold off;
        exportgraphics(gcf,filename,'Resolution',300);

        % LEAVE ANGLE IN DEGREES TO AVOID IMAGINARY PROBLEMS
        % file_angle_full = rad2deg(file_angle_full);
        % 
        % mean_stepsize_per_sim = mean(file_step_full);
        % mean_angle_per_sim = mean(file_angle_full);
        % 
        % mean_stepsize_matrix(Drval,:) = mean_stepsize_per_sim;
        % mean_angle_matrix(Drval,:) = mean_angle_per_sim;
        % mean_persfac_matrix(Drval,:) = file_persfac_full;
    end

end

%% STEPSIZE VS PERSFAC

% step = mean_stepsize_matrix;
% angle = mean_angle_matrix;
% persfac = mean_persfac_matrix;
% figure;
% % Dr 0
% scatter(step(1,:), persfac(1,:), 20, [0.5 0.2 0.55], 'filled')
% hold on
% % Dr 0.1
% scatter(step(2,:), persfac(2,:), 20, [0.1 0.35 0.7], 'filled')
% hold on
% % Dr 0.5
% scatter(step(3,:), persfac(3,:), 20, [0.1 0.6 1], 'filled')
% hold on
% % Dr 1
% scatter(step(4,:), persfac(4,:), 20, [0.45 0.75 0.35], 'filled')
% hold on
% % Dr 5
% scatter(step(5,:), persfac(5,:), 20, [1 0.4 0.1], 'filled')
% hold on
% % Dr 10
% scatter(step(6,:), persfac(6,:), 20, [0.65 0.1 0.2], 'filled')
% hold off
% 
% % title(['\alpha = 0 and H = ', num2str(H_value)])
% % xlabel('Stepsize')
% % ylabel('Relative Angle')
% axis([0 0.2 0 1])
% axis on;
% box on;
% set(gca,'LineWidth',3)
% set(gca, 'XTickLabel', []);
% set(gca, 'YTickLabel', []);
% set(gca,'TickLength',[0 0])
% 
% %% ANGLE VS PERSFAC
% figure;
% % Dr 0
% scatter(angle(1,:), persfac(1,:), 20, [0.5 0.2 0.55], 'filled')
% hold on
% % Dr 0.1
% scatter(angle(2,:), persfac(2,:), 20, [0.1 0.35 0.7], 'filled')
% hold on
% % Dr 0.5
% scatter(angle(3,:), persfac(3,:), 20, [0.1 0.6 1], 'filled')
% hold on
% % Dr 1
% scatter(angle(4,:), persfac(4,:), 20, [0.45 0.75 0.35], 'filled')
% hold on
% % Dr 5
% scatter(angle(5,:), persfac(5,:), 20, [1 0.4 0.1], 'filled')
% hold on
% % Dr 10
% scatter(angle(6,:), persfac(6,:), 20, [0.65 0.1 0.2], 'filled')
% hold off
% 
% % title(['\alpha = 0 and H = ', num2str(H_value)])
% % xlabel('Stepsize')
% % ylabel('Relative Angle')
% axis([0 90 0 1])
% axis on;
% box on;
% set(gca,'LineWidth',3)
% set(gca, 'XTickLabel', []);
% set(gca, 'YTickLabel', []);
% set(gca,'TickLength',[0 0])
% 
% 












