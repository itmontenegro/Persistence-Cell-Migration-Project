close all; clear all;
%% STEPSIZE MEAN, StdDev
H_values = [0.5,0.75,0.99];
Dr_values = [0,0.1,1,10];

mean_stepsize_matrix = zeros(length(Dr_values),200);
mean_angle_matrix = zeros(length(Dr_values),200);
mean_persfac_matrix = zeros(length(Dr_values),200);
mean_distance_matrix = zeros(length(Dr_values),200);
mean_displacement_matrix = zeros(length(Dr_values),200);
noise_matrix_x = zeros(length(Dr_values),200);
noise_matrix_y = zeros(length(Dr_values),200);

theta_matrix = zeros(length(Dr_values),200);
% For each H value
for Hval = 1:length(H_values)
    H_value = H_values(Hval);
    H_str = strrep(num2str(H_values(Hval)), '.', '_');
    mainFolderPath = ['../Model1_H_cnst/Model1_2D_Batch200_v5/Alpha_0_25/H_',H_str,'/'];  
    % For each Dr value in order of reading Dr_values
    for Drval = 1:length(Dr_values)  
        Dr_value = Dr_values(Drval);
        Dr_str = strrep(num2str(Dr_value), '.', '_');
        subfolder = ['/data__Alpha_0_25__Dr_',Dr_str,'__H_',H_str]
        subfolderPath = fullfile(mainFolderPath,subfolder);
        csvFiles = dir(fullfile(subfolderPath,'*.csv')); 
        % Array for each model that will have all values
        file_step_full = zeros(999,length(csvFiles));
        file_angle_full = zeros(999,length(csvFiles));
        file_persfac_full = zeros(1,length(csvFiles));
        file_distance_full = zeros(1,length(csvFiles));
        file_displacement_full = zeros(1,length(csvFiles));
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
                file_step_full(t,j) = stepsize;
                step_0 = sqrt((data_sim.X(t+1)-data_sim.X(t))^2 + (data_sim.Y(t+1)-data_sim.Y(t))^2);
                step_1 = sqrt((data_sim.X(t+2)-data_sim.X(t+1))^2 + (data_sim.Y(t+2)-data_sim.Y(t+1))^2);
                step_2 = sqrt((data_sim.X(t+2)-data_sim.X(t))^2 + (data_sim.Y(t+2)-data_sim.Y(t))^2);                
                file_angle_full(t, j) = pi - acos((step_0^2+step_1^2-step_2^2)/(2*step_0*step_1));
            end
            displacement = sqrt(data_sim.X(t)^2 + data_sim.Y(t)^2);
            pers_fac_sim = displacement/total_distance;
            file_persfac_full(1,j) = pers_fac_sim;
            file_distance_full(1,j) = total_distance;
            file_displacement_full(1,j) = displacement;
            noise_matrix_x(Drval,j) = data_sim.Xi_x(2);
            noise_matrix_y(Drval,j) = data_sim.Xi_y(2);
            theta_matrix(Drval,j) = data_sim.Theta(1);
        end   

        % LEAVE ANGLE IN DEGREES TO AVOID IMAGINARY PROBLEMS
        mean_stepsize_per_sim = mean(file_step_full);
        mean_angle_per_sim = mean(file_angle_full);
        mean_stepsize_matrix(Drval,:) = mean_stepsize_per_sim;
        mean_angle_matrix(Drval,:) = mean_angle_per_sim;
        mean_persfac_matrix(Drval,:) = file_persfac_full;
        mean_distance_matrix(Drval,:) = file_distance_full;
        mean_displacement_matrix(Drval,:) = file_displacement_full;
    end

    % Here you have matrixes for one H value with ALL Drs
    %% Violin for P.F.
    persfac = mean_persfac_matrix;
    x_pos = 1:length(Dr_values);
    figure;
    % Dr 0
    scatter(x_pos(1)+(rand(1,200)-0.5)*0.1, persfac(1,:), 20, [0.5 0.2 0.55], 'filled')
    hold on
    violinplot(x_pos(1), real(persfac(1,:)), 'EdgeColor' ,[0.5 0.2 0.55], 'FaceColor',[0.5 0.2 0.55])
    hold on
    % Dr 0.1
    scatter(x_pos(2)+(rand(1,200)-0.5)*0.1, persfac(2,:), 20, [0.1 0.35 0.7], 'filled')
    hold on
    violinplot(x_pos(2), real(persfac(2,:)), 'EdgeColor' ,[0.1 0.35 0.7], 'FaceColor', [0.1 0.35 0.7])
    hold on
    % Dr 1
    scatter(x_pos(3)+(rand(1,200)-0.5)*0.1, persfac(3,:), 20, [0.45 0.75 0.35], 'filled')
    hold on
    violinplot(x_pos(3), real(persfac(3,:)), 'EdgeColor' ,[0.45 0.75 0.35], 'FaceColor',[0.45 0.75 0.35])
    hold on
    % Dr 10
    scatter(x_pos(4)+(rand(1,200)-0.5)*0.1, persfac(4,:), 20, [0.65 0.1 0.2], 'filled')
    hold on
    violinplot(x_pos(4), real(persfac(4,:)),'EdgeColor' ,[0.65 0.1 0.2], 'FaceColor',[0.65 0.1 0.2])
    hold off

    ylim([0 1])
    axis on;
    box on;
    set(gca,'LineWidth',3)
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca,'TickLength',[0 0])
    % Saving
    filename = ['../FIGURES_2025/' ...
        'Alpha_0_25__H_',H_str,'__PF_Violin.png'];
    exportgraphics(gcf,filename,'Resolution',300);
    close(gcf);

    %% Violin for INCREMENT
    increment = mean_stepsize_matrix;
    x_pos = 1:length(Dr_values);
    figure;
    % Dr 0
    scatter(x_pos(1)+(rand(1,200)-0.5)*0.1, increment(1,:), 20, [0.5 0.2 0.55], 'filled')
    hold on
    violinplot(x_pos(1), real(increment(1,:)), 'EdgeColor' ,[0.5 0.2 0.55], 'FaceColor',[0.5 0.2 0.55])
    hold on
    % Dr 0.1
    scatter(x_pos(2)+(rand(1,200)-0.5)*0.1, increment(2,:), 20, [0.1 0.35 0.7], 'filled')
    hold on
    violinplot(x_pos(2), real(increment(2,:)), 'EdgeColor' ,[0.1 0.35 0.7], 'FaceColor', [0.1 0.35 0.7])
    hold on
    % Dr 1
    scatter(x_pos(3)+(rand(1,200)-0.5)*0.1, increment(3,:), 20, [0.45 0.75 0.35], 'filled')
    hold on
    violinplot(x_pos(3), real(increment(3,:)), 'EdgeColor' ,[0.45 0.75 0.35], 'FaceColor',[0.45 0.75 0.35])
    hold on
    % Dr 10
    scatter(x_pos(4)+(rand(1,200)-0.5)*0.1, increment(4,:), 20, [0.65 0.1 0.2], 'filled')
    hold on
    violinplot(x_pos(4), real(increment(4,:)),'EdgeColor' ,[0.65 0.1 0.2], 'FaceColor',[0.65 0.1 0.2])
    hold off

    ylim([0 0.2])
    axis on;
    box on;
    set(gca,'LineWidth',3)
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca,'TickLength',[0 0])
    % Saving
    filename = ['../FIGURES_2025/' ...
        'Alpha_0_25__H_',H_str,'__Increment_Violin.png'];
    exportgraphics(gcf,filename,'Resolution',300);
    close(gcf);

    %% Violin for RELATIVE ANGLE
    relat_angle = real(mean_angle_matrix);
    x_pos = 1:length(Dr_values);
    figure;
    % Dr 0
    scatter(x_pos(1)+(rand(1,200)-0.5)*0.1, relat_angle(1,:), 20, [0.5 0.2 0.55], 'filled')
    hold on
    violinplot(x_pos(1), real(relat_angle(1,:)), 'EdgeColor' ,[0.5 0.2 0.55], 'FaceColor',[0.5 0.2 0.55])
    hold on
    % Dr 0.1
    scatter(x_pos(2)+(rand(1,200)-0.5)*0.1, relat_angle(2,:), 20, [0.1 0.35 0.7], 'filled')
    hold on
    violinplot(x_pos(2), real(relat_angle(2,:)), 'EdgeColor' ,[0.1 0.35 0.7], 'FaceColor', [0.1 0.35 0.7])
    hold on
    % Dr 1
    scatter(x_pos(3)+(rand(1,200)-0.5)*0.1, relat_angle(3,:), 20, [0.45 0.75 0.35], 'filled')
    hold on
    violinplot(x_pos(3), real(relat_angle(3,:)), 'EdgeColor' ,[0.45 0.75 0.35], 'FaceColor',[0.45 0.75 0.35])
    hold on
    % Dr 10
    scatter(x_pos(4)+(rand(1,200)-0.5)*0.1, relat_angle(4,:), 20, [0.65 0.1 0.2], 'filled')
    hold on
    violinplot(x_pos(4), real(relat_angle(4,:)),'EdgeColor' ,[0.65 0.1 0.2], 'FaceColor',[0.65 0.1 0.2])
    hold off

    ylim([0 pi/2])
    axis on;
    box on;
    set(gca,'LineWidth',3)
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca,'TickLength',[0 0])
    % Saving
    filename = ['../FIGURES_2025/' ...
        'Alpha_0_25__H_',H_str,'__RelatAngle_Violin.png'];
    exportgraphics(gcf,filename,'Resolution',300);
    close(gcf);

    %% Violin for TOTAL DISTANCE
    tot_dist = mean_distance_matrix;
    x_pos = 1:length(Dr_values);
    figure;
    % Dr 0
    scatter(x_pos(1)+(rand(1,200)-0.5)*0.1, tot_dist(1,:), 20, [0.5 0.2 0.55], 'filled')
    hold on
    violinplot(x_pos(1), real(tot_dist(1,:)), 'EdgeColor' ,[0.5 0.2 0.55], 'FaceColor',[0.5 0.2 0.55])
    hold on
    % Dr 0.1
    scatter(x_pos(2)+(rand(1,200)-0.5)*0.1, tot_dist(2,:), 20, [0.1 0.35 0.7], 'filled')
    hold on
    violinplot(x_pos(2), real(tot_dist(2,:)), 'EdgeColor' ,[0.1 0.35 0.7], 'FaceColor', [0.1 0.35 0.7])
    hold on
    % Dr 1
    scatter(x_pos(3)+(rand(1,200)-0.5)*0.1, tot_dist(3,:), 20, [0.45 0.75 0.35], 'filled')
    hold on
    violinplot(x_pos(3), real(tot_dist(3,:)), 'EdgeColor' ,[0.45 0.75 0.35], 'FaceColor',[0.45 0.75 0.35])
    hold on
    % Dr 10
    scatter(x_pos(4)+(rand(1,200)-0.5)*0.1, tot_dist(4,:), 20, [0.65 0.1 0.2], 'filled')
    hold on
    violinplot(x_pos(4), real(tot_dist(4,:)),'EdgeColor' ,[0.65 0.1 0.2], 'FaceColor',[0.65 0.1 0.2])
    hold off

    ylim([0 200])
    axis on;
    box on;
    set(gca,'LineWidth',3)
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca,'TickLength',[0 0])
    % Saving
    filename = ['../FIGURES_2025/' ...
        'Alpha_0_25__H_',H_str,'__TotalDistance_Violin.png'];
    exportgraphics(gcf,filename,'Resolution',300);
    close(gcf);

    %% Violin for DISPLACEMENT
    sims_displacement = mean_displacement_matrix;
    x_pos = 1:length(Dr_values);
    figure;
    % Dr 0
    scatter(x_pos(1)+(rand(1,200)-0.5)*0.1, sims_displacement(1,:), 20, [0.5 0.2 0.55], 'filled')
    hold on
    violinplot(x_pos(1), real(sims_displacement(1,:)), 'EdgeColor' ,[0.5 0.2 0.55], 'FaceColor',[0.5 0.2 0.55])
    hold on
    % Dr 0.1
    scatter(x_pos(2)+(rand(1,200)-0.5)*0.1, sims_displacement(2,:), 20, [0.1 0.35 0.7], 'filled')
    hold on
    violinplot(x_pos(2), real(sims_displacement(2,:)), 'EdgeColor' ,[0.1 0.35 0.7], 'FaceColor', [0.1 0.35 0.7])
    hold on
    % Dr 1
    scatter(x_pos(3)+(rand(1,200)-0.5)*0.1, sims_displacement(3,:), 20, [0.45 0.75 0.35], 'filled')
    hold on
    violinplot(x_pos(3), real(sims_displacement(3,:)), 'EdgeColor' ,[0.45 0.75 0.35], 'FaceColor',[0.45 0.75 0.35])
    hold on
    % Dr 10
    scatter(x_pos(4)+(rand(1,200)-0.5)*0.1, sims_displacement(4,:), 20, [0.65 0.1 0.2], 'filled')
    hold on
    violinplot(x_pos(4), real(sims_displacement(4,:)),'EdgeColor' ,[0.65 0.1 0.2], 'FaceColor',[0.65 0.1 0.2])
    hold off

    ylim([0 200])
    axis on;
    box on;
    set(gca,'LineWidth',3)
    set(gca, 'XTickLabel', []);
    set(gca, 'YTickLabel', []);
    set(gca,'TickLength',[0 0])
    % Saving
    filename = ['../FIGURES_2025/' ...
        'Alpha_0_25__H_',H_str,'__Displacement_Violin.png'];
    exportgraphics(gcf,filename,'Resolution',300);
    close(gcf);

    %% Scatter for INCREMENT vs PF
    % figure;
    % % Dr 0
    % scatter(increment(1,:), persfac(1,:), 20, [0.5 0.2 0.55], 'filled')
    % hold on
    % % Dr 0.1
    % scatter(increment(2,:), persfac(2,:), 20, [0.1 0.35 0.7], 'filled')
    % hold on
    % % Dr 1
    % scatter(increment(4,:), persfac(4,:), 20, [0.45 0.75 0.35], 'filled')
    % hold on
    % % Dr 10
    % scatter(increment(6,:), persfac(6,:), 20, [0.65 0.1 0.2], 'filled')
    % hold off
    % xlim([0 0.2])
    % ylim([0 1])
    % axis on;
    % box on;
    % set(gca,'LineWidth',3)
    % set(gca, 'XTickLabel', []);
    % set(gca, 'YTickLabel', []);
    % set(gca,'TickLength',[0 0])
    % % Saving
    % filename = ['../FIGURES_2025/' ...
    %     'Alpha_0_25__H_',H_str,'__Increment_vs_PF.png'];
    % exportgraphics(gcf,filename,'Resolution',300);
    % close(gcf);

    %% Scatter for RELATIVE ANGLE vs PF
    % figure;
    % % Dr 0
    % scatter(relat_angle(1,:), persfac(1,:), 20, [0.5 0.2 0.55], 'filled')
    % hold on
    % % Dr 0.1
    % scatter(relat_angle(2,:), persfac(2,:), 20, [0.1 0.35 0.7], 'filled')
    % hold on
    % % Dr 1
    % scatter(relat_angle(4,:), persfac(4,:), 20, [0.45 0.75 0.35], 'filled')
    % hold on
    % % Dr 10
    % scatter(relat_angle(6,:), persfac(6,:), 20, [0.65 0.1 0.2], 'filled')
    % hold off
    % 
    % xlim([0 pi/2])
    % ylim([0 1])
    % axis on;
    % box on;
    % set(gca,'LineWidth',3)
    % set(gca, 'XTickLabel', []);
    % set(gca, 'YTickLabel', []);
    % set(gca,'TickLength',[0 0])
    % % Saving
    % filename = ['../FIGURES_2025/' ...
    %     'Alpha_0_25__H_',H_str,'__RelatAngle_vs_PF.png'];
    % exportgraphics(gcf,filename,'Resolution',300);
    % close(gcf);

    %% Scatter for INCREMENT vs RELATIVE ANGLE
    % increment = mean_stepsize_matrix;
    % relat_angle = real(mean_angle_matrix);
    % figure;
    % % Dr 0
    % scatter(increment(1,:), real(relat_angle(1,:)), 20, [0.5 0.2 0.55], 'filled')
    % hold on
    % % Dr 0.1
    % scatter(increment(2,:), real(relat_angle(2,:)), 20, [0.1 0.35 0.7], 'filled')
    % hold on
    % % Dr 1
    % scatter(increment(4,:), real(relat_angle(4,:)), 20, [0.45 0.75 0.35], 'filled')
    % hold on
    % % Dr 10
    % scatter(increment(6,:), real(relat_angle(6,:)), 20, [0.65 0.1 0.2], 'filled')
    % hold off
    % 
    % xlim([0 0.2])
    % ylim([0 pi/2])
    % axis on;
    % box on;
    % set(gca,'LineWidth',3)
    % set(gca, 'XTickLabel', []);
    % set(gca, 'YTickLabel', []);
    % set(gca,'TickLength',[0 0])
    % % Saving
    % filename = ['Alpha_0_25__H_',H_str,'__Increment_vs_RelatAngle.png'];
    % exportgraphics(gcf,filename,'Resolution',300);
    % close(gcf);
end

