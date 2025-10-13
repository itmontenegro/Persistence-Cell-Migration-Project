close all; clear all;
%% STEPSIZE MEAN, StdDev
H_values = [0.5];
Dr_values = [0,0.1,1,10];
f_tax = 1;
% For each H value
for Hval = 1:length(H_values)
    % Matric for the MEAN VALUE OF EACH SIMULATION
    mean_stepsize_matrix = zeros(length(Dr_values),100);
    mean_angle_matrix = zeros(length(Dr_values),100);
    mean_persfac_matrix = zeros(length(Dr_values),100);
    mean_total_distance = zeros(length(Dr_values),100);
    
    noise_matrix_x = zeros(length(Dr_values),100);
    noise_matrix_y = zeros(length(Dr_values),100);
    
    theta_matrix = zeros(length(Dr_values),100);
    theta_org_matrix = zeros(length(Dr_values),100);
    mean_dist_to_org = zeros(length(Dr_values),100);

    H_value = H_values(Hval);
    H_str = strrep(num2str(H_values(Hval)), '.', '_');
    f_tax_str = strrep(num2str(f_tax), '.', '_');
    mainFolderPath = ['DATA_202555/alpha_0_f_tax_',f_tax_str,'/H_',H_str,'/']
    % For each Dr value in order of reading Dr_values
    for Drval = 1:length(Dr_values)  
        Dr_value = Dr_values(Drval);
        Dr_str = strrep(num2str(Dr_value), '.', '_');
        subfolder = ['/Dr_',Dr_str];
        subfolderPath = fullfile(mainFolderPath,subfolder);
        csvFiles = dir(fullfile(subfolderPath,'*.csv')); 
        % Array for each model that will have all values
        file_step_full = zeros(999,length(csvFiles));
        file_angle_full = zeros(999,length(csvFiles));
        file_persfac_full = zeros(1,length(csvFiles));
        total_distance_full = zeros(1,length(csvFiles));
        dist_to_org_full = zeros(1,length(csvFiles));
        % For each csv file        
        for j = 1:length(csvFiles)  
            csvFileName = csvFiles(j).name;        
            csvFilePath =fullfile(subfolderPath,csvFileName);
            data_sim = readtable(csvFilePath);
            % For each timestep
            total_distance = 0;
            for t = 1:(height(data_sim)-2)
                diff_X = (data_sim.x_array(t+1) - data_sim.x_array(t))^2;
                diff_Y = (data_sim.y_array(t+1) - data_sim.y_array(t))^2;
                stepsize = sqrt(diff_X + diff_Y);
                total_distance = total_distance + stepsize;
                file_step_full(t,j) = stepsize;
                step_0 = sqrt((data_sim.x_array(t+1)-data_sim.x_array(t))^2 + (data_sim.y_array(t+1)-data_sim.y_array(t))^2);
                step_1 = sqrt((data_sim.x_array(t+2)-data_sim.x_array(t+1))^2 + (data_sim.y_array(t+2)-data_sim.y_array(t+1))^2);
                step_2 = sqrt((data_sim.x_array(t+2)-data_sim.x_array(t))^2 + (data_sim.y_array(t+2)-data_sim.y_array(t))^2);                
                file_angle_full(t, j) = pi - acos((step_0^2+step_1^2-step_2^2)/(2*step_0*step_1));
            end
            pos_org = -100/sqrt(2); % Same coordinate for x and y
            displacement = sqrt(data_sim.x_array(t)^2 + data_sim.y_array(t)^2);
            pers_fac_sim = displacement/total_distance;
            file_persfac_full(1,j) = pers_fac_sim;
            total_distance_full(1,j) = total_distance;
            noise_matrix_x(Drval,j) = data_sim.xi_x_array(2);
            noise_matrix_y(Drval,j) = data_sim.xi_y_array(2);
            theta_matrix(Drval,j) = data_sim.theta_array(1);
            theta_org_matrix(Drval,j) = data_sim.theta_org(t); % Final angle cell v/s x_org
            dist_to_org_x = (data_sim.x_array(t) - pos_org)^2;
            dist_to_org_y = (data_sim.y_array(t) - pos_org)^2;
            final_dist_org = sqrt(dist_to_org_x + dist_to_org_y);
            dist_to_org_full(1,j) = final_dist_org; % Distance to X_org FOR ALL SIMS
        end
        
        % END WITH A MATRIX FOR EACH H VALUE ALL DRs --> MEAN VALUES
        mean_stepsize_per_sim = mean(file_step_full);
        mean_angle_per_sim = mean(file_angle_full);
        mean_stepsize_matrix(Drval,:) = mean_stepsize_per_sim;
        mean_angle_matrix(Drval,:) = mean_angle_per_sim;
        mean_persfac_matrix(Drval,:) = file_persfac_full;
        mean_total_distance(Drval,:) = total_distance_full;
        mean_dist_to_org(Drval,:) = dist_to_org_full;
    end
    %% SCATTER PLOTS FOR TAXIS

    % Distance to X_org vs P.F.
    total_dist = mean_total_distance;
    dist_to_org = real(mean_dist_to_org);
    persfac = mean_persfac_matrix;

    writematrix(total_dist,['TotDist_alpha_0_f_tax_',f_tax_str,'_H_',H_str,'.csv']);
    writematrix(dist_to_org,['DistToOrg_alpha_0_f_tax_',f_tax_str,'_H_',H_str,'.csv']);
    
    figure;
    % Dr 0
    scatter(dist_to_org(1,:), persfac(1,:), 20, [0.5 0.2 0.55], 'filled')
    hold on
    % Dr 0.1
    scatter(dist_to_org(2,:), persfac(2,:), 20, [0.1 0.35 0.7], 'filled')
    hold on
    % Dr 1
    scatter(dist_to_org(3,:), persfac(3,:), 20, [0.45 0.75 0.35], 'filled')
    hold on
    % Dr 10
    scatter(dist_to_org(4,:), persfac(4,:), 20, [0.65 0.1 0.2], 'filled')
    hold off
    axis([0 200 0 1])
    axis on;box on;set(gca,'LineWidth',3);set(gca, 'XTickLabel', []);set(gca, 'YTickLabel', []);set(gca,'TickLength',[0 0])
    % SAVING PNG FIGURE
    filename = ['Taxis_Analys/alpha_0_H_',H_str,'_f_',f_tax_str,'_DistORG_PF.png'];
    exportgraphics(gcf,filename,'Resolution',300,'Width',800,'Height',800);
    close(gcf);   

    %% Total Distance vs P.F.
    figure;
    % Dr 0
    scatter(total_dist(1,:), persfac(1,:), 20, [0.5 0.2 0.55], 'filled')
    hold on
    % Dr 0.1
    scatter(total_dist(2,:), persfac(2,:), 20, [0.1 0.35 0.7], 'filled')
    hold on
    % Dr 1
    scatter(total_dist(3,:), persfac(3,:), 20, [0.45 0.75 0.35], 'filled')
    hold on
    % Dr 10
    scatter(total_dist(4,:), persfac(4,:), 20, [0.65 0.1 0.2], 'filled')
    hold off
    axis([0 150 0 1])
    axis on;box on;set(gca,'LineWidth',3);set(gca, 'XTickLabel', []);set(gca, 'YTickLabel', []);set(gca,'TickLength',[0 0])
    % SAVING PNG FIGURE
    filename = ['Taxis_Analys/alpha_0_H_',H_str,'_f_',f_tax_str,'_TotDist_PF.png'];
    exportgraphics(gcf,filename,'Resolution',300,'Width',800,'Height',800);
    close(gcf);  
end


 

