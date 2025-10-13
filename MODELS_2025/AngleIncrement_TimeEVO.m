close all;clear all;

%%
Dr_values = [0,0.1,1,10];
H_values = [0.5];
for Hval = 1:length(H_values)
    for Drval = 1:length(Dr_values)
        %% Generating based on values of H and Dr
        Dr = Dr_values(Drval); H = H_values(Hval); alpha = 0;
        % Loading 10 random datasets of the 200 sims
        permuted_indices = randperm(200);
        r = permuted_indices(1:10);
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

        %% NOW TO MAKE THE ANGLE EVOLUTION OF THESE TRAJECTORIES
        file_angle_full = zeros(998,10);
        file_step_full = zeros(998,10);
        for j = 1:10
            data_name = ['name_', num2str(j)]
            data_sim = readtable(eval(data_name));
            % For each timestep
            for t = 3:height(data_sim)
                diff_X = (data_sim.X(t) - data_sim.X(t-1))^2;
                diff_Y = (data_sim.Y(t) - data_sim.Y(t-1))^2;
                increment = sqrt(diff_X + diff_Y);
                file_step_full(t,j) = increment;
                step_0 = sqrt((data_sim.X(t-1)-data_sim.X(t-2))^2 + (data_sim.Y(t-1)-data_sim.Y(t-2))^2);
                step_1 = sqrt((data_sim.X(t)-data_sim.X(t-1))^2 + (data_sim.Y(t)-data_sim.Y(t-1))^2);
                step_2 = sqrt((data_sim.X(t)-data_sim.X(t-2))^2 + (data_sim.Y(t)-data_sim.Y(t-2))^2);                
                file_angle_full(t, j) = pi - acos((step_0^2+step_1^2-step_2^2)/(2*step_0*step_1));
            end
        end
        %% PLOT FOR ANGLE TIME EVOLUTION
        time_vector = [3:1:1000].';
        figure;
        for curve = 1:3
            plot(time_vector(1:998),(file_angle_full(3:1000,curve)),'LineWidth', 1);
            hold on;
        end
        hold off;
        axis ([0 1000 0 pi])
        % Show the axes without tick values and labels
        axis on;
        box on;
        set(gca,'LineWidth',3)
        set(gca, 'XTickLabel', []);
        set(gca, 'YTickLabel', []);
        set(gca,'TickLength',[0 0])
        % Saving
        filename = ['../FIGURES_2025/AngleEVO/' ...
            'Alpha_',alpha_str,'__H_',H_str,'__Dr_',Dr_str,'__AngleEVO.png'];
        exportgraphics(gcf,filename,'Resolution',300);
        close(gcf);
        
        %% PLOT FOR INCREMENT TIME EVOLUTION
        time_vector = [3:1:1000].';
        figure;
        for curve = 1:3
            plot(time_vector(1:998),(file_step_full(3:1000,curve)),'LineWidth', 1);
            hold on;
        end
        hold off;
        axis ([0 1000 0 0.3])
        % Show the axes without tick values and labels
        axis on;
        box on;
        set(gca,'LineWidth',3)
        set(gca, 'XTickLabel', []);
        set(gca, 'YTickLabel', []);
        set(gca,'TickLength',[0 0])
        % Saving
        filename = ['../FIGURES_2025/IncrementEVO/' ...
            'Alpha_',alpha_str,'__H_',H_str,'__Dr_',Dr_str,'__IncrementEVO.png'];
        exportgraphics(gcf,filename,'Resolution',300);
        close(gcf);
    end
end