clear all; close all;

%% LOAD PROCESSED CSVs

mainFolderPath = 'Model1_H_cnst/Model1_2D_Batch200/Alpha_1/H_0_01/';
fmpi_x_data = readtable([mainFolderPath, 'fmpi_x.csv']);
fmpi_y_data = readtable([mainFolderPath, 'fmpi_y.csv']);
xi_x_data = readtable([mainFolderPath, 'xi_x.csv']);
xi_y_data = readtable([mainFolderPath, 'xi_y.csv']);
norm_fmpi_data = readtable([mainFolderPath, 'norm_fmpi.csv']);
norm_xi_data = readtable([mainFolderPath, 'norm_xi.csv']);
time_norm_xi_data = readtable([mainFolderPath, 'mean_time_norm_xi.csv']);

%% SECTION CALCULATES THE MEAN OF THE NORMS
%% For H = 0.01
norm_fmpi_1 = norm_fmpi_data.Alpha_1__Dr_1__H_0_01;
norm_xi_1 = norm_xi_data.Alpha_1__Dr_1__H_0_01;

figure;
plot(1:1000, (norm_fmpi_1), 'r-', 'LineWidth', 2);  % Make this line thicker
hold on;  % Hold the plot to add another line
plot(1:1000, (norm_xi_1), 'b--', 'LineWidth', 1);  % Keep this line with regular thickness
hold on;
% Plotting the mean value over time as a horizontal line
mean_value = mean(norm_xi_1);
refline(0, mean_value);
text(500, mean_value + 0.15, ['\downarrow Mean value: ' num2str(mean_value)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left','FontWeight', 'bold', 'BackgroundColor', 'yellow', 'EdgeColor', 'red', 'LineWidth', 1.5);
hold off;
title('Plot for the MEAN OF THE NORMS for F_m\cdot p_i and \alpha\cdot\xi(t)')
subtitle('H = 0.01, Alpha = 1')
axis ([0 1000 0 7]);
xlabel('t');
ylabel('Value');
legend('F_m\cdot p_i', '\alpha\cdot\xi(t)');  % Add legend labels
grid on;  % Add grid lines for better visualization
saveas(gcf, ('Model1_H_cnst/Model1_2D_Batch200/Alpha_1/Figures/Mean_Norms_H_0_01.png'));
%close(gcf);
%% For H = 0.25
norm_fmpi_2 = norm_fmpi_data.Alpha_1__Dr_1__H_0_25;
norm_xi_2 = norm_xi_data.Alpha_1__Dr_1__H_0_25;

figure;
plot(1:1000, (norm_fmpi_2), 'r-', 'LineWidth', 2);  % Make this line thicker
hold on;  % Hold the plot to add another line
plot(1:1000, (norm_xi_2), 'b--', 'LineWidth', 1);  % Keep this line with regular thickness
hold on;
% Plotting the mean value over time as a horizontal line
mean_value = mean(norm_xi_2);
refline(0, mean_value);
text(500, mean_value + 0.15, ['\downarrow Mean value: ' num2str(mean_value)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left','FontWeight', 'bold', 'BackgroundColor', 'yellow', 'EdgeColor', 'red', 'LineWidth', 1.5);
hold off;
title('Plot for the MEAN OF THE NORMS for F_m\cdot p_i and \alpha\cdot\xi(t)')
subtitle('H = 0.25, Alpha = 1')
axis ([0 1000 0 7]);
xlabel('t');
ylabel('Value');
legend('F_m\cdot p_i', '\alpha\cdot\xi(t)');  % Add legend labels
grid on;  % Add grid lines for better visualization
saveas(gcf, ('Model1_H_cnst/Model1_2D_Batch200/Alpha_1/Figures/Mean_Norms_H_0_25.png'));
% close(gcf);

%% For H = 0.5
norm_fmpi_3 = norm_fmpi_data.Alpha_1__Dr_1__H_0_5;
norm_xi_3 = norm_xi_data.Alpha_1__Dr_1__H_0_5;

figure;
plot(1:1000, (norm_fmpi_3), 'r-', 'LineWidth', 2);  % Make this line thicker
hold on;  % Hold the plot to add another line
plot(1:1000, (norm_xi_3), 'b--', 'LineWidth', 1);  % Keep this line with regular thickness
hold on;
% Plotting the mean value over time as a horizontal line
mean_value = mean(norm_xi_3);
refline(0, mean_value);
text(500, mean_value + 0.15, ['\downarrow Mean value: ' num2str(mean_value)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left','FontWeight', 'bold', 'BackgroundColor', 'yellow', 'EdgeColor', 'red', 'LineWidth', 1.5);
hold off;
title('Plot for the MEAN OF THE NORMS for F_m\cdot p_i and \alpha\cdot\xi(t)')
subtitle('H = 0.5, Alpha = 1')
axis ([0 1000 0 7]);
xlabel('t');
ylabel('Value');
legend('F_m\cdot p_i', '\alpha\cdot\xi(t)');  % Add legend labels
grid on;  % Add grid lines for better visualization
saveas(gcf, ('Model1_H_cnst/Model1_2D_Batch200/Alpha_1/Figures/Mean_Norms_H_0_50.png'));
%close(gcf);

%% For H = 0.75
norm_fmpi_4 = norm_fmpi_data.Alpha_1__Dr_1__H_0_75;
norm_xi_4 = norm_xi_data.Alpha_1__Dr_1__H_0_75;

figure;
plot(1:1000, (norm_fmpi_4), 'r-', 'LineWidth', 2);  % Make this line thicker
hold on;  % Hold the plot to add another line
plot(1:1000, (norm_xi_4), 'b--', 'LineWidth', 1);  % Keep this line with regular thickness
hold on;
% Plotting the mean value over time as a horizontal line
mean_value = mean(norm_xi_4);
refline(0, mean_value);
text(500, mean_value + 0.15, ['\downarrow Mean value: ' num2str(mean_value)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left','FontWeight', 'bold', 'BackgroundColor', 'yellow', 'EdgeColor', 'red', 'LineWidth', 1.5);
hold off;
title('Plot for the MEAN OF THE NORMS for F_m\cdot p_i and \alpha\cdot\xi(t)')
subtitle('H = 0.75, Alpha = 1')
axis ([0 1000 0 7]);
xlabel('t');
ylabel('Value');
legend('F_m\cdot p_i', '\alpha\cdot\xi(t)');  % Add legend labels
grid on;  % Add grid lines for better visualization
saveas(gcf, ('Model1_H_cnst/Model1_2D_Batch200/Alpha_1/Figures/Mean_Norms_H_0_75.png'));
%close(gcf);

%% For H = 0.99
norm_fmpi_5 = norm_fmpi_data.Alpha_1__Dr_1__H_0_99;
norm_xi_5 = norm_xi_data.Alpha_1__Dr_1__H_0_99;

figure;
plot(1:1000, (norm_fmpi_5), 'r-', 'LineWidth', 2);  % Make this line thicker
hold on;  % Hold the plot to add another line
plot(1:1000, (norm_xi_5), 'b--', 'LineWidth', 1);  % Keep this line with regular thickness
hold on;
% Plotting the mean value over time as a horizontal line
mean_value = mean(norm_xi_5);
refline(0, mean_value);
text(500, mean_value + 0.15, ['\downarrow Mean value: ' num2str(mean_value)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left','FontWeight', 'bold', 'BackgroundColor', 'yellow', 'EdgeColor', 'red', 'LineWidth', 1.5);
hold off;
title('Plot for the MEAN OF THE NORMS for F_m\cdot p_i and \alpha\cdot\xi(t)')
subtitle('H = 0.99, Alpha = 1')
axis ([0 1000 0 7]);
xlabel('t');
ylabel('Value');
legend('F_m\cdot p_i', '\alpha\cdot\xi(t)');  % Add legend labels
grid on;  % Add grid lines for better visualization
saveas(gcf, ('Model1_H_cnst/Model1_2D_Batch200/Alpha_1/Figures/Mean_Norms_H_0_99.png'));
%close(gcf);