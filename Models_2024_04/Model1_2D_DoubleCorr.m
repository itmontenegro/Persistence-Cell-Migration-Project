close all; clear all; clc;
start_time = tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Euler-Maruyama method on Smeets Model                                   %         
%                         SINGLE CELL MODEL                                               %
%                                                                                         %
%       1) Eq. of Motion: Fm*{cos(theta),sin(theta)} = gamma_s*v_i + alpha*dfW2(t)/dt     %
%       2) Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)/dt                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SET UP
Dr_values = [1];
H1_values = [0.5, 0.75, 0.99]; % H for Equation of Angle
H2_values = [0.5, 0.75, 0.99]; % H for Equation of Motion --> 2D (x and y)

for Hval1 = 1:length(H1_values)
    for Hval2 = 1:length(H2_values)
        for Dr_idx = 1:length(Dr_values)
            % Model Parameters
            Fm = 1; gamma_s = 1; Dr = Dr_values(Dr_idx); alpha = 0.25;
            % Simulation Parameters
            dt = 0.1; T = 100; Nts = T/dt;
            Nts2 = linspace(0, T, Nts);
            % Valores H actuales
            H1 = H1_values(Hval1); % H para dfW1
            H2 = H2_values(Hval2); % H para dfW2

            % Simulaciones
            for i = 1:50
                % Simulación individual
                dfW1 = zeros(Nts,1); 
                fbm_noise1 = fbm(Nts2, H1);
                dfW1(2:end) = diff(fbm_noise1);

                dfW2_x = zeros(Nts,1); 
                fbm_noise2_x = fbm(Nts2, H2);
                dfW2_x(2:end) = diff(fbm_noise2_x);            

                dfW2_y = zeros(Nts,1); 
                fbm_noise2_y = fbm(Nts2, H2);
                dfW2_y(2:end) = diff(fbm_noise2_y);

                % Inicialización
                x_array = zeros(Nts,1); y_array = zeros(Nts,1); theta_array = zeros(Nts,1);
                theta_array(1) = rand * 2 * pi;

                fmpi_x_array = zeros(Nts,1); fmpi_y_array = zeros(Nts,1);
                xi_x_array = zeros(Nts,1); xi_y_array = zeros(Nts,1);

                for t = 2:Nts
                    dtheta = sqrt(2*Dr) * dfW1(t);
                    theta_array(t) = theta_array(t-1) + dtheta;

                    force_term_dx = Fm * cos(theta_array(t)) / gamma_s;
                    stoch_term_dx = alpha * dfW2_x(t) / gamma_s;
                    force_term_dy = Fm * sin(theta_array(t)) / gamma_s;
                    stoch_term_dy = alpha * dfW2_y(t) / gamma_s;

                    x_array(t) = x_array(t-1) + force_term_dx * dt + stoch_term_dx;
                    y_array(t) = y_array(t-1) + force_term_dy * dt + stoch_term_dy;

                    fmpi_x_array(t) = force_term_dx * dt;
                    fmpi_y_array(t) = force_term_dy * dt;
                    xi_x_array(t) = stoch_term_dx;
                    xi_y_array(t) = stoch_term_dy;
                end

                % Guardado robusto
                H1_str = strrep(num2str(H1), '.', '_');
                H2_str = strrep(num2str(H2), '.', '_');

                base_folder = 'DATA_2025_DoubleCorr_Na2';
                main_folder = fullfile(base_folder, ['HAngle_', H1_str, '__HMotion_', H2_str]);

                if ~exist(main_folder, 'dir')
                    mkdir(main_folder);
                end

                filename = ['Sim_', num2str(i), '__HAngle_', H1_str, '__HMotion_', H2_str, '.csv'];
                path_save = fullfile(main_folder, filename);

                data = table(x_array, y_array, theta_array, fmpi_x_array, fmpi_y_array, xi_x_array, xi_y_array, ...
                    'VariableNames', {'X', 'Y', 'Theta', 'FmPi_x', 'FmPi_y', 'Xi_x', 'Xi_y'});

                writetable(data, path_save);
                fprintf('Saved: %s\n', path_save);
            end
        end
    end
end

end_time = toc(start_time);
fprintf('Completed ALL Combinations in: %.2f seconds\n', end_time);
