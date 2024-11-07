close all;clear all;clc;
start_time = tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Euler-Maruyama method on Smeets Model                                   %         
%                         SINGLE CELL MODEL                                               %
%                                                                                         %
%       1) Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + alpha*dfW2(t)          %
%       2) Repolarization SDE: d(theta)/dt = sqrt(2*Dr)*dfW1(t)                           %
%                                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SET UP
% Parámetros del modelo
Dr_values = [10]; % Coeficiente de difusión
H_values = [0.5];  % Parámetro de Hurst
Fm = 1;
gamma_s = 1;
alpha = 0;
dt = 0.1; 
T = 100;   
Nts = T / dt; 
ftax = 0.1;
Nts2 = linspace(0,T,Nts);     

% Definición del área de movimiento (límites del cuadrado)
area_limit = 100; % Define el tamaño del cuadrado este lo puedes cambiar ignacio
x_limits = [-area_limit, area_limit];
y_limits = [-area_limit, area_limit];

% Posición del CENTRO ORGANIZADOR
particle_x = -50; % x_ORG
particle_y = -50; % y_ORG

%% PARAMETER VALUES
% Iterar sobre los valores de Dr
for Dr = Dr_values 
    % Graficar la trayectoria para cada Dr
    figure;
    % Definition for Fractional Brownian Motion 1 -> White Noise Angle
    dfW1 = zeros(Nts,1); 
    fbm_noise1 = fbm(Nts2,0.5);
    dfW1(2:end) = diff(fbm_noise1);

    % Definition for Fractional Brownian Motion 2 -> Correlated Noise
    H = H_values(1);
    dfW2_x = zeros(Nts,1); 
    fbm_noise2_x = fbm(Nts2,H);
    dfW2_x(2:end) = diff(fbm_noise2_x);            
    dfW2_y = zeros(Nts,1); 
    fbm_noise2_y = fbm(Nts2,H);
    dfW2_y(2:end) = diff(fbm_noise2_y);  
    
    %% SIMULATION INITIALIZATION
    % Inicialización de variables
    x_array = zeros(Nts, 1);
    y_array = zeros(Nts, 1);
    theta_array = zeros(Nts, 1);
    theta_array(1) = rand * 2 * pi; % Ángulo inicial aleatorio

    % Simulación del movimiento celular
    for t = 2:Nts

        %% 1) ANGLE CALCULATION
        % Calcular las diferencias de posición con el CENTRO ORGANIZADOR
        delta_x = particle_x - x_array(t - 1) ; % Diferencia en x
        delta_y = particle_y - y_array(t - 1) ; % Diferencia en y
        normalized_dx = delta_x/(delta_x^2 + delta_y^2);
        normalized_dy = delta_y/(delta_x^2 + delta_y^2);
        
        % Calcular theta2 usando atan2 (desde la partícula hacia el eje x)
        % theta2 = atan2(delta_y, delta_x); % Calcular theta2 con arcotangente pero esta pendiente 
        %verificar que si este bien planteada y funcione bien el angulo
        %% UNSURE, A CORRECT ALREADY TESTED OPTION IS
        theta_org_alignment = normalized_dx * cos(theta_array(t-1)) + normalized_dy * sin(theta_array(t-1));
        theta_org = acos(theta_org_alignment);
        %% MISSING SIGN OF THE REQUIRED REORIENTATION
        cross_angle = normalized_dx * sin(theta_array(t-1)) - normalized_dy * cos(theta_array(t-1));
        sign_angle = sign(cross_angle);

        %% CORRECTED EQUATION
        % Repolarización SDE: d(theta)/dt = -ftax(theta-theta_org) + sqrt(2*Dr)*dfW1(t)
        dtheta = -ftax*theta_org*sign_angle + sqrt(2*Dr)*dfW1(t); % Ajuste aquí la ecucion
        theta_array(t) = theta_array(t - 1) + dtheta * dt;

        %% CORRECTED POSITION UPDATE
        % Eq. of Motion: Fm*{cos(theta),sin(theta)}=gamma_s*v_i + alpha*dfW2(t)
        dx_dt = (Fm*cos(theta_array(t)) - alpha*dfW2_x(t))/(gamma_s);
        dy_dt = (Fm*sin(theta_array(t)) - alpha*dfW2_y(t))/(gamma_s);
        % Update of Position
        x_array(t) = x_array(t-1) + dx_dt*dt;
        y_array(t) = y_array(t-1) + dy_dt*dt;


        % Asegurarse de que la célula no salga del área delimitada esta es
        % como restriccion tambien toca hacer eso con particula pero por
        % ahora es una prueba
        % x_array(t) = max(min(x_array(t), area_limit), -area_limit);
        % y_array(t) = max(min(y_array(t), area_limit), -area_limit);
        %% UNNECESARY FOR CELLS TO BE CONFINED WITH THIS MODEL

    end

    % Graficar el área delimitada queda como punteada
    hold on;
    rectangle('Position', [x_limits(1), y_limits(1), area_limit * 2, area_limit * 2], ...
              'EdgeColor', 'k', 'LineWidth', 1.5, 'LineStyle', '--'); % Área delimitada

    % Graficar la trayectoria
    plot(x_array, y_array, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Trajectory'); % Trayectoria
    hold on;

    % Marcar el punto final
    final_x = x_array(end);
    final_y = y_array(end);
    plot(final_x, final_y, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Cell'); % Punto final

    % Graficar la partícula
    plot(particle_x, particle_y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Org. Cent.'); % Partícula

    % Graficar el vector de dirección en el punto final de la célula
    quiver(final_x, final_y, cos(theta_array(end)), sin(theta_array(end)), 0.5, ...
        'k', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', '\vec{p}'); % Vector de dirección

    % Graficar el vector theta2 desde la partícula
    quiver(particle_x, particle_y, cos(theta_org), sin(theta_org), 0.5, ...
        'g', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', '\vec{v}_{org}'); % Vector theta2

    % Etiquetas y título
    xlabel('X');
    ylabel('Y');
    title(['Cell migration for Dr = ', num2str(Dr)]);
    axis ([-100 100 -100 100]); % Mantener la relación de aspecto
    grid on;

    % Agregar leyenda
    legend('show'); % Mostrar leyenda
end

end_time = toc(start_time);
fprintf('Tiempo total de simulación: %.2f segundos\n', end_time);