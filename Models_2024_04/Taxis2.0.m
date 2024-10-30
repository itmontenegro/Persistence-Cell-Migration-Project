close all; clear all; clc;
start_time = tic;

% Parámetros del modelo
Dr_values = [0.5, 100]; % Coeficiente de difusión
H = 0.5;  % Parámetro de Hurst
dt = 0.5; 
T = 50;   
Nts = T / dt; 

% Definición del área de movimiento (límites del cuadrado)
area_limit = 20; % Define el tamaño del cuadrado este lo puedes cambiar ignacio
x_limits = [-area_limit, area_limit];
y_limits = [-area_limit, area_limit];

% Posición de la partícula este ya la pone aleatoria queda pendiente que no
% lo mande fuera de la zona delimitada 
particle_x = randi([-area_limit, area_limit]); % Posición aleatoria en x
particle_y = randi([-area_limit, area_limit]); % Posición aleatoria en y

% Iterar sobre los valores de Dr
for Dr = Dr_values 
    % Graficar la trayectoria para cada Dr
    figure;
    
    % Definición para el Movimiento Browniano Fraccionario (FBM)
    fbm_noise1 = fbm(Nts, H); % Generar ruido
    dfW1 = [0; diff(fbm_noise1)]; % Añadir 0 para mantener la longitud

    % Inicialización de variables
    x_array = zeros(Nts, 1);
    y_array = zeros(Nts, 1);
    theta_array = zeros(Nts, 1);
    theta_array(1) = rand * 2 * pi; % Ángulo inicial aleatorio

    % Simulación del movimiento celular
    for t = 2:Nts
        % Calcular las diferencias de posición
        delta_x = x_array(t - 1) - particle_x; % Diferencia en x
        delta_y = y_array(t - 1) - particle_y; % Diferencia en y
        
        % Calcular theta2 usando atan2 (desde la partícula hacia el eje x)
        theta2 = atan2(delta_y, delta_x); % Calcular theta2 con arcotangente pero esta pendiente 
        %verificar que si este bien planteada y funcione bien el angulo

        % Repolarización SDE: d(theta)/dt = sqrt(2*Dr) - dfW1(t) * (theta - theta2)
        dtheta = sqrt(2 * Dr) - dfW1(t) * (theta_array(t-1) - theta2); % Ajuste aquí la ecucion
        theta_array(t) = theta_array(t - 1) + dtheta * dt;

        % Ecuación de Movimiento: dx/dt y dy/dt
        dx_dt = cos(theta_array(t));
        dy_dt = sin(theta_array(t)); 

        % Actualización de la posición
        x_array(t) = x_array(t - 1) + dx_dt * dt;
        y_array(t) = y_array(t - 1) + dy_dt * dt;
        
        % Asegurarse de que la célula no salga del área delimitada esta es
        % como restriccion tambien toca hacer eso con particula pero por
        % ahora es una prueba
        x_array(t) = max(min(x_array(t), area_limit), -area_limit);
        y_array(t) = max(min(y_array(t), area_limit), -area_limit);
    end

    % Graficar el área delimitada queda como punteada
    hold on;
    rectangle('Position', [x_limits(1), y_limits(1), area_limit * 2, area_limit * 2], ...
              'EdgeColor', 'k', 'LineWidth', 1.5, 'LineStyle', '--'); % Área delimitada

    % Graficar la trayectoria
    plot(x_array, y_array, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Trayectoria'); % Trayectoria
    hold on;

    % Marcar el punto final
    final_x = x_array(end);
    final_y = y_array(end);
    plot(final_x, final_y, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Punto final de la célula'); % Punto final

    % Graficar la partícula
    plot(particle_x, particle_y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Partícula'); % Partícula

    % Graficar el vector de dirección en el punto final de la célula
    quiver(final_x, final_y, cos(theta_array(end)), sin(theta_array(end)), 0.5, ...
        'k', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', '\theta_1 (Célula)'); % Vector de dirección

    % Graficar el vector theta2 desde la partícula
    quiver(particle_x, particle_y, cos(theta2), sin(theta2), 0.5, ...
        'g', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', '\theta_2 (Partícula)'); % Vector theta2

    % Etiquetas y título
    xlabel('Posición X');
    ylabel('Posición Y');
    title(['Trayectoria de la Célula para Dr = ', num2str(Dr)]);
    axis equal; % Mantener la relación de aspecto
    xlim(x_limits); % Limitar el eje x
    ylim(y_limits); % Limitar el eje y
    grid on;

    % Agregar leyenda
    legend('show'); % Mostrar leyenda
end

end_time = toc(start_time);
fprintf('Tiempo total de simulación: %.2f segundos\n', end_time);

% Función para generar el Movimiento Browniano Fraccionario
function fbm_series = fbm(N, H)
    T = (1:N)';
    cov_matrix = 0.5 * ( (T * T') .^ (2 * H) + (T * T') .^ (2 * H) - abs(T - T').^ (2 * H) );
    
    L = chol(cov_matrix, 'lower');
    
    % Generación de la serie usando ruido blanco
    w = randn(N, 1);
    fbm_series = L * w; 
end
