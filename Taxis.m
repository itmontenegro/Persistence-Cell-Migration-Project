close all; clear all; clc;
start_time = tic;

%% SET UP
Dr_values = [0.1, 0.5, 1, 5, 100]; 
taxis_results = []; % Almacena resultados de taxis

for Drval = 1:length(Dr_values)
    start_time_model = tic; 
    % Parámetros del modelo
    Fm = 1; 
    gamma_s = 1; 
    Dr = Dr_values(Drval); 
    alpha = 2.5; 

    % Inicializa variables para la simulación
    dt = 0.1; 
    T = 100; 
    Nts = T/dt; 
    dfW1 = zeros(Nts, 1); 
    fbm_noise1 = fbm(Nts, 0.5); 
    dfW1(2:end) = diff(fbm_noise1); 

    % Condiciones iniciales
    x_array = zeros(Nts, 1); 
    y_array = zeros(Nts, 1); 
    theta_array = zeros(Nts, 1); 
    x_array(1) = 0; 
    y_array(1) = 0;  
    theta_array(1) = rand * 2 * pi; 

    %% Generar posición aleatoria para la partícula
    particle_x = rand * 100 - 50; 
    particle_y = rand * 100 - 50; 

    %% SIMULACIÓN
    for t = 2:Nts
        H = abs(x_array(t-1)) * 0.005 + 0.5;
        Nts2 = linspace(0, t*dt, t);  
        fbm_noise2 = fbm(length(Nts2), H);  
        dfW2 = diff(fbm_noise2); 

        % Calcula el movimiento a partir del tiempo anterior
        if t > 2 % Asegúrate de que t sea mayor que 2
            delta_x = x_array(t-1) - x_array(t-2); 
            delta_y = y_array(t-1) - y_array(t-2); 
            theta2 = atan2(delta_y, delta_x); 
        else
            theta2 = theta_array(t-1); % Si t es 2, solo usa el último theta
        end

        dtheta = sqrt(2*Dr) * dfW1(t) - dfW1(t) * (theta_array(t-1) - theta2);
        theta_array(t) = theta_array(t-1) + dtheta * dt; 

        dx_dt = (Fm * cos(theta_array(t)) - alpha * dfW2(t-1)) / gamma_s;
        x_array(t) = x_array(t-1) + dx_dt * dt;
        y_array(t) = y_array(t-1) + Fm * sin(theta_array(t)); 
        
        % Determina si es taxis positivo o negativo
        distance_to_particle = sqrt((x_array(t) - particle_x)^2 + (y_array(t) - particle_y)^2);
        if distance_to_particle < 1
            taxis_type = 'Positivo'; % Si está cerca de la partícula
        else
            taxis_type = 'Negativo'; % Si está lejos de la partícula
        end

        % Almacenar el resultado de taxis
        taxis_results(t, 1) = t * dt; % Almacenar tiempo
        taxis_results(t, 2) = strcmp(taxis_type, 'Positivo'); % 1 para positivo, 0 para negativo

        % Imprimir el resultado de cada paso
        fprintf('Tiempo: %.2f s, Taxis: %s\n', taxis_results(t, 1), taxis_type);
    end

    % Gráfica de la trayectoria de la célula y la partícula
    figure; 
    hold on;

    % Graficar la trayectoria de la célula
    plot(x_array, y_array, 'o-', 'MarkerSize', 4, 'MarkerFaceColor', 'b');

    % Agregar el punto de la partícula (posición estática)
    scatter(particle_x, particle_y, 100, 'filled', 'MarkerFaceColor', 'g');

    % Última posición de la célula
    last_cell_x = x_array(end);
    last_cell_y = y_array(end);
    scatter(last_cell_x, last_cell_y, 100, 'filled', 'MarkerFaceColor', 'r');

    % Calcular los ángulos theta1 y theta2
    theta1 = theta_array(end); % Último ángulo de la célula
    theta2 = atan2(particle_y - last_cell_y, particle_x - last_cell_x); % Ángulo hacia la partícula

    % Graficar vectores de ángulo theta1 (célula) y theta2 (partícula)
    quiver(last_cell_x, last_cell_y, cos(theta1), sin(theta1), 2, 'k', 'LineWidth', 2, 'DisplayName', 'Theta1 (Célula)');
    quiver(last_cell_x, last_cell_y, cos(theta2), sin(theta2), 2, 'm', 'LineWidth', 2, 'DisplayName', 'Theta2 (Partícula)');

    % Configuración de la gráfica
    xlabel('Posición en X');
    ylabel('Posición en Y');
    title(['Trayectoria de la Célula y Partícula (Dr = ' num2str(Dr) ')']);
    legend('Trayectoria Célula', 'Partícula', 'Última Posición Célula', 'Theta1 (Célula)', 'Theta2 (Partícula)');
    grid on;
    axis equal;
    hold off;

    end_time = toc(start_time_model);
    fprintf('Completed Model in: %.2f seconds\n', end_time);   
end

end_time = toc(start_time);
fprintf('Completed ALL in: %.2f seconds\n', end_time);

% Función para generar movimiento fraccional Browniano
function G = fbm(N, H)
    % Verifica que N sea un número entero positivo
    if N <= 0 || floor(N) ~= N
        error('N debe ser un entero positivo.');
    end

    % Genera ruido blanco gaussiano
    G = randn(1, N); 

    % Generar el movimiento fraccional mediante la integración de un ruido blanco
    for i = 2:N
        G(i) = G(i-1) + (1/2) * (G(i-1) + G(i)) * (1 - H) * (1/(i^(2*H))); 
    end
    G = cumsum(G); % Suma acumulativa para simular el movimiento
end
