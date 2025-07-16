function van_der_pol_sim(mu)
    if nargin < 1
        mu = 1;  % Default mu if not specified
    end

    % Time span and initial conditions
    tspan = [0 1000];
    x0 = [0.01; 0];

    % Solve ODE
    [t, X] = ode45(@(t,x) vdp_ode(t, x, mu), tspan, x0);

    % Plot phase portrait
    figure;
    plot(X(:,1), X(:,2), 'b', 'LineWidth', 1.5); hold on;
    xlabel('$x$', 'Interpreter','latex');
    ylabel('$\dot{x}$', 'Interpreter','latex');
    title(['Van der Pol Oscillator ($\mu = ', num2str(mu), '$)'], 'Interpreter','latex');
    axis equal;
    grid on;
end

function dxdt = vdp_ode(~, x, mu)
    dxdt = [x(2); mu*(1 - x(1)^2)*x(2) - x(1)];
end