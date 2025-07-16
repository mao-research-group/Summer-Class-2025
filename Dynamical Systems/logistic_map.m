%% Cobweb Plot
clear;clc;

% Parameters
r = 3.6;          % Growth rate
x0 = 0.2;         % Initial condition
N = 100;           % Number of iterations

% Allocate and iterate
x = zeros(1,N+1);
x(1) = x0;
for n = 1:N
    x(n+1) = r * x(n) * (1 - x(n));
end

% Plot cobweb
figure;
xn = x0;
for n = 1:N
    subplot(1,2,1);
    fplot(@(x) r*x.*(1 - x), [0,1], 'b', 'LineWidth', 2); 
    hold on;
    fplot(@(x) x, [0,1], 'k--');
    yn = r * xn * (1 - xn);
    plot([xn, xn], [xn, yn], 'r');     % vertical
    plot([xn, yn], [yn, yn], 'r');     % horizontal
    xn = yn;
    xlabel('$x_n$', 'Interpreter','latex');
    ylabel('$x_{n+1}$', 'Interpreter','latex');
    axis square;
    hold off;
    subplot(1,2,2);
    if n == 1
        plot(n,x(n),'ko');
    else
        plot(1:n,x(1:n),'k-','LineWidth', 2);
        hold on;
        plot(n,x(n),'ko');
    end
    hold off;
    drawnow;
    pause(0.1);
end

%% Bifurcation Diagram
clear;clc;

r_vals = linspace(0, 4, 10000);
num_iters = 1000;
transient = 100;

x = 0.2 * ones(1, length(r_vals));  % initial x for each r
lambda = zeros(size(r_vals));

figure; hold on;
for n = 1:num_iters
    x = r_vals .* x .* (1 - x);
    if n > transient
        plot(r_vals, x, 'k.', 'MarkerSize', 0.1);
    end
    lambda = lambda + log(abs(r_vals - 2 * r_vals .* x));
end
lambda = lambda / num_iters;

xlabel('r');
ylabel('x');
title('Logistic Map Bifurcation Diagram');
xlim([0 4]);
ylim([0 1]);

figure;
plot(r_vals, lambda, 'b');
yline(0, 'k--');
xlabel('r'); ylabel('\lambda');
title('Lyapunov Exponent of Logistic Map');
xlim([0 4]);
