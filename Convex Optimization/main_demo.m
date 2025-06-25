% main_optimization_demo.m
% Main script to run and visualize various optimization algorithms on the Rosenbrock function.

clear; clc; close all;

%% --- Problem Definition ---
% Define the objective function and its derivatives
f = @quadratic;
grad_f = @quadratic_grad;
hess_f = @quadratic_hess;

% Starting point
x0 = [-4; 1];

% Optimization parameters
max_iter = 200;
tol = 1e-6;

% Create a plot of the quadratic function
figure('Name','Function','Position',[100,100,500,400]);
[X, Y] = meshgrid(-5:0.05:5, -1.2:0.05:1.2);
Z = 1/2*X.^2 + 10*Y.^2;
surf(X, Y, Z,'EdgeColor','none','FaceAlpha',0.5);
shading interp;
colormap jet;
hold on;
plot3(x0(1),x0(2),quadratic(x0),'m.','MarkerSize',30);
view([32,32]);
xlabel('x_{1}');ylabel('x_{2}');zlabel('y');

%% --- Algorithm Execution ---
% Run each optimization algorithm

% 1. Gradient Descent
[x_gd_fixed, path_gd_fixed] = gradient_descent(grad_f, x0, max_iter, tol, 0.02, false);
[x_gd_ls, path_gd_ls] = gradient_descent(grad_f, x0, max_iter, tol, [], true);

% 2. Accelerated Gradient Descent (Nesterov)
[x_agd_fixed, path_agd_fixed] = accelerated_gradient_descent(grad_f, x0, max_iter, tol, 0.01, false);
[x_agd_ls, path_agd_ls] = accelerated_gradient_descent(grad_f, x0, max_iter, tol, [], true);

% 3. Newton's Method
[x_newton, path_newton_fixed] = newton_method(grad_f, hess_f, x0, max_iter, tol, false);
[x_newton_ls, path_newton_ls] = newton_method(grad_f, hess_f, x0, max_iter, tol, true);

% 4. Quasi-Newton Method (BFGS)
[x_bfgs, path_bfgs_fixed] = quasi_newton_bfgs(grad_f, x0, max_iter, tol, false);
[x_bfgs_ls, path_bfgs_ls] = quasi_newton_bfgs(grad_f, x0, max_iter, tol, true);

%% --- Visualization ---
figure('Position', [100 100 1000 500]);
subplot(1,2,1);
hold on;

[X, Y] = meshgrid(-5:0.05:5, -1.2:0.05:1.2);
Z = 1/2*X.^2 + 10*Y.^2;
contour(X, Y, Z, 20);
shading interp;

% Plot optimization paths
plot(path_gd_fixed(1,:), path_gd_fixed(2,:), 'r-', 'DisplayName', 'GD', 'LineWidth', 1.5);
hold on;
plot(path_agd_fixed(1,:), path_agd_fixed(2,:), 'y-', 'DisplayName', 'Accelerated GD', 'LineWidth', 1.5);
plot(path_newton_fixed(1,:), path_newton_fixed(2,:), 'g-', 'DisplayName', 'Newton', 'LineWidth', 1.5);
plot(path_bfgs_fixed(1,:), path_bfgs_fixed(2,:), 'm-', 'DisplayName', 'Quasi-Newton BFGS', 'LineWidth', 1.5);

% Plot the minimum
plot(0, 0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'red', 'DisplayName', 'Global Minimum');
xlabel('x_1');
ylabel('x_2');
legend('Location', 'best');
title('Fixed step');
hold off;
axis([-5,5,-1.2,1.2]);

subplot(1,2,2);
hold on;

contour(X, Y, Z, 20);
shading interp;

% Plot optimization paths
plot(path_gd_ls(1,:), path_gd_ls(2,:), 'r-', 'DisplayName', 'GD', 'LineWidth', 1.5);
hold on;
plot(path_agd_ls(1,:), path_agd_ls(2,:), 'y-', 'DisplayName', 'Accelerated GD', 'LineWidth', 1.5);
plot(path_newton_ls(1,:), path_newton_ls(2,:), 'g-', 'DisplayName', 'Newton', 'LineWidth', 1.5);
plot(path_bfgs_ls(1,:), path_bfgs_ls(2,:), 'm-', 'DisplayName', 'Quasi-Newton BFGS', 'LineWidth', 1.5);

% Plot the minimum
plot(0, 0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'red', 'DisplayName', 'Global Minimum');
title('Line Search');
xlabel('x_1');
ylabel('x_2');
legend('Location', 'best');
hold off;
axis([-5,5,-1.2,1.2]);

sgtitle('Comparison of Optimization Algorithms on Quadratic Function');

%% Sub-Gradient Method vs Proximal Gradient Descent
clear;clc;
% Parameters
A = [3, 0.5; 1.5, 2];
b = [1; 1];
lambda = 5;
eta_pg = 0.05;
eta_sub = 0.05;
max_iter = 20;
x_pg = [-1; 1];
x_sub = [-1; 1];
path_pg = x_pg;
path_sub = x_sub;

% Define the objective
f = @(x) 0.5 * norm(A * x - b)^2 + lambda * norm(x, 1);

% Create contour grid
[x1, x2] = meshgrid(linspace(-2, 2, 200), linspace(-2, 2, 200));
Z = zeros(size(x1));
for i = 1:numel(x1)
    Z(i) = f([x1(i); x2(i)]);
end

% Initialize figure
figure('Position', [100, 100, 600, 600]);
contour(x1, x2, Z, 30);
hold on;
axis equal;
xlim([-2, 2]);
ylim([-2, 2]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
title('Proximal Gradient vs Subgradient ($\ell_1$-regularized)', 'Interpreter', 'latex');

pg_plot = plot(x_pg(1), x_pg(2), 'ro-', 'LineWidth', 2, 'DisplayName', 'Proximal Gradient');
sub_plot = plot(x_sub(1), x_sub(2), 'bs:', 'LineWidth', 1.5, 'DisplayName', 'Subgradient');

legend('show');

% Run optimization loop
for k = 1:max_iter
    % Gradient of smooth part
    grad_pg = A' * (A * x_pg - b);
    % Proximal gradient step (soft thresholding)
    x_pg = soft_threshold(x_pg - eta_pg * grad_pg, eta_pg * lambda);
    path_pg = [path_pg, x_pg];

    % Subgradient step
    grad_sub = A' * (A * x_sub - b);
    subgrad_l1 = sign(x_sub);
    x_sub = x_sub - eta_sub * (grad_sub + lambda * subgrad_l1);
    path_sub = [path_sub, x_sub];

    % Update plots
    set(pg_plot, 'XData', path_pg(1, :), 'YData', path_pg(2, :));
    set(sub_plot, 'XData', path_sub(1, :), 'YData', path_sub(2, :));
    pause();
end