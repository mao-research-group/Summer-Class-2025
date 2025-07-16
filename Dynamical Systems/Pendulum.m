clear; clc;

%% Simulate pendulum
g = 9.8;
pendulum_rhs = @(t, x) [x(2); -g * sin(x(1))];
x0 = [pi/2; 0];  % initial angle and velocity
tspan = 0:0.01:10;

[t, X] = ode45(pendulum_rhs, tspan, x0);
dt = t(2) - t(1);

figure;
plot(X(:,1),X(:,2),'r-');
xlabel('\theta');ylabel('\omega');

%% Extract position, velocity and acceleration
theta = X(:,1); dtheta = X(:,2);
ddtheta = gradient(dtheta, dt);

%% Build implicit library
Theta = [sin(theta), cos(theta),...
    dtheta, dtheta.^2,...
    sin(theta) .* dtheta, cos(theta).*dtheta];

% Theta = Theta./vecnorm(Theta);

labels = {'sin(theta)', 'cos(theta)', ...
          'dtheta', 'dtheta^2',...
          'sin(theta)*dtheta', 'cos(theta)*dtheta'};

%% Null space solution
[~, ~, V] = svd(Theta, 'econ');
xi = V(:, end);   % last column = smallest singular vector

%% Sparsify
lambda = 0.01;
xi(abs(xi) < lambda) = 0;

%% Print equation
fprintf('\n--- Implicit SINDy: Conserved Pendulum Dynamics ---\n');
terms = find(abs(xi) > 1e-6);
for i = 1:length(terms)
    fprintf('%+.4f * %s ', xi(terms(i)), labels{terms(i)});
end
fprintf(' = 0\n');
