clear; clc;

%% Simulate the true system
f = @(t,x) x^2 / (1 + x);
x0 = 1;
tspan = 0:0.01:5;
[t, x] = ode45(f, tspan, x0);
dt = t(2) - t(1);

figure;
plot(t,x,'r-');

%% Estimate dx/dt
dx = gradient(x, dt);

%% Build joint library: Theta(x, dx)
Theta = [dx, x, x.^2, x .* dx, dx.^2, x.^3, x.^2.*dx];
% Theta = Theta./vecnorm(Theta);

labels = {'dx', 'x', 'x^2', 'x*dx', 'dx^2', 'x^3', 'x^2*dx'};

%% Null space: solve Theta * xi = 0
[~, S, V] = svd(Theta, 'econ');
xi = V(:,end);

%% Sparsify
lambda = 0.35;
xi(abs(xi) < lambda) = 0;

%% Display result
fprintf('\n--- Implicit SINDy Recovered Equation ---\n');
terms = find(abs(xi) > 1e-6);
for i = 1:length(terms)
    fprintf('%+.4f * %s ', xi(terms(i)), labels{terms(i)});
end
fprintf(' = 0\n');
