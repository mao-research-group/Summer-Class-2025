clear; clc;

%% Simulate true system: [position; velocity]
g = 9.8; alpha = 0.5;  % gravity and damping coefficient

fall_rhs = @(t, y) [y(2); -g - alpha * y(2)];
y0 = [10; 0];           % initial height = 10m, velocity = 0
tspan = 0:0.01:1.5;

[t, Y] = ode45(fall_rhs, tspan, y0);
dt = t(2) - t(1);

figure;
plot(t,Y(:,1),'r-');

%% Extract position, velocity and acceleration
pos = Y(:,1);
vel = Y(:,2);
acc = gradient(vel, dt);

%% Build library: position and velocity terms
Theta = [ones(size(pos)), pos, pos.^2, vel, vel.^2, vel.^3 ];

labels = {'1', 'y', 'y^2', 'v', 'v^2', 'v^3'};

%% Apply SINDy
% Sparse regression (thresholded least squares)
lambda = 0.05;
Xi = Theta \ acc;
for k = 1:10
    small = abs(Xi) < lambda;
    Xi(small) = 0;
    big = ~small;
    Xi(big) = Theta(:, big) \ acc;
end

% Print discovered acceleration model
fprintf('\n--- SINDy-Discovered Acceleration Model ---\n');
fprintf('a = ');
for i = 1:length(Xi)
    if abs(Xi(i)) > 1e-6
        fprintf('%+.4f * %s ', Xi(i), labels{i});
    end
end
fprintf('\n');

% Plot comparison
acc_pred = Theta * Xi;
figure;
plot(t, acc, 'k', 'LineWidth', 1.5); hold on;
plot(t, acc_pred, 'r--', 'LineWidth', 1.5);
xlabel('Time'); ylabel('Acceleration');
legend('True', 'SINDy'); title('Acceleration: True vs. SINDy');