clear; clc;

%% Define initial state and simulate ground truth
x0 = [2; 0];               % Initial conditions
tspan = 0:0.01:20;

[t, X] = ode45(@(t,x) [x(2); -x(1) - x(1)^3], tspan, x0);
dt = t(2) - t(1);

figure;
subplot(2,1,1);
plot(t,X(:,1),'r-');
xlabel('t');ylabel('x_{1}');
subplot(2,1,2);
plot(t,X(:,2),'b-');
xlabel('t');ylabel('x_{2}');

%% Compute time derivatives
dXdt = zeros(size(X));
dXdt(:,1) = X(:,2);                     % dx1/dt = x2
dXdt(:,2) = gradient(X(:,2), dt);       % Estimate dx2/dt numerically

%% Build polynomial library Theta(x)
x1 = X(:,1); x2 = X(:,2);
Theta = [ones(size(x1)), x1, x2, x1.^2, x1.*x2, x2.^2, x1.^3];
labels = {'1', 'x1', 'x2', 'x1^2', 'x1*x2', 'x2^2', 'x1^3'};

%% Apply SINDy
lambda = 0.1;
Xi = Theta \ dXdt;
for k = 1:10
    small = abs(Xi) < lambda;
    Xi(small) = 0;
    for i = 1:2
        big = ~small(:,i);
        Xi(big,i) = Theta(:,big) \ dXdt(:,i);
    end
end

%% Print discovered equations
fprintf('\n--- SINDy Recovered Equations ---\n');
for eq = 1:2
    fprintf('dx%d/dt = ', eq);
    for j = 1:length(labels)
        if abs(Xi(j,eq)) > 1e-6
            fprintf('%+.4f * %s ', Xi(j,eq), labels{j});
        end
    end
    fprintf('\n');
end

%% Simulate SINDy model
[tS, XS] = ode45(@(t,x) sindy_rhs(x, Xi), tspan, x0);

%% Compare
figure;
subplot(2,1,1);
plot(t, X(:,1), 'k', 'linewidth',3);
hold on;
plot(tS, XS(:,1), 'r--', 'linewidth',1); 
ylabel('x_1'); 
legend('True','SINDy');
subplot(2,1,2);
plot(t, X(:,2), 'k', 'linewidth',3);
hold on;
plot(tS, XS(:,2), 'r--', 'linewidth',1); 
ylabel('x_2'); xlabel('Time');


