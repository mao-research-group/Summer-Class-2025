clear;clc;

%% Colinearity
n = 100;
x1 = linspace(0, 10, n)';
x2 = x1 + 0.2 * randn(n, 1);  % x2 is highly correlated with x1
X = [ones(n,1), x1, x2];       % Design matrix with intercept
beta_true = [1; 2; -1];        % True coefficients

% Generate noisy observations
y = X * beta_true + 0.05 * randn(n,1);

% Check correlation
corrcoef(x1, x2)

%% SVD
[U, S, V] = svd(X, 'econ');
beta_svd = V*inv(S)*U'*y;

fprintf('SVD coefficients:    %.4f  %.4f  %.4f\n', beta_svd);
fprintf('True coefficients:   %.4f  %.4f  %.4f\n', beta_true);

cond_number = cond(X' * X);  % Large if near-singular
fprintf('Condition number of X^T X: %.2e\n', cond_number);


%% Another example
clear;clc;

% Generate 1D data
n = 20;
x = linspace(0, 10, n)';
a_true = 1.5;       % intercept
b_true = 2.0;       % slope
noise = 1.0 * randn(n, 1);   % noise

% True model
y = a_true + b_true * x + noise;

% Design matrix for linear regression: y ≈ a + b x
A = [ones(n,1), x];

[U, S, V] = svd(A, 'econ');
beta_svd = V * (S \ (U' * y));
a_est = beta_svd(1);
b_est = beta_svd(2);

% True line
y_true = a_true + b_true * x;

% Estimated line
y_est = a_est + b_est * x;

figure;
scatter(x, y, 50, 'ko', 'filled'); hold on;
plot(x, y_true, 'b-', 'LineWidth', 2);
plot(x, y_est, 'r--', 'LineWidth', 2);
legend('Noisy data', 'True line', 'SVD regression', 'Location', 'northwest');
xlabel('x'); ylabel('y');
title('1D Regression via SVD');
grid on;
