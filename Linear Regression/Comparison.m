% Demonstrate the difference among models using a synthetic dataset
clear;clc;

%% 1. Generate a Synthetic Dataset
% We will create a dataset with many features, but only a few of them
% will actually be used to generate the output.

num_samples = 100;     % Number of data points
num_features = 20;     % Total number of features
num_relevant_features = 4; % Number of features that are actually useful

% Create random predictor data
X_data = randn(num_samples, num_features);

% Create a "true" coefficient vector where most coefficients are zero
b_true = zeros(num_features, 1);
b_true(1:num_relevant_features) = [5; -3.5; 2; 4.2]; % Set the first few to non-zero values

% Generate the response variable y = X*b + noise
noise = 2 * randn(num_samples, 1);
y = X_data * b_true + noise;

% --- Prepare data for models ---
% Add a column of ones for the intercept term
X = [ones(num_samples, 1), X_data];
[m, p] = size(X); % m = samples, p = features + intercept

%% 2. Define Regularization Parameters
% These values are chosen to clearly show the effects of each model.
alpha_ridge = 40;
alpha_lasso = 40; % Higher alpha to enforce sparsity

%% 3. Solve

% --- Linear Regression (Normal Equation) ---
b_linear = (X' * X) \ (X' * y);

% --- Ridge Regression (Regularized Normal Equation) ---
I = eye(p);
I(1,1) = 0;
b_ridge = (X' * X + alpha_ridge * I) \ (X' * y);

% --- LASSO Regression (Coordinate Descent) ---
num_iterations = 1000;
b_lasso = zeros(p, 1); % Initialize coefficients

for iter = 1:num_iterations
    for j = 1:p
        b_lasso(j) = 0;
        y_pred = X * b_lasso;
        r = y - y_pred;
        rho_j = sum(X(:, j) .* r);
        
        if j == 1 % Don't penalize intercept
            b_lasso(j) = rho_j / sum(X(:,j).^2);
        else
            % Apply soft-thresholding
            if rho_j < -alpha_lasso / 2
                b_lasso(j) = (rho_j + alpha_lasso / 2) / sum(X(:,j).^2);
            elseif rho_j > alpha_lasso / 2
                b_lasso(j) = (rho_j - alpha_lasso / 2) / sum(X(:,j).^2);
            else
                b_lasso(j) = 0;
            end
        end
    end
end

%% 4. Compare the Coefficients

% Add the true coefficients (with a 0 for the intercept) for comparison
b_true_full = [0; b_true];

T = table(b_true_full, b_linear, b_ridge, b_lasso, 'VariableNames', ...
    {'True_Coeffs', 'Linear_NoReg', 'Ridge_L2', 'LASSO_L1'});

disp('Comparison of Regression Coefficients:');
disp(T);

% --- Plot the results for a clear visual ---
figure;
subplot(4,1,1);
stem(0:num_features, b_true_full, 'k', 'LineWidth', 2, 'MarkerFaceColor', 'k');
legend('True Coefficients', 'Location', 'northeast');
xlabel('Coefficient Index');ylabel('Coefficient Value');
grid on;
xlim([-1, num_features + 1]);
subplot(4,1,2);
stem(0:num_features, b_linear, 'b', 'LineWidth', 2, 'MarkerFaceColor', 'b');
legend('Linear','Location', 'northeast');
xlabel('Coefficient Index');ylabel('Coefficient Value');
grid on;
xlim([-1, num_features + 1]);
subplot(4,1,3);
stem(0:num_features, b_ridge, 'g', 'LineWidth', 2, 'MarkerFaceColor', 'g');
legend('Ridge', 'Location', 'northeast');
xlabel('Coefficient Index');ylabel('Coefficient Value');
grid on;
xlim([-1, num_features + 1]);
subplot(4,1,4);
stem(0:num_features, b_lasso, 'r', 'LineWidth', 2, 'MarkerFaceColor', 'r');
legend('LASSO', 'Location', 'northeast');
xlabel('Coefficient Index');ylabel('Coefficient Value');
grid on;
xlim([-1, num_features + 1]);
title('Comparison of True vs. Estimated Coefficients');
