function [L, S] = rpca(D, lambda, tol, max_iter)
% Robust PCA via Inexact ALM (no dependencies)
% Inputs:
%   D         -- data matrix (m x n)
%   lambda    -- weighting parameter (default: 1/sqrt(max(m,n)))
%   tol       -- convergence tolerance (default: 1e-7)
%   max_iter  -- max iterations (default: 500)
%
% Outputs:
%   L         -- low-rank component
%   S         -- sparse component

if nargin < 2, lambda = 1 / sqrt(max(size(D))); end
if nargin < 3, tol = 1e-7; end
if nargin < 4, max_iter = 500; end

[m, n] = size(D);
L = zeros(m, n);
S = zeros(m, n);
Y = zeros(m, n);
mu = 1e-3;

for iter = 1:max_iter
    % Update L using Singular Value Thresholding
    [U, Sigma, V] = svd(D - S + (1/mu)*Y, 'econ');
    Sigma = diag(Sigma);
    thresh = max(Sigma - 1/mu, 0);
    L = U * diag(thresh) * V';

    % Update S using soft thresholding
    T = D - L + (1/mu)*Y;
    S = sign(T) .* max(abs(T) - lambda/mu, 0);

    % Update multiplier
    Z = D - L - S;
    Y = Y + mu * Z;

    % Convergence check
    err = norm(Z, 'fro') / norm(D, 'fro');
    if err < tol
        fprintf('RPCA converged at iteration %d with error %.2e\n', iter, err);
        break;
    end
end
end
