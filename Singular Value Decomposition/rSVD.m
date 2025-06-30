clear;clc;

%% Parameters
m = 100; n = 60;      % matrix dimensions
k = 50;                % target rank
p = 5;                % oversampling
A = randn(m, n);      % random input matrix

%% Step 1: Random projection
Omega = randn(n, k + p);         % size: [n x (k+p)]

%% Step 2: Form sketch Y = A * Omega
Y = A * Omega;                   % size: [m x (k+p)]

%% Step 3: Orthonormalize Y to Q
[Q, ~] = qr(Y, 0);               % economy QR, size: [m x (k+p)]

%% Step 4: Project A to lower-dim space
B = Q' * A;                      % size: [(k+p) x n]

%% Step 5: SVD on small matrix B
[U_tilde, S, V] = svd(B, 'econ');

%% Step 6: Final approximation
U = Q * U_tilde;                % size: [m x (k+p)]

%% Truncate to rank-k
U_k = U(:, 1:k);
S_k = S(1:k, 1:k);
V_k = V(:, 1:k);

%% Reconstructed low-rank approximation
A_rsvd = U_k * S_k * V_k';

%% Compare with Exact SVD
[U_exact, S_exact, V_exact] = svd(A, 'econ');
U_exact_k = U_exact(:, 1:k);
S_exact_k = S_exact(1:k, 1:k);
V_exact_k = V_exact(:, 1:k);
A_exact = U_exact_k * S_exact_k * V_exact_k';

% Error comparison
fprintf('||A - A_exact||_F = %.4e\n', norm(A - A_exact, 'fro'));
fprintf('||A - A_rsvd ||_F = %.4e\n', norm(A - A_rsvd, 'fro'));

%% Visualization of singular values
figure;
plot(diag(S_k)/sum(diag(S_k)),'b-','DisplayName','rSVD');
hold on;
plot(diag(S_exact_k)/sum(diag(S_k)),'r-','DisplayName','SVD');
xlabel('Index');ylabel('\lambda');
legend();

%% Visualization of matrix
figure;
subplot(1,2,1);
imagesc(A); title('Original A'); 
clim([min(min(A)),max(max(A))]);
colormap jet;colorbar;

subplot(1,2,2);
imagesc(A_rsvd); title(sprintf('rSVD Rank-%d Approximation', k)); 
clim([min(min(A)),max(max(A))]);
colormap jet;colorbar;
