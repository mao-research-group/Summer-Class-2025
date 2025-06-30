clear;clc;

%% Read and prepare the image
originalImage = im2double(imread('Messier57.tif'));
[m,n,~] = size(originalImage);
figure;imshow(originalImage);

%% Perform Singular Value Decomposition
originalImage_RGB = [originalImage(:,:,1);originalImage(:,:,2);originalImage(:,:,3)];
[U, S, V] = svd(originalImage_RGB,'econ');

%% Singular Values Spectrum
figure;
singularValues = diag(S);
plot(singularValues/sum(singularValues), 'o-');
set(gca,'YScale','log');
xlabel('Index');
ylabel('Singular Value');
title('Singular Values Spectrum');
grid on;

%% --- Reconstruct with different 'k' values ---
figure;
subplot(2,4,1);
imshow(originalImage);
title('Original');
% Use only the top k singular values
k = [1,10,20,30,40];
for i = 1:length(k)
    reconstructed_image = U(:, 1:k(i)) * S(1:k(i), 1:k(i)) * V(:, 1:k(i))';
    reconstructed_image_k = cat(3, reconstructed_image(1:m,:), reconstructed_image(m+1:2*m,:), reconstructed_image(2*m+1:3*m,:));
    
    % Display the reconstructed image for the current k value
    subplot(2, 4, i+1); imshow(reconstructed_image_k);
    title(['Reconstructed (k=', num2str(k(i)), ')']);
    
    % Error
    error_l2 = norm(originalImage_RGB - reconstructed_image); % l2 norm error
    error_F = norm(originalImage_RGB - reconstructed_image, 'fro'); % Frobenius norm error

    % Display Error
    subplot(2,4,7);
    plot(k(i),error_l2,'ko');
    set(gca,'YScale','log');
    xlabel('k');ylabel('l_2 norm error');
    title('l_2 norm error');
    hold on;grid on;
    subplot(2,4,8);
    plot(k(i), error_F, 'ro');
    set(gca,'YScale','log');
    xlabel('k');ylabel('Frobenius norm error');
    title('Frobenius norm error');
    hold on;grid on;

    drawnow;
end

%% rSVD image compression
clear;clc;

% Read the image and prepare the parameters
originalImage = im2double(imread('Messier57.tif'));
A = [originalImage(:,:,1);originalImage(:,:,2);originalImage(:,:,3)];
[m,n] = size(A);
k = 40;               % target rank
p = 10;               % oversampling

tic;
% Step 1: Random projection
Omega = randn(n, k + p);         % size: [n x (k+p)]

% Step 2: Form sketch Y = A * Omega
Y = A * Omega;                   % size: [m x (k+p)]

% Step 3: Orthonormalize Y to Q
[Q, ~] = qr(Y, 0);               % economy QR, size: [m x (k+p)]

% Step 4: Project A to lower-dim space
B = Q' * A;                      % size: [(k+p) x n]

% Step 5: SVD on small matrix B
[U_tilde, S, V] = svd(B, 'econ');

% Step 6: Final approximation
U = Q * U_tilde;                % size: [m x (k+p)]

% Truncate to rank-k
U_k = U(:, 1:k);
S_k = S(1:k, 1:k);
V_k = V(:, 1:k);

% Reconstructed low-rank approximation
A_rsvd = U_k * S_k * V_k';
rSVD_image_k = cat(3, A_rsvd(1:m/3,:), A_rsvd(m/3+1:2*m/3,:), A_rsvd(2*m/3+1:m,:));
t_rsvd = toc;

% Exact SVD
tic;
[U_exact, S_exact, V_exact] = svd(A, 'econ');
A_exact = U_exact(:,1:k) * S_exact(1:k,1:k) * V_exact(:,1:k)';
SVD_image_k = cat(3, A_exact(1:m/3,:), A_exact(m/3+1:2*m/3,:), A_exact(2*m/3+1:m,:));
t_svd = toc;

% Visualization
figure;
subplot(1,3,1);
imshow(originalImage);
title('Original Image'); 

subplot(1,3,2);
imshow(SVD_image_k);
title(sprintf('SVD Rank-%d Approximation %.2f Seconds', k, t_svd)); 

subplot(1,3,3);
imshow(rSVD_image_k);
title(sprintf('rSVD Rank-%d Approximation %.2f Seconds', k, t_rsvd)); 

