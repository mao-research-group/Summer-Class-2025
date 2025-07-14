clear;clc;

%% Load and preprocess image
img = imread('cameraman.tif');
img = imresize(img, [48 48]);
img = double(img) / 255;
figure;
imshow(img);

[n1, n2] = size(img);
N = n1 * n2;
x = img(:);

%% Define sampling (random pixel sampling)
m = round(0.3 * N); % 30% measurements
sample_idx = sort(randperm(N, m)); % indices of sampled pixels

% Measurement matrix as subsampled identity
A = eye(N);
A = A(sample_idx, :);
y = A * x;

% Compressed sensing recovery
F = dftmtx(N)/sqrt(N);
Phi = A * F';
cvx_begin quiet
    variable x_hat(N) complex
    minimize(norm(x_hat, 1))
    subject to
        Phi * x_hat == y;
cvx_end
x_cs = real(F' * x_hat);
img_cs = reshape(x_cs, [n1, n2]);

% Direct recovery
x_direct = real(Phi' * ((Phi * Phi') \ y));
img_direct = reshape(x_direct, [n1, n2]);

% Plot sampled points
[row_idx, col_idx] = ind2sub([n1 n2], sample_idx);

figure;
subplot(1,3,1)
imshow(img); title('Original Image'); 
hold on;
plot(col_idx, row_idx, 'r.', 'MarkerSize', 6);

subplot(1,3,2)
imshow(img_cs); title('CS Reconstructed Image');

subplot(1,3,3)
imshow(img_direct); title('Direct Recovery');

