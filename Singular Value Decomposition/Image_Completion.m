clear;clc;

%% 1. Load and prepare the image
M_orig = imread('Messier57.tif');
M_orig = imresize(M_orig,0.2);
M_orig = im2double(M_orig);

% Separate the image into its Red, Green, and Blue channels.
R_orig = M_orig(:,:,1);
G_orig = M_orig(:,:,2);
B_orig = M_orig(:,:,3);

figure;imshow(M_orig);

fprintf('Original image size: %d x %d x %d\n', size(M_orig,1), size(M_orig,2), size(M_orig,3));

%% 2. Corrupt the image by removing a block
% Define the region to remove. Let's remove one of the peppers.
row_start = 300; row_end = 320;
col_start = 300; col_end = 320;

% Create a mask of the missing area.
missing_mask = false(size(R_orig));
missing_mask(row_start:row_end, col_start:col_end) = true;

% Apply the mask to each channel.
R_incomplete = R_orig; R_incomplete(missing_mask) = NaN;
G_incomplete = G_orig; G_incomplete(missing_mask) = NaN;
B_incomplete = B_orig; B_incomplete(missing_mask) = NaN;

% Create a visible incomplete image for display purposes.
M_incomplete = M_orig;
M_incomplete(repmat(missing_mask, [1, 1, 3])) = 0.5; % Make the block gray
figure;imshow(M_incomplete);

fprintf('Removed a %dx%d block of pixels.\n\n', row_end-row_start+1, col_end-col_start+1);

%% 3. Matrix completion (image inpainting) for each channel
% --- Parameters ---
% We can use the same rank for all channels, or tune them individually.
% We'll use the same for simplicity.
target_rank = 100;
num_iterations = 10;

fprintf('Approximating each channel with rank = %d\n', target_rank);

% Store channels in a cell array to loop over them easily.
channels_incomplete = {R_incomplete, G_incomplete, B_incomplete};
channels_reconstructed = cell(1, 3);

for k = 1:3 % Loop over R, G, B
    % Get the current channel
    X_channel = channels_incomplete{k};
    
    % Initialize with a simple guess
    X_filled = X_channel;
    X_filled(isnan(X_filled)) = 0.5; 
    
    % Get the indices of the missing values for the update step
    missing_indices = isnan(X_channel);

    % --- Iterative SVD loop ---
    for i = 1:num_iterations
        [U, S, V] = svd(X_filled, 'econ');
        S_approx = S;
        S_approx(target_rank+1:end, target_rank+1:end) = 0;
        X_approx = U * S_approx * V';
        X_filled(missing_indices) = X_approx(missing_indices);
    end
    
    % Store the completed channel
    channels_reconstructed{k} = X_filled;
end

%% 4. Combine channels and display results
% Combine the reconstructed channels back into a single color image.
M_reconstructed = cat(3, channels_reconstructed{1}, ...
                         channels_reconstructed{2}, ...
                         channels_reconstructed{3});

% Clip values to be in the valid [0, 1] range.
M_reconstructed = max(0, min(1, M_reconstructed));

% Calculate the RMSE on the missing block to quantify the result
error = M_orig(repmat(missing_mask,[1 1 3])) - M_reconstructed(repmat(missing_mask,[1 1 3]));
rmse = sqrt(mean(error.^2));
fprintf('\nOverall Reconstruction RMSE on the missing block: %f\n', rmse);

%% Display the original, corrupted, and reconstructed images.
figure;
sgtitle('Color Image Inpainting using Low-Rank Matrix Completion');

subplot(1, 3, 1);
imshow(M_orig);
title('Original Image');

subplot(1, 3, 2);
imshow(M_incomplete);
title('Corrupted Image');

subplot(1, 3, 3);
imshow(M_reconstructed);
title(sprintf('Reconstructed (Rank %d)', target_rank));