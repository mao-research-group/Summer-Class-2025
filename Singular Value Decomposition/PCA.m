clear;clc
%% 1. Generate and Visualize Sample Data
rng('default'); % for reproducibility
data = mvnrnd([2 3], [1 0.8; 0.8 1], 200); % Multivariate normal random numbers

figure(1);
plot(data(:,1), data(:,2), 'b.');
title('Original Data');
xlabel('Feature 1');
ylabel('Feature 2');
axis equal;
grid on;

%% 2. Center the Data
meanData = mean(data);
centeredData = data - meanData;
figure(2);
plot(centeredData(:,1), centeredData(:,2), 'b.');
title('Centered Data');
xlabel('Feature 1');
ylabel('Feature 2');
axis equal;
grid on;

%% 3. Perform SVD
% Apply SVD to the centered data matrix.
% [U, S, V] = svd(X) produces X = U*S*V'
% The columns of V are the principal component directions (eigenvectors).
[U, S, V] = svd(centeredData, 'econ'); 

% The principal axes are the columns of V
principalAxes = V;

%% 4. Project the Data onto the Principal Components
% The projected data (scores) can be calculated directly.
% Scores are the representation of the data in the new PCA space.
scores = centeredData*V;
figure(3);
plot(scores(:,1), scores(:,2), 'b.');
title('Projected Data');
xlabel('Feature 1');
ylabel('Feature 2');
axis equal;
grid on;

%% 5. Visualize the Principal Axes
figure(4);
plot(centeredData(:,1), centeredData(:,2), 'b.');
hold on;
quiver(0, 0, principalAxes(1,1), principalAxes(2,1), 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, principalAxes(1,2), principalAxes(2,2), 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
title('Centered Data with Principal Axes (from SVD)');
xlabel('Centered Feature 1');
ylabel('Centered Feature 2');
legend({'Data', 'Principal Component 1', 'Principal Component 2'});
axis equal;
grid on;
hold off;