clear;clc;

%% 1. Setup and Image Loading
Path = 'CatDog/';

imageSize = [64, 64]; % Resize images to a consistent small size
numTrain = 90; % 45 cats + 45 dogs
numTest = 20;   % 10 cats + 10 dogs

% Create the image matrix (each column is one flattened image)
trainMatrix = zeros(prod(imageSize), numTrain);
trainLabels = strings(numTrain, 1);

figure;
for i = 1:45
    % Load cats
    img = imread(fullfile(Path, sprintf('cat (%d).jpg', i)));
    subplot(1,2,1);
    imshow(img);
    img = imresize(im2gray(img), imageSize);
    trainMatrix(:, i) = double(img(:));
    trainLabels(i) = "cat";
    
    % Load dogs
    img = imread(fullfile(Path, sprintf('dog (%d).jpg', i)));
    subplot(1,2,2);
    imshow(img);
    img = imresize(im2gray(img), imageSize);
    trainMatrix(:, i+45) = double(img(:));
    trainLabels(i+45) = "dog";
    drawnow;
    pause(1);
end

%% 2. Perform PCA using SVD
% Calculate the mean image and center the data
meanImage = mean(trainMatrix, 2);
centeredImages = trainMatrix - meanImage;

% Use SVD to find the principal components ("eigen-pets")
% The columns of U are the principal components
[U, S, V] = svd(centeredImages, 'econ');

% Select a subset of principal components (e.g., the first 80)
numComponents = 80;
featureSpace = U(:, 1:numComponents);

%% 3. Singular Value Spectrum and Features
figure;
plot(diag(S)/sum(diag(S)),'ko-');
xlabel('Index');ylabel('\lambda');
title('Singular Value Spectrum');

figure;
for i = 1:12
    subplot(3,4,i);
    imagesc(reshape(featureSpace(:,i),imageSize));
    axis equal;axis off;
    colormap gray;
end
sgtitle('Top 12 Eigen-Pets');

%% 4. Extract Features for the Classifier
% Project the centered training data onto the feature space
trainFeatures = featureSpace' * centeredImages;

%% 5. Train a Classifier
% We use the features (not the raw pixels) to train the model
svmModel = fitcsvm(trainFeatures', trainLabels);

%% 6. Test the Classifier on New Images
% Load and process test images
testMatrix = zeros(prod(imageSize), numTest);
testLabels = strings(numTest, 1);
figure;
for i = 1:10
    % Load cats
    img = imread(fullfile(Path, sprintf('cat (%d).jpg', 45+i)));
    subplot(1,2,1);
    imshow(img);
    img = imresize(im2gray(img), imageSize);
    testMatrix(:, i) = double(img(:));
    testLabels(i) = "cat";

    % Load dogs
    img = imread(fullfile(Path, sprintf('dog (%d).jpg', 45+i)));
    subplot(1,2,2);
    imshow(img);
    img = imresize(im2gray(img), imageSize);
    testMatrix(:, i+10) = double(img(:));
    testLabels(i+10) = "dog";
    drawnow;
    pause(1);
end

% Project the test images onto the same feature space
centeredTestImages = testMatrix - meanImage;
testFeatures = featureSpace' * centeredTestImages;

% Use the trained SVM to predict labels
predictedLabels = predict(svmModel, testFeatures');

%% 7. Display Results
correctPredictions = sum(predictedLabels == testLabels);
accuracy = correctPredictions / numTest * 100;
fprintf('\nClassification Accuracy: %.2f%%\n', accuracy);