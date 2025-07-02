clear;clc;

%% 1. Load and Prepare the Face Data
imageSize = [112, 92];
numSubjects = 40;
numImagesPerSubject = 9;
numImages = numSubjects * numImagesPerSubject;

faceMatrix = zeros(prod(imageSize), numImages);
imagePath = 'Faces/';

for i = 1:numSubjects
    for j = 1:numImagesPerSubject
        filePath = fullfile(imagePath, sprintf('s%d', i), sprintf('%d.pgm', j));
        img = imread(filePath);
        subplot(3,3,j);
        imshow(img);
        faceMatrix(:, (i-1)*numImagesPerSubject + j) = double(img(:));
    end
    drawnow;
    % pause(1);
end

%% 2. Calculate the Average Face and Center the Data
averageFace = mean(faceMatrix, 2);
centeredFaces = faceMatrix - averageFace;
figure;
imshow(reshape(averageFace,imageSize)/255);

%% 3. Compute the Eigenfaces using SVD
% The columns of U are the eigenfaces.
[eigenfaces, S, ~] = svd(centeredFaces, 'econ');

%% 4. Singular Value Spectrum and Features
figure;
plot(diag(S)/sum(diag(S)),'ko-');
xlabel('Index');ylabel('\lambda');
title('Singular Value Spectrum');

figure;
for i = 1:16
    subplot(4,4,i);
    imagesc(reshape(eigenfaces(:,i),imageSize));
    axis equal;axis off;
    colormap gray;
end
sgtitle('Top 16 Eigenfaces');

%% 5. Project Faces onto the Eigenface Space
% Each face is represented by its weights in the eigenface space.
weights = eigenfaces' * centeredFaces;

%% 6. Reconstruct Testing Faces
testfaceMatrix = zeros(prod(imageSize), numSubjects);
for i = 1:numSubjects
    filePath = fullfile(imagePath, sprintf('s%d', i), sprintf('%d.pgm', 10));
    img = imread(filePath);
    imshow(img);
    testfaceMatrix(:, i) = double(img(:));
    drawnow;
    % pause(1);
end

ind_testImage = 2;
figure;
subplot(2,3,1);
imshow(reshape(testfaceMatrix(:,ind_testImage),imageSize)/255);
count = 1;
for r = [1,100,200,300,360]
    count = count + 1;
    subplot(2,3,count);
    reconFace = eigenfaces(:,1:r)*eigenfaces(:,1:r)'*testfaceMatrix;
    imshow(reshape(reconFace(:,ind_testImage),imageSize)/255);
end

%% 7. Classification of Faces
person1 = 2;
person2 = 20;
personfaceMatrix = [];count = 1;
for i = [person1,person2]
    for j = 1:10
        filePath = fullfile(imagePath, sprintf('s%d', i), sprintf('%d.pgm', j));
        img = imread(filePath);
        subplot(4,5,count);
        imshow(img);
        personfaceMatrix(:,count) = double(img(:));
        count = count + 1;
    end
    drawnow;
end
PC = [1,2];
ProjFace = eigenfaces(:,PC)'*personfaceMatrix;
figure;
plot(ProjFace(1,1:9),ProjFace(2,1:9),'r*','DisplayName','Person1');
hold on;
plot(ProjFace(1,10:end),ProjFace(2,10:end),'b*','DisplayName','Person2');
legend;
