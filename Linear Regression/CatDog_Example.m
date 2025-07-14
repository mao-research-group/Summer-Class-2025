clear;clc;

%% Setup and Image Loading
Path = 'CatDog/';

imageSize = [64, 64]; % Resize images to a consistent small size
num = 50; % 50 cats

% Create the image matrix (each column is one flattened image)
petMatrix = zeros(prod(imageSize), num);
petLabels = strings(num, 1);

ind_flawed = randi([1,num],5,1);
figure;
for i = 1:50
    % Load cats
    img = imread(fullfile(Path, sprintf('cat (%d).jpg', i)));
    img = double(imresize(im2gray(img), imageSize));
    petLabels(i) = "cat";
    % white block flaw
    if ismember(i,ind_flawed)
        patch = ones(imageSize);
        patch(45:55, 22:40) = 0;
        img = img.*patch;
        petLabels(i) = "flawed cat";
    end
    imshow(img/255);
    petMatrix(:,i) = img(:);
    drawnow;
    pause(0.5);
end

%% Apply RPCA
[L, S] = rpca(petMatrix);

%% Display results
figure;
for i = 1:length(ind_flawed)
    img = imread(fullfile(Path, sprintf('cat (%d).jpg', ind_flawed(i))));
    orig = double(imresize(im2gray(img), imageSize));
    clean = reshape(L(:,ind_flawed(i)), imageSize);
    flaw  = reshape(S(:,ind_flawed(i)), imageSize);

    subplot(1,3,1); 
    imshow(orig/255); title('Original Cat');
    subplot(1,3,2); 
    imshow(clean/255); title('Recovered Cat');
    subplot(1,3,3); 
    imshow(abs(flaw)/255); title('Sparse Flaw');
    drawnow;
    pause;
end
