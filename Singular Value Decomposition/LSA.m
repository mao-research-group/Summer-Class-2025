clear;clc;

%% 1. Define the Corpus
% A small set of documents with two main topics.
documents = {
    'NASA sends a rover to explore the surface of Mars', ...
    'The James Webb space telescope captures images of distant stars', ...
    'Machine learning in Python makes data analysis simple', ...
    'Data scientists develop new algorithms for data mining'
    };

%% 2. Create the Term-Document Matrix
% Preprocess the text to create a vocabulary and the matrix.
corpus = lower(documents);
corpus = tokenizedDocument(corpus);
vocab = corpus.Vocabulary;

% Create the matrix where A(i, j) is the count of term i in document j.
A = zeros(numel(vocab), numel(documents));

% Populate the matrix with word counts
for j = 1:numel(documents) % Iterate through each document (columns)
    % Get the list of tokens for the current document
    current_doc_tokens = string(corpus(j));
    for i = 1:numel(vocab) % Iterate through each vocabulary word (rows)
        % Count how many times the vocab word appears in the current document
        A(i, j) = sum(strcmp(vocab{i}, current_doc_tokens));
    end
end

%% 3. Apply SVD and Reduce Dimensionality
[U, S, V] = svd(A, 'econ');

% Reduce to k dimensions (concepts).
k = 2; 
Uk = U(:, 1:k);
Sk = S(1:k, 1:k);
Vk = V(:, 1:k);

%% 4. Map Terms and Documents into the Concept Space ---
% The new coordinates for terms and documents are calculated from the truncated matrices.
term_coords = Uk * Sk;      % Coordinates for each term in the concept space
doc_coords = Vk * Sk;       % Coordinates for each document in the concept space

%% 5. Visualize the Results
figure('Position', [100, 100, 1000, 600]); % Create a wider figure

% --- Left Panel: The Scatter Plot ---
subplot(1, 2, 1);
hold on;
grid on;
title('LSA Concept Space');
xlabel('Concept 1: Space Exploration');
ylabel('Concept 2: Data Science');
scatter(term_coords(:,1), term_coords(:,2), 100, 'b', 'o');
scatter(doc_coords(:,1), doc_coords(:,2), 200, 'r', 'x', 'LineWidth', 2);
legend({'Terms', 'Documents'}, 'Location', 'best');
hold off;

% --- Right Panel: The Term List ---
subplot(1, 2, 2);
axis off;
title('Term Coordinates');

header = sprintf('%-15s | Concept 1 | Concept 2', 'Term');
line_sep = '-------------------------------------------';
text(0, 1, header, 'FontName', 'Courier New', 'FontSize', 9, 'FontWeight', 'bold');
text(0, 0.97, line_sep, 'FontName', 'Courier New', 'FontSize', 8);

for i = 1:numel(vocab)
    y_position = 0.97 - (i * 0.03);
    line_text = sprintf('%-15s | %9.4f | %9.4f', vocab{i}, term_coords(i,1), term_coords(i,2));
    text(0, y_position, line_text, 'FontName', 'Courier New', 'FontSize', 8);
end