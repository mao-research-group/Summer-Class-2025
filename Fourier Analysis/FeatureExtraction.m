clear; clc; close all;

%% Step 1: Generate Synthetic Sensor Data
Fs = 1000;              % Sampling frequency in Hz
t = 0:1/Fs:1-1/Fs;      % Time vector for 1 second of data
numSamples = 100;       % 50 normal, 50 faulty
data = cell(numSamples, 2); % {Signal, Label}

% Generate 50 'Normal' signals
for i = 1:numSamples/2
    normalSignal = 2.5 * sin(2*pi*60*t) + 0.5 * randn(size(t)); % 60 Hz + noise
    data{i, 1} = normalSignal;
    data{i, 2} = 'Normal';
end

% Generate 50 'Faulty' signals
for i = (numSamples/2 + 1):numSamples
    faultySignal = 2.5 * sin(2*pi*60*t) + 1 * sin(2*pi*120*t) + 0.5 * randn(size(t)); % 60 Hz + 180 Hz fault + noise
    data{i, 1} = faultySignal;
    data{i, 2} = 'Faulty';
end

%% Visualize one of each type in time domain
figure('Name', 'Time Domain Signals');
subplot(2,1,1);
plot(t, data{1,1});
title('Sample "Normal" Signal');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,1,2);
plot(t, data{51,1});
title('Sample "Faulty" Signal');
xlabel('Time (s)'); ylabel('Amplitude');

%% Step 2: Feature Extraction using FFT

% The key feature will be the maximum amplitude in the "fault" frequency band
features = zeros(numSamples, 1);
labels = categorical(data(:,2)); % Use categorical labels for classification

for i = 1:numSamples
    signal = data{i,1};
    
    % Compute the FFT
    Y = fft(signal);
    L = length(signal);
    
    % Compute the single-sided amplitude spectrum
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    
    % Extract the feature: max amplitude in the fault region
    fault_indices = f >= 100 & f <= 210;
    features(i) = max(P1(fault_indices));
end

%% Visualize the features for the two classes
figure('Name', 'Extracted Features');
gscatter(features, ones(size(features)), labels, 'br', 'o');
title('Feature Space');
xlabel('Max Amplitude in Fault Band (100-210 Hz)');
legend('Normal', 'Faulty');
yticks([]);