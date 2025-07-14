clc; clear; close all;

%%
% Simulate high-frequency sparse signal
fs = 80;                   % Sampling rate
n = 128;
t = (0:n-1)' / fs;         % Time vector

% Frequencies beyond fs/2 = 40 Hz
freqs = [40, 50];          % > Nyquist frequency
x = sin(2*pi*freqs(1)*t) + 0.8*sin(2*pi*freqs(2)*t);

% Make it length n, then take DFT
x_freq = fft(x)/n;
x_freq(abs(x_freq) < 1e-2) = 0;   % Keep only strong components
x_time = real(ifft(x_freq * n)); % Clean time-domain signal

% Sample m < n random time points
m = 40;
sample_idx = sort(randperm(n, m));
A = eye(n); 
A = A(sample_idx,:);
y = A * x_time;

figure; 
plot(t, x_time, 'b'); 
hold on;
plot(t(sample_idx), y, 'ro');
ylim([-1,1]);

%% Measurement matrix in frequency domain
F = dftmtx(n)/sqrt(n);
Phi = A * F';

%% Compressed sensing reconstruction
cvx_begin
    variable x_hat(n) complex
    minimize(norm(x_hat,1))
    subject to
        Phi * x_hat == y;
cvx_end

x_rec = real(F' * x_hat);

%% Direct recovery
x_direct = real(Phi' * ((Phi * Phi') \ y));

%% Plot
figure;
subplot(3,1,1);
plot(t, x_time, 'b-','LineWidth',2); hold on;
plot(t(sample_idx), x_time(sample_idx), 'ro','MarkerSize',10);
ylim([-1,1]);
title('Sampling of High Frequency Signal');

subplot(3,1,2);
plot(t, x_time, 'b-','LineWidth',2); hold on;
plot(t, x_rec, 'r-','LineWidth',1);
ylim([-1,1]);
title('Reconstruction via CS');

subplot(3,1,3);
plot(t, x_time, 'b-','LineWidth',2); hold on;
plot(t, x_direct, 'r-','LineWidth',1);
ylim([-1,1]);
title('Reconstruction via normal equation');
