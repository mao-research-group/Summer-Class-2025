clear;clc;

%% 1. Setup and Parameters
L = 2; % Period of the wave
x = linspace(-L, L, 1000); % Create a high-resolution x-axis

% Define the ideal square wave using the sign() function
% This creates a wave that is -1 for x<0 and +1 for x>0.
f_ideal = sign(sin(2*pi*(1/L)*x));
f_ideal(f_ideal == 0) = 1; % Handle the zero-crossing points

figure;
plot(x,f_ideal,'k-', 'LineWidth', 2);
ylim([-2,2]);
xlabel('x');ylabel('y');

%% 2. Calculate and Plot the Series
% Number of terms to use in the Fourier series approximation
N_terms = 1:1:30; % Use odd numbers for a square wave

figure('Name', 'Fourier Series of a Square Wave');
hold on;

% Plot the ideal square wave for reference
plot(x, f_ideal, 'k--', 'LineWidth', 2);
xlabel('x');ylabel('y');

% Initialize the Fourier series sum
f_series = zeros(size(x));

% Loop through harmonics and add them to the series
b_n = 0; % Initialize coefficient
c = cool(length(N_terms));
for i = 1:length(N_terms)
    n = N_terms(i);
    % Coefficients are non-zero only for odd n
    if mod(n, 2) ~= 0
        % Calculate the b_n coefficient for the square wave
        b_n = 4 / (n * pi);

        % Add the current sine term to the total series
        term = b_n * sin(n * pi * x);
        f_series = f_series + term;

        plot(x, f_series, 'color', c(n,:), 'LineWidth', 1.5);
        title(['n=',num2str(n)]);
        drawnow;
        pause;
    end
end
