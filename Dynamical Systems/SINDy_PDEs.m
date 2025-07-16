clear all; close all; clc;

%% Generate Data (from the Heat Equation)
% Define physical parameters and domain
alpha = 0.5;           % Thermal diffusivity
L = 20;                % Length of the domain
n = 128;               % Number of spatial grid points
x2 = linspace(-L/2, L/2, n+1); 
x = x2(1:n);           % Spatial domain
dx = x(2) - x(1);

% Define time domain
t = 0:0.1:10;          
dt = t(2)-t(1);
nt = length(t);

% Define wavenumbers for spectral differentiation
k = (2*pi/L)*[0:(n/2-1) -n/2:-1]';

% Set the initial condition (a Gaussian pulse)
u0 = exp(-x.^2)';
ut = fft(u0);

% Solve the PDE in Fourier space (u_t = -alpha*k^2*u)
% and transform back to real space at each time step
u_data = zeros(length(x), length(t));
for j = 1:length(t)
    u_sol_k = ut .* exp(-alpha*k.^2*t(j)); % Solution in k-space
    u_data(:,j) = ifft(u_sol_k);          % Inverse FFT to get solution in x-space
end

figure;
pcolor(t, x, real(u_data));
shading interp;
xlabel('Time (t)');
ylabel('Space (x)');
colorbar;
drawnow;

%% Compute time derivatives u_t and spatial derivatives u_x u_xx u_xxx
% Compute time derivative u_t using a finite difference
u_t = zeros(length(x), length(t));
for i = 1:length(x)
    u_t(i,:) = gradient(u_data(i,:), dt);
end

% Compute spatial derivatives using spectral method (FFT) for high accuracy
u = u_data;
ux = ifft(1i*k .* fft(u_data));
uxx = ifft(-k.^2 .* fft(u_data));
uxxx = ifft(-1i*k.^3 .* fft(u_data));

% Reshape all data into long column vectors for the regression
u_t_vec = reshape(u_t, n*nt, 1);
u_vec = reshape(u, n*nt, 1);
ux_vec = reshape(ux, n*nt, 1);
uxx_vec = reshape(uxx, n*nt, 1);
uxxx_vec = reshape(uxxx, n*nt, 1);

%% Build Library
Theta = [ones(n*nt,1), u_vec, u_vec.^2, ux_vec, uxx_vec, uxxx_vec];
labels = {'1', 'u', 'u^2', 'u_x', 'u_{xx}', 'u_{xxx}'};

%% Apply SINDy
lambda = 0.02;  % Sparsification threshold
max_iter = 10;  % Max iterations for STLS

Xi = Theta \ u_t_vec;
for k = 1:10
    small = abs(Xi) < lambda;
    Xi(small) = 0;
    big = ~small;
    Xi(big) = Theta(:,big) \ u_t_vec;
end

%% Display the Results

disp('SINDy Discovered PDE');
disp('-------------------');
disp('Candidate Library Terms:');
disp(labels);

disp('Learned Coefficients (Xi):');
disp(Xi');

disp('Reconstructed Equation:');
equation_str = 'u_t = ';
for i = 1:length(Xi)
    if Xi(i) ~= 0
        equation_str = [equation_str, sprintf('%.4f * %s + ', Xi(i), labels{i})];
    end
end
% Clean up the end of the string
if endsWith(equation_str, '+ ')
    equation_str = equation_str(1:end-2);
end
disp(equation_str);