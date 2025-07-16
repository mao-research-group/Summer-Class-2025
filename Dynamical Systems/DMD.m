clear;clc;

%% Spring-Mass System with Fixed Last Mass
% System parameters
N_masses = 10;            % Number of movable masses (last mass fixed)
m = ones(N_masses, 1);    % Masses [kg]
k = 100*ones(N_masses, 1);% Spring constants [N/m] (between masses)
c = 0.5*ones(N_masses,1); % Damping coefficients [N·s/m]

% Initial conditions: displace first mass
x0 = [zeros(N_masses, 1); 1; zeros(N_masses-1, 1)]; % [positions; velocities]

% Time parameters
dt = 0.01;                 % Time step [s]
tspan = 0:dt:10;            % Simulation time

% Run ODE45 simulation
options = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t_sim, X] = ode45(@(t,x) spring_ode(t,x,m,k,c,N_masses), tspan, x0, options);
X = X'; % Transpose to [state × time]

figure;
subplot(2,1,1);
imagesc(t_sim,1:N_masses,X(1:N_masses,:));
set(gca,'YDir','normal');
colorbar;
xlabel('t');ylabel('N');
title('Displacement');
subplot(2,1,2);
imagesc(t_sim,1:N_masses,X(N_masses+1:end,:));
set(gca,'YDir','normal');
colorbar;
xlabel('t');ylabel('N');
title('Velocity');

%% DMD Analysis
X_train = X(:,1:round(size(X,2)/2));
X1 = X_train(:, 1:end-1);   % Snapshot matrix X
X2 = X_train(:, 2:end);     % Time-shifted snapshot matrix X'

% SVD-based DMD
[U, S, V] = svd(X1, 'econ');
energy = cumsum(diag(S))/sum(diag(S));
r = find(energy > 0.99, 1);  % 99% energy truncation

Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

% DMD operator and modes
A_tilde = Ur'*X2*Vr/Sr;
[W, Lambda] = eig(A_tilde);
Phi = X2*Vr/Sr*W;      % DMD modes
b = Phi\X(:,1);        % Initial amplitudes

% DMD reconstruction and prediction
t_pred = 0:dt:10;      % Extended prediction time
X_dmd = zeros(r, length(t_pred));
for k = 1:length(t_pred)
    X_dmd(:,k) = b.*(diag(Lambda).^(k-1));
end
X_pred = Phi * X_dmd;  % Full state prediction

%% Vibration Analysis
% Eigenvalue Spectrum (Damped vibrations)
figure;
plot(exp(1i*linspace(0,2*pi,200)),'k--');
hold on;
scatter(real(diag(Lambda)), imag(diag(Lambda)), 20, 'filled');
axis square;
title('DMD Eigenvalues');
xlabel('Re(\lambda)'), ylabel('Im(\lambda)');
axis equal;

% Natural Frequencies and Damping Ratio
omega_n = abs(log(diag(Lambda)))/dt;       % Angular frequencies [rad/s]
freq_Hz = omega_n/(2*pi);                  % Frequency [Hz]
zeta = -real(log(diag(Lambda)))./omega_n;  % Damping ratios

figure; 
subplot(2,3,1);
[~,idx] = sort(freq_Hz);
bar(freq_Hz(idx));
xlabel('Mode Number'), ylabel('Frequency (Hz)');
grid on;
subplot(2,3,4);
bar(zeta(idx));
xlabel('Mode Number'), ylabel('Damping Ratio');
grid on;

% Mode Shapes (First 3 modes)
subplot(2,3,[2,3,5,6]);
plot(1:N_masses, real(Phi(1:N_masses, idx(1))),'o-','LineWidth',2,'MarkerSize',10);
hold on;
plot(1:N_masses, real(Phi(1:N_masses, idx(3))),'^-','LineWidth',2,'MarkerSize',10);
plot(1:N_masses, real(Phi(1:N_masses, idx(5))),'s-','LineWidth',2,'MarkerSize',10);
title('Spatial Mode Shapes')
xlabel('Mass Position'), ylabel('Displacement Amplitude')
legend('Mode 1', 'Mode 2', 'Mode 3'), grid on

% 4. Time Response (First mass)
figure;
for i = 1:N_masses
    subplot(2,5,i)
    plot(t_sim, X(i,:), 'b-', 'LineWidth', 2);
    hold on;
    plot(t_pred, real(X_pred(i,:)), 'r--', 'LineWidth', 1)
    xline(max(t_sim)/2, 'k--', 'LineWidth', 1.5);
    title(['mass ',num2str(i)]);
    grid on;
end