clear;clc;

%% Parameters
N = 10;                          % Number of masses
k = rand(N-1,1)*10 + 10;         % Random spring constants
m = ones(N,1);                   % Masses
M = diag(m);                     % Mass matrix

% Stiffness matrix
K = zeros(N,N);
for i = 1:N-1
    K(i,i) = K(i,i) + k(i);
    K(i+1,i+1) = K(i+1,i+1) + k(i);
    K(i,i+1) = K(i,i+1) - k(i);
    K(i+1,i) = K(i+1,i) - k(i);
end

%% Fix mass 1
free_idx = 2:N;        % Only keep DOFs 2 to N (mass 1 is fixed)
ndof = length(free_idx);

% Modified mass and stiffness matrices
Kf = K(free_idx, free_idx);
Mf = M(free_idx, free_idx);

% Modified force
Fext = @(t) (t < 0.01) * [100; zeros(N-2,1)];  % Impulse on mass 2

% Initial condition
u0 = zeros(ndof,1);
v0 = zeros(ndof,1);
y0 = [u0; v0];

% ODE system with fixed mass 1 removed
f = @(t,y) [
    y(ndof+1:end);
    Mf \ (Fext(t) - Kf * y(1:ndof))
];

% Time vector
tspan = [0 10];
dt = 0.01;
t_eval = tspan(1):dt:tspan(2);

% Solve
opts = odeset('RelTol',1e-6, 'AbsTol',1e-9);
[t,Y] = ode45(f, t_eval, y0, opts);

% Extract displacements
U = Y(:,1:ndof)';  % rows = DOFs, columns = time
U = [zeros(1,size(U,2));U];

%% Visualize the displacement
figure;
for i = 1:size(U,2)
    plot(1:N,U(:,i),'ko-');
    ylim([-max(max(abs(U))),max(max(abs(U)))]);
    xlabel('Index');ylabel('Displacement');
    title(['Time: ',num2str(t(i))]);
    drawnow;
end

%% POD
[U_pod, S, V] = svd(U, 'econ');

%% Singular value spectrum
figure;
plot(diag(S)/sum(diag(S)),'ko-');
xlabel('#');ylabel('\lambda');
title('Singular Value Spectrum');

%%
r = 5;
Phi = U_pod(:,1:r);
a = S(1:r,1:r) * V(:,1:r)';

%% Plot spatial modes
figure;
for i = 1:r
    subplot(r,1,i);
    plot(1:N, Phi(:,i), 'ko-');
    title(['Spatial Mode ', num2str(i)]);
    ylabel('\phi(x)');
end

%% Plot time coefficients
figure;
for i = 1:r
    subplot(r,1,i);
    plot(t, a(i,:));
    title(['Time Coefficient ', num2str(i)]);
    xlabel('Time (s)');
    ylabel(['a_',num2str(i),'(t)']);
end