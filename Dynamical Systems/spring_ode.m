%% ODE45 Simulation with Fixed Boundary
% Define system dynamics (last mass fixed)
function dxdt = spring_ode(t, x, m, k, c, N)
    positions = x(1:N);
    velocities = x(N+1:end);
    accelerations = zeros(N,1);
    
    % Force calculations (with fixed boundary condition)
    for i = 1:N
        % Left spring force
        if i == 1
            left_force = -k(1)*positions(1);
        else
            left_force = k(i-1)*(positions(i-1) - positions(i));
        end
        
        % Right spring force (last mass connected to fixed wall)
        if i == N
            right_force = -k(end)*positions(end);
        else
            right_force = k(i)*(positions(i+1) - positions(i));
        end
        
        % Damping force
        damping = -c(i)*velocities(i);
        
        % Total force
        F_total = left_force + right_force + damping;
        accelerations(i) = F_total/m(i);
    end
    
    dxdt = [velocities; accelerations];
end