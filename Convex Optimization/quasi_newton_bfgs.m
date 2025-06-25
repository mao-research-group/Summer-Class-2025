% quasi_newton_bfgs.m
function [x, path] = quasi_newton_bfgs(grad_f, x0, max_iter, tol, use_line_search)
    x = x0;
    n = length(x0);
    B = eye(n); % Initialize Hessian approximation as identity matrix
    path = [x];
    g = grad_f(x);

    % Create a contour plot of the Rosenbrock function
    figure('Name','Quasi-Newton Method','Position',[100,100,500,400]);
    [X, Y] = meshgrid(-5:0.05:5, -1.2:0.05:1.2);
    Z = 1/2*X.^2+10*Y.^2;
    contour(X, Y, Z, 20);
    shading interp;
    hold on;
    plot(0,0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
    plot(path(1,:),path(2,:),'m.','MarkerSize',30);
    hold off;
    xlabel('x_{1}');ylabel('x_{2}');
    % pause();

    for i = 1:max_iter
        if norm(g) < tol
            break;
        end
        
        % Solve B*p = -g for search direction
        p = -B \ g;

        alpha = 1.0;
        if use_line_search
            f = @quadratic;
            alpha = backtracking_line_search(f, grad_f, x, p, 1.0, 0.5, 1e-4);
        else
            % Fixed step is not typical for BFGS, but included for demonstration
            alpha = 0.1;
        end

        s = alpha * p;
        x_next = x + s;
        
        g_next = grad_f(x_next);
        y = g_next - g;
        
        % BFGS update
        if y' * s > 1e-9 % Curvature condition
            B = B - (B * s * s' * B) / (s' * B * s) + (y * y') / (y' * s);
        end
        
        x = x_next;
        g = g_next;
        path = [path, x];

        contour(X, Y, Z, 20);
        shading interp;
        hold on;
        plot(0,0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
        plot(path(1,end),path(2,end),'m.','MarkerSize',30);
        plot(path(1,:),path(2,:),'r-','LineWidth',2);
        hold off;
        xlabel('x_{1}');ylabel('x_{2}');
        drawnow;
        % pause(0.5);
    end
end