% newton_method.m
function [x, path] = newton_method(grad_f, hess_f, x0, max_iter, tol, use_line_search)
    x = x0;
    path = [x];

    % Create a contour plot of the Rosenbrock function
    figure('Name','Newton Method','Position',[100,100,500,400]);
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
        g = grad_f(x);
        H = hess_f(x);

        if norm(g) < tol
            break;
        end

        % Solve the linear system H*p = -g for the Newton direction
        p = -H \ g;
        
        alpha = 1.0; % Default step for pure Newton's method
        if use_line_search
            f = @quadratic;
            alpha = backtracking_line_search(f, grad_f, x, p, 1.0, 0.5, 1e-4);
        end
        
        x = x + alpha * p;
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