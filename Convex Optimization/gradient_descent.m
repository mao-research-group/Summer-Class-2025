% gradient_descent.m
function [x, path] = gradient_descent(grad_f, x0, max_iter, tol, alpha_fixed, use_line_search)
    x = x0;
    path = [x];

    % Create a contour plot of the Rosenbrock function
    figure('Name','Gradient Descent','Position',[100,100,500,400]);
    [X, Y] = meshgrid(-5:0.05:5, -1.2:0.05:1.2);
    Z = 1/2*X.^2+10*Y.^2;
    contour(X, Y, Z, 20);
    shading interp;
    hold on;
    plot(0,0,'kp','MarkerSize',15,'MarkerFaceColor','y');
    plot(path(1,:),path(2,:),'m.','MarkerSize',30);
    hold off;
    xlabel('x_{1}');ylabel('x_{2}');
    % pause();

    for i = 1:max_iter
        g = grad_f(x);
        if norm(g) < tol
            break;
        end

        p = -g; % Descent direction

        if use_line_search
            % Note: For GD, the objective function f is needed for line search.
            % We define it anonymously here for simplicity.
            f = @quadratic;
            alpha = backtracking_line_search(f, grad_f, x, p, 1.0, 0.5, 1e-4);
        else
            alpha = alpha_fixed;
        end

        x = x + alpha * p;
        path = [path, x];

        contour(X, Y, Z, 20);
        shading interp;
        hold on;
        plot(0,0,'kp','MarkerSize',15,'MarkerFaceColor','y');
        plot(path(1,end),path(2,end),'m.','MarkerSize',30);
        plot(path(1,:),path(2,:),'r-','LineWidth',2);
        hold off;
        xlabel('x_{1}');ylabel('x_{2}');
        drawnow;
        % pause(0.5);

    end
end