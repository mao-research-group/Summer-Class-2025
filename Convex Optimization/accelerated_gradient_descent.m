% accelerated_gradient_descent.m
function [x, path] = accelerated_gradient_descent(grad_f, x0, max_iter, tol, alpha_fixed, use_line_search)
    x = x0;
    y = x0;
    t = 1;
    path = [x];

    % Create a contour plot of the Rosenbrock function
    figure('Name','Accelerated Gradient Descent','Position',[100,100,500,400]);
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
        grad_y = grad_f(y);
        if norm(grad_y) < tol
            break;
        end
        
        p = -grad_y;

        if use_line_search
            f = @quadratic;
            alpha = backtracking_line_search(f, grad_f, y, p, 0.005, 0.5, 1e-4);
        else
            alpha = alpha_fixed;
        end

        x_next = y + alpha * p;

        t_next = (1 + sqrt(1 + 4 * t^2)) / 2;
        y = x_next + ((t - 1) / t_next) * (x_next - x);

        x = x_next;
        t = t_next;
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