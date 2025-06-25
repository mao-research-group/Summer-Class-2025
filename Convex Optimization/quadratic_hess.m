function h = quadratic_hess(x)
    % Hessian of quadratic function
    h = zeros(2,2);
    h(1,1) = 1;
    h(1,2) = 0;
    h(2,1) = 0;
    h(2,2) = 20;
end