function g = quadratic_grad(x)
    % Gradient of quadratic function
    g = zeros(2,1);
    g(1) = x(1);
    g(2) = 20*x(2);
end