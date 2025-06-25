% backtracking_line_search.m
function alpha = backtracking_line_search(f, grad_f, x, p, alpha_init, rho, c)
    % Backtracking line search to find a suitable step size alpha.
    % f: objective function
    % grad_f: gradient of the objective function
    % x: current point
    % p: search direction
    % alpha_init: initial step size
    % rho: backtracking factor (e.g., 0.5)
    % c: Wolfe condition constant (e.g., 1e-4)

    alpha = alpha_init;
    while f(x + alpha * p) > f(x) + c * alpha * grad_f(x)' * p
        alpha = rho * alpha;
    end
end