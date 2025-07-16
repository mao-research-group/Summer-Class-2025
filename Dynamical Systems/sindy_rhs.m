function dx = sindy_rhs(x, Xi)
    lib = [1; x(1); x(2); x(1)^2; x(1)*x(2); x(2)^2; x(1)^3];
    dx = Xi' * lib;
end