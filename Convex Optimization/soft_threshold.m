function x_thresh = soft_threshold(x, tau)
    x_thresh = sign(x) .* max(abs(x) - tau, 0);
end
