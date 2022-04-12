function dydx = numerical_diff(f, x)
    %numerical_diff 数値微分関数
    h = 1e-4;   % 0.0001
    dydx = (f(x+h) - f(x-h)) ./ (2 .* h);
end
