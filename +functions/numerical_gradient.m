function grad = numerical_gradient(f, x)
    %numerical_gradient 数値勾配計算関数

    h = 1e-4;   % 0.0001
    grad = zeros(size(x));  % xと同じ形状の0行列を作成

    for idx = 1:numel(x)
        tmp = x(idx);
        % f(x+h)の計算
        x(idx) = tmp + h;
        fxh1 = f(x);

        % f(x-h)の計算
        x(idx) = tmp - h;
        fxh2 = f(x);

        grad(idx) = (fxh1 - fxh2) ./ (2 .* h);
        x(idx) = tmp;
    end
end
