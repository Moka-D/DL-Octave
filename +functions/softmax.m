function y = softmax(x)
    %softmax Softmax関数
    x_dim = ndims(x);
    x = x - max(x, [], x_dim - 1);
    y = exp(x) ./ sum(exp(x), x_dim - 1);
end
