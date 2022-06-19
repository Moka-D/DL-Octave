function y = softmax(x)
    %softmax Softmax関数
    x = x - max(x, [], ndims(x));
    y = exp(x) ./ sum(exp(x), ndims(x));
end
