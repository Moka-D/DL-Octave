function y = softmax(x)
    %softmax Softmaxä÷êî
    x_tmp = x - max(x, [], 2);
    y = exp(x_tmp) ./ sum(exp(x_tmp), 2);
end
