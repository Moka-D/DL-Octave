function y = softmax(x)
    %softmax Softmaxä÷êî
    x = x - max(x, [], 2);
    y = exp(x) ./ sum(x, 2);
end
