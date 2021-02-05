function y = softmax(x)
    %softmax Softmaxä÷êî
    exp_x = exp(x - max(x, [], 2));
    y = exp_x ./ sum(exp_x, 2);
end
