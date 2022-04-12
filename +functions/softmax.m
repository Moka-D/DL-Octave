function y = softmax(x)
    %softmax Softmax関数
    exp_x = exp(x - max(x, [], 2));
    y = exp_x ./ sum(exp_x, 2);
end
