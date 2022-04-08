function y = softmax(x)
    %softmax Softmax�֐�
    exp_x = exp(x - max(x, [], 2));
    y = exp_x ./ sum(exp_x, 2);
end
