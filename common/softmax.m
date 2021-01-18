function out = softmax(a)
    %softmax Softmax関数
    c = max(a);
    exp_a = exp(a - c); % オーバーフロー対策
    out = exp_a ./ sum(exp_a);
end
