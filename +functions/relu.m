function y = relu(x)
    %relu ReLU関数
    mask = x < 0;
    y = x;
    y(mask) = 0;
end
