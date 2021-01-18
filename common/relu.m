function y = relu(x)
    mask = x < 0;
    y = x;
    y(mask) = 0;
end
