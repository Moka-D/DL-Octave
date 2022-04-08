function y = relu(x)
    %relu ReLUä÷êî
    mask = x < 0;
    y = x;
    y(mask) = 0;
end
