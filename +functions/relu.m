function y = relu(x)
    %relu ReLU�֐�
    mask = x < 0;
    y = x;
    y(mask) = 0;
end
