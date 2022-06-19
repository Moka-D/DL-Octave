function grad = relu_grad(x)
    grad = zeros(size(x));
    grad(x>=0) = 1;
end
