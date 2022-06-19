function grad = sigmoid_grad(x)
    grad = (1 - functions.sigmoid(x)) .* functions.sigmoid(x);
end
