function loss = softmax_loss(X, t)
    y = functions.softmax(X);
    loss = functions.cross_entropy_error(y, t);
end
