function out = sigmoid(x)
    %sigmoid シグモイド関数
    out = 1 ./ (1 + exp(-x));
end
