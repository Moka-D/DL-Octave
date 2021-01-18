function loss = cross_entropy_error(y, t)
    %cross_entropy_error 交差エントロピー誤差関数
    if ndims(y) == 1
        t = reshape(t, 1, length(t));
        y = reshape(y, 1, length(y));
    end

    batch_size = size(y, 1);
    loss = -sum(t .* log(y + 1e-7)) ./ batch_size;
end
