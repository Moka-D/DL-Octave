function loss = cross_entropy_error(y, t)
    %cross_entropy_error 交差エントロピー誤差関数

    if size(y, 1) == 1 || size(y, 2) == 1
        t = reshape(t, 1, length(t));
        y = reshape(y, 1, length(y));
    end

    batch_size = size(y, 1);

    if size(t, 2) == size(y, 2)
        idx = t == 1;
    else
        idx = repmat(1:size(y, 2), batch_size, 1) .* t > 0;
    end

    tmp = log(y(idx) + 1e-7);
    loss = -sum(tmp(:)) ./ batch_size;
end
