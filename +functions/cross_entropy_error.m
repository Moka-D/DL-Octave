function loss = cross_entropy_error(y, t)
    %cross_entropy_error 交差エントロピー誤差関数

    if ndims(y) == 1
        t = reshape(t, 1, length(t));
        y = reshape(y, 1, length(y));
    end

    % one-hot-labelの場合は変換
    if (size(t, 1) == size(y, 1)) && (size(t, 2) == size(y, 2))
        [~, t] = max(t, [], 2);
    end

    batch_size = size(y, 1);
    tmp = log(y(1:batch_size, t) + 1e-7);
    loss = -sum(tmp(:)) ./ batch_size;
end
