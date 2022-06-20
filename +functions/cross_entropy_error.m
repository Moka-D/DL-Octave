function loss = cross_entropy_error(y, t)
    %cross_entropy_error 交差エントロピー誤差関数

    if isvector(y)
        t = reshape(t, numel(t), 1);
        y = reshape(y, numel(y), 1);
    end

    if isequal(size(t), size(y))
        [~, t] = max(t, [], 1);
    end

    batch_size = size(y, ndims(y));
    ind = sub2ind(size(y), t, 1:batch_size);
    loss = -sum(log(y(ind) + 1e-7), 'all') ./ batch_size;
end
