function loss = cross_entropy_error(y, t)
    %cross_entropy_error 交差エントロピー誤差関数

    if isvector(y)
        t = reshape(t, 1, length(t));
        y = reshape(y, 1, length(y));
    end

    if isequal(size(t), size(y))
        [~, t] = max(t, [], 2);
    end

    batch_size = size(y, 1);
    ind = sub2ind(size(y), 1:batch_size, t.');
    loss = -sum(log(y(ind) + 1e-7), 'all') ./ batch_size;
end
