function [x, t] = get_spiral(train)
    if ~exist('train', 'var')
        train = true;
    end

    if train
        seed = 1984;
    else
        seed = 2020;
    end
    rng(seed);

    num_data = 100;
    num_class = 3;
    input_dim = 2;
    data_size = num_class * num_data;
    x = zeros(input_dim, data_size);
    t = zeros(1, data_size, 'int32');

    for c_i = 0:num_class - 1
        for d_i = 0:num_data - 1
            rate = d_i / num_data;
            radius = 1 * rate;
            theta = c_i * 4 + 4 * rate + randn() * 0.2;
            ix = num_data * c_i + d_i + 1;
            x(:, ix) = [radius * sin(theta); radius * cos(theta)];
            t(ix) = c_i;
        end
    end

    % Shuffle
    indices = randperm(data_size);
    x = x(:, indices);
    t = t(indices);
end
