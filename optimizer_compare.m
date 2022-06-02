function optimizer_compare
    %optimizer_compare 最適化手法の比較

    [x_train, t_train, ~, ~] = dataset.load_mnist_data(true, true, true);

    train_size = size(x_train, 1);
    batch_size = 128;
    max_iterations = 2000;

    % 実験の設定
    optimizers = struct( ...
        'SGD',      optimizer.SGD(), ...
        'Momentum', optimizer.Momentum(), ...
        'AdaGrad',  optimizer.AdaGrad(), ...
        'Adam',     optimizer.Adam() ...
    );

    networks = struct();
    train_loss = struct();
    keys = fieldnames(optimizers);
    for idx = 1:length(keys)
        key = keys{idx};
        networks.(key) = models.MultiLayerNet(784, [100 100 100 100], 10);
        train_loss.(key) = zeros(1, max_iterations);
    end

    % 訓練の開始
    for iter = 1:max_iterations
        batch_mask = randperm(train_size, batch_size);
        x_batch = x_train(batch_mask, :);
        t_batch = t_train(batch_mask, :);

        for idx = 1:length(keys)
            key = keys{idx};
            grads = networks.(key).gradient(x_batch, t_batch);
            networks.(key).params = optimizers.(key).update(networks.(key).params, grads);

            loss = networks.(key).loss(x_batch, t_batch);
            train_loss.(key)(1, iter) = loss;
        end

        if mod(iter, 100) == 0
            fprintf('===========iteration:%d===========\n', iter);
            for idx = 1:length(keys)
                key = keys{idx};
                loss = networks.(key).loss(x_batch, t_batch);
                fprintf('%s:%f\n', key, loss);
            end
        end
    end

    % グラフの描画
    x = 1:max_iterations;
    figure;
    hold on;
    for idx = 1:length(keys)
        key = keys{idx};
        plot(x, util.smooth_curve(train_loss.(key)), 'DisplayName', key);
    end
    xlabel('iterations');
    ylabel('loss');
    ylim([0 1]);
    legend;
end
