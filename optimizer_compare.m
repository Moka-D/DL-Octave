function optimizer_compare
    %optimizer_compare 最適化手法の比較

    [x_train, t_train, ~, ~] = dataset.load_mnist_data('one_hot_label', true);

    train_size = size(x_train, 2);
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
    for key = fieldnames(optimizers)'
        networks.(key{1}) = models.MultiLayerNet(784, [100 100 100 100], 10);
        train_loss.(key{1}) = nan(1, max_iterations);
    end

    % 訓練の開始
    for iter = 0:max_iterations - 1
        batch_mask = randperm(train_size, batch_size);
        x_batch = x_train(:, batch_mask);
        t_batch = t_train(:, batch_mask);

        for key = fieldnames(optimizers)'
            grads = networks.(key{1}).gradient(x_batch, t_batch);
            optimizers.(key{1}).update(networks.(key{1}).params, grads);
            train_loss.(key{1})(iter+1) = networks.(key{1}).loss(x_batch, t_batch);
        end

        if mod(iter, 100) == 0
            fprintf('===========iteration:%d===========\n', iter);
            for key = fieldnames(optimizers)'
                loss = networks.(key{1}).loss(x_batch, t_batch);
                fprintf('%s:%f\n', key{1}, loss);
            end
        end
    end

    % グラフの描画
    x = 1:max_iterations;
    figure;
    hold on;
    for key = fieldnames(optimizers)'
        plot(x, train_loss.(key{1}), 'DisplayName', key{1});
    end
    xlabel('iterations');
    ylabel('loss');
    ylim([0 1]);
    legend;
end
