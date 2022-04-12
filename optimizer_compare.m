%% 最適化手法の比較

clear all;
[x_train, t_train, x_test, t_test] = dataset.load_mnist_data(true, true, true);

train_size = size(x_train, 1);
batch_size = 128;
max_iterations = 2000;

% 実験の設定
optimizers = struct();
optimizers.SGD      = optimizer.SGD();
optimizers.Momentum = optimizer.Momentum();
optimizers.AdaGrad  = optimizer.AdaGrad();
optimizers.Adam     = optimizer.Adam();

networks = struct();
train_loss = struct();
optimizer_names = fieldnames(optimizers);
for i_optimizer = 1:length(optimizer_names)
    key = optimizer_names{i_optimizer};
    net = models.TwoLayerNet(784, 50, 10);
    networks = setfield(networks, key, net);
    train_loss = setfield(train_loss, key, zeros(1, max_iterations));
end

% 訓練の開始
for iter = 1:max_iterations
    batch_mask = randperm(train_size, batch_size);
    x_batch = x_train(batch_mask, :);
    t_batch = t_train(batch_mask, :);

    for i_optimizer = 1:length(optimizer_names)
        key = optimizer_names{i_optimizer};
        net = getfield(networks, key);
        optim = getfield(optimizers, key);
        loss_list = getfield(train_loss, key);

        grads = net.gradient(x_batch, t_batch);
        net.params = optim.update(net.params, grads);

        loss_list(1, iter) = net.loss(x_batch, t_batch);
        train_loss = setfield(train_loss, key, loss_list);
        networks = setfield(networks, key, net);
    end

    if mod(iter, 100) == 0
        fprintf('===========iteration:%d===========\n', iter);
        for i_optimizer = 1:length(optimizer_names)
            key = optimizer_names{i_optimizer};
            net = getfield(networks, key);
            loss = net.loss(x_batch, t_batch);
            fprintf('%s:%f\n', key, loss);
        end
    end
end
