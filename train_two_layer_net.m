function train_two_layer_net
    %train_tow_layer_net 二層ニューラルネットワークの学習

    [x_train, t_train, x_test, t_test] = dataset.load_mnist_data('one_hot_label', true);
    net = models.TwoLayerNet(784, 50, 10);
    iters_num = 10000;
    train_size = size(x_train, 2);
    batch_size = 100;
    optim = optimizer.SGD();
    iter_per_epoch = max(iters_num / batch_size, 1);

    train_loss_list = nan(1, iters_num);
    acc_list_size = round(iters_num / iter_per_epoch);
    train_acc_list = nan(1, acc_list_size);
    test_acc_list = nan(1, acc_list_size);

    for iter = 0:iters_num - 1
        batch_mask = randperm(train_size, batch_size);
        x_batch = x_train(:, batch_mask);
        t_batch = t_train(:, batch_mask);

        % 誤差逆伝播によって勾配を求める
        grads = net.gradient(x_batch, t_batch);

        % 更新
        net.params = optim.update(net.params, grads);

        loss = net.loss(x_batch, t_batch);
        train_loss_list(iter + 1) = loss;

        if mod(iter, iter_per_epoch) == 0
            train_acc = net.accuracy(x_train, t_train);
            test_acc = net.accuracy(x_test, t_test);
            train_acc_list(round(iter / iter_per_epoch) + 1) = train_acc;
            test_acc_list(round(iter / iter_per_epoch) + 1) = test_acc;
            fprintf('Train Acc:%f, Test Acc:%f\n', train_acc, test_acc);
        end
    end

    plot(1:acc_list_size, train_acc_list, 1:acc_list_size, test_acc_list);
    legend('Train Accuracy', 'Test Accuracy', 'Location', 'southeast');
end
