%% BatchNormalizationの評価

clear all;
[x_train, t_train, ~, ~] = dataset.load_mnist_data(true, true, true);

% 学習データを削減
x_train = x_train(1:1000, :);
t_train = t_train(1:1000, :);

max_epochs = 20;
batch_size = 100;
learning_rate = 0.01;


function [train_acc_list, bn_train_acc_list] = train(weight_init_std)
    x_train = evalin('base', 'x_train');
    t_train = evalin('base', 't_train');
    learning_rate = evalin('base', 'learning_rate');
    batch_size = evalin('base', 'batch_size');
    max_epochs = evalin('base', 'max_epochs');

    bn_network = models.MultiLayerNetExtend(784, [100 100 100 100 100], 10, ...
                                            'relu', weight_init_std, 0, ...
                                            false, 0.5, true);
    network = models.MultiLayerNetExtend(784, [100 100 100 100 100], 10, ...
                                        'relu', weight_init_std, 0, ...
                                        false, 0.5, false);
    optim = optimizer.SGD(learning_rate);

    max_iters = 1000000000;
    train_size = size(x_train, 1);
    iter_per_epoch = max(train_size / batch_size, 1);
    epoch_cnt = 1;

    train_acc_list = zeros(1, max_epochs);
    bn_train_acc_list = zeros(1, max_epochs);

    for iter = 1:max_iters
        batch_mask = randperm(train_size, batch_size);
        x_batch = x_train(batch_mask, :);
        t_batch = t_train(batch_mask, :);

        grads = bn_network.gradient(x_batch, t_batch);
        bn_network.params = optim.update(bn_network.params, grads);

        grads = network.gradient(x_batch, t_batch);
        network.params = optim.update(network.params, grads);

        if mod(iter, iter_per_epoch) == 0
            train_acc = network.accuracy(x_train, t_train);
            bn_train_acc = bn_network.accuracy(x_train, t_train);
            train_acc_list(epoch_cnt) = train_acc;
            bn_train_acc_list(epoch_cnt) = bn_train_acc;

            fprintf('epoch: %d | %f - %f\n', epoch_cnt, train_acc, bn_train_acc);

            epoch_cnt = epoch_cnt + 1;
            if epoch_cnt > max_epochs
                break;
            end
        end
    end
end


% グラフの描画
weight_scale_list = logspace(0, -4, 16);
x = 1:max_epochs;

figure;
hold on;
for idx = 1:length(weight_scale_list)
    w = weight_scale_list(idx);

    fprintf('============== %d/16 ==============\n', idx);
    [train_acc_list, bn_train_acc_list] = train(w);

    subplot(4, 4, idx)
    title(strcat('W:', num2str(w)));
    if idx == 16
        plot(x, bn_train_acc_list, 'DisplayName', 'Batch Normalization', ...
             x, train_acc_list, 'DisplayName', 'Normal(without BatchNorm)');
    else
        plot(x, bn_train_acc_list, x, train_acc_list);
    end

    ylim([0 1]);
    if mod(idx, 4) ~= 1
        yticks([]);
    else
        ylabel('accuracy');
    end
    if idx < 13
        xticks([]);
    else
        xlabel('epochs');
    end
end
