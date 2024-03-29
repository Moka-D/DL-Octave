function batch_norm_test
    %batch_norm_test BatchNormalizationの評価

    [x_train, t_train, ~, ~] = dataset.load_mnist_data('one_hot_label', true);

    % 学習データを削減
    x_train = x_train(:, 1:1000);
    t_train = t_train(:, 1:1000);

    max_epochs = 20;
    batch_size = 100;
    learning_rate = 0.01;

    function [train_acc_list, bn_train_acc_list] = train(weight_init_std)
        bn_network = models.MultiLayerNetExtend(784, [100 100 100 100 100], 10, ...
                                                'weight_init_std', weight_init_std, ...
                                                'use_batchnorm', true);
        network = models.MultiLayerNetExtend(784, [100 100 100 100 100], 10, ...
                                             'weight_init_std', weight_init_std);
        optim = optimizer.SGD(learning_rate);

        max_iters = 1000000000;
        train_size = size(x_train, 2);
        iter_per_epoch = max(train_size / batch_size, 1);
        epoch_cnt = 0;

        train_acc_list = nan(1, max_epochs);
        bn_train_acc_list = nan(1, max_epochs);

        for iter = 0:max_iters - 1
            batch_mask = randperm(train_size, batch_size);
            x_batch = x_train(:, batch_mask);
            t_batch = t_train(:, batch_mask);

            grads = bn_network.gradient(x_batch, t_batch);
            optim.update(bn_network.params, grads);

            grads = network.gradient(x_batch, t_batch);
            optim.update(network.params, grads);

            if mod(iter, iter_per_epoch) == 0
                train_acc = network.accuracy(x_train, t_train);
                bn_train_acc = bn_network.accuracy(x_train, t_train);
                train_acc_list(epoch_cnt + 1) = train_acc;
                bn_train_acc_list(epoch_cnt + 1) = bn_train_acc;

                fprintf('epoch: %d | %f - %f\n', epoch_cnt, train_acc, bn_train_acc);

                epoch_cnt = epoch_cnt + 1;
                if epoch_cnt >= max_epochs
                    break
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
            h = plot(x, bn_train_acc_list, x, train_acc_list);
            set(h, {'DisplayName'}, {'Batch Normalization'; 'Normal(without BatchNorm)'})
            legend;
        else
            plot(x, bn_train_acc_list, x, train_acc_list);
        end

        ylim([0 1]);
        if mod(idx - 1, 4)
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
end
