%% ��w�j���[�����l�b�g���[�N�̊w�K

clear all;
[x_train, t_train, x_test, t_test] = util.loadMNISTdata(true, true, true);
net = multi_layer_net.TwoLayerNet(784, 50, 10);
iters_num = 10000;
train_size = size(x_train, 1);
batch_size = 100;
learning_rate = 0.1;
optim = optimizer.SGD(learning_rate);
iter_per_epoch = max(iters_num / batch_size, 1);

train_loss_list = zeros(1, iters_num);
acc_list_size = round(iters_num / iter_per_epoch);
train_acc_list = zeros(1, acc_list_size);
test_acc_list = zeros(1, acc_list_size);

for iter = 1:iters_num
    batch_mask = randperm(train_size, batch_size);
    x_batch = x_train(batch_mask, :);
    t_batch = t_train(batch_mask, :);

    % �덷�t�`�d�ɂ���Č��z�����߂�
    grads = net.gradient(x_batch, t_batch);

    % �X�V
    net.params = optim.update(net.params, grads);

    loss = net.loss(x_batch, t_batch);
    train_loss_list(1, iter) = loss;

    if mod(iter, iter_per_epoch) == 0
        train_acc = net.accuracy(x_train, t_train);
        test_acc = net.accuracy(x_test, t_test);
        train_acc_list(1, round(iter / iter_per_epoch)) = train_acc;
        test_acc_list(1, round(iter / iter_per_epoch)) = test_acc;
        fprintf('Train Acc:%f, Test Acc:%f\n', train_acc, test_acc);
    end
end

plot(1:acc_list_size, train_acc_list, 1:acc_list_size, test_acc_list);
legend('Train Accuracy', 'Test Accuracy', 'Location', 'southeast');

