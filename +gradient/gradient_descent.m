function grad = gradient_descent(f, init_x, lr, step_num)
    %gradient_descent 勾配降下法

    % デフォルト引数設定
    if ~exist('lr', 'var')
        lr = 0.01;
    end
    if ~exist('step_num', 'var')
        step_num = 100;
    end

    x = init_x;
    for k = 1:step_num
        grad = functions.numerical_gradient(f, x);
        x = x - lr .* grad;
    end
end
