classdef SoftmaxWithLoss < handle
    %SoftmaxWithLoss Softmax-with-Lossレイヤクラス

    properties
        loss    % 損失
        y       % softmaxの出力
        t       % 教師データ (one-hot vector)
    end

    methods
        function obj = SoftmaxWithLoss()
            % コンストラクタ
            obj.loss = [];
            obj.y = [];
            obj.t = [];
        end

        function loss = forward(obj, x, t)
            % 順伝播
            obj.t = t;
            obj.y = functions.softmax(x);
            obj.loss = functions.cross_entropy_error(obj.y, obj.t);
            loss = obj.loss;
        end

        function dx = backward(obj, dout)
            % 逆伝播
            if ~exist('dout', 'var')
                dout = 1;
            end

            batch_size = size(obj.t, 1);

            if (size(obj.t, 1) == size(obj.y, 1)) && ...
               (size(obj.t, 2) == size(obj.y, 2))   % one-hot-labelの場合
                dx = (obj.y - obj.t) ./ batch_size;
            else
                dx = obj.y;
                tmp = zeros(size(dx));
                tmp(repmat(1:size(y, 2), batch_size, 1) .* t > 0) = 1;
                dx = (dx - tmp) ./ batch_size;
            end
        end
    end
end
