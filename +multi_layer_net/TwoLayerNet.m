classdef TwoLayerNet < handle
    %TwoLayerNet 2層ニューラルネットワーククラス

    properties
        params = struct('W1', {}, 'b1', {}, 'W2', {}, 'b2', {})     % 各層のパラメータ
        layers = struct('Affine1', {}, 'Relu1', {}, 'Affine2', {})  % 各レイヤ
        lastLayer   % 最終層の関数ハンドル
    end

    methods
        function obj = TwoLayerNet(input_size, hidden_size, output_size, weight_init_std)
            % コンストラクタ

            % デフォルト引数設定
            if ~exist('weight_init_std', 'var')
                weight_init_std = 0.01;
            end

            % 重みの初期化
            obj.params(1).W1 = weight_init_std .* randn(input_size, hidden_size);
            obj.params(1).b1 = zeros(1, hidden_size);
            obj.params(1).W2 = weight_init_std .* randn(hidden_size, output_size);
            obj.params(1).b2 = zeros(1, output_size);

            % レイヤの生成
            obj.layers(1).Affine1 = layers.Affine(obj.params.W1, obj.params.b1);
            obj.layers(1).Relu1 = layers.Relu();
            obj.layers(1).Affine2 = layers.Affine(obj.params.W2, obj.params.b2);
            obj.lastLayer = layers.SoftmaxWithLoss();
        end

        function y = predict(obj, x)
            % 推論

            obj.layers(1).Affine1.W = obj.params(1).W1;
            obj.layers(1).Affine1.b = obj.params(1).b1;
            obj.layers(1).Affine2.W = obj.params(1).W2;
            obj.layers(1).Affine2.b = obj.params(1).b2;

            names = fieldnames(obj.layers);
            for i = 1:length(names)
                layer = getfield(obj.layers(1), names{i});
                x = layer.forward(x);
            end
            y = x;
        end

        function ret = loss(obj, x, t)
            % 損失
            y = obj.predict(x);
            ret = obj.lastLayer.forward(y, t);
        end

        function ret = accuracy(obj, x, t)
            % 正解率

            y = obj.predict(x);
            [~, y] = max(y, [], 2);
            if size(t, 2) ~= 1
                [~, t] = max(t, [], 2);
            end

            ret = sum(y == t) / size(x, 1);
        end

        function grads = gradient(obj, x, t)
            % 勾配計算

            % 順伝播計算
            obj.loss(x, t);

            % 逆伝播計算
            dout = 1;
            dout = obj.lastLayer.backward(dout);
            names = fieldnames(obj.layers);
            for i = length(names):-1:1
                layer = getfield(obj.layers(1), names{i});
                dout = layer.backward(dout);
            end

            % 勾配設定
            grads = struct('W1', {}, 'b1', {}, 'W2', {}, 'b2', {});
            grads(1).W1 = obj.layers(1).Affine1.dW;
            grads(1).b1 = obj.layers(1).Affine1.db;
            grads(1).W2 = obj.layers(1).Affine2.dW;
            grads(1).b2 = obj.layers(1).Affine2.db;
        end
    end
end
