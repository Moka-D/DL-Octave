classdef TwoLayerNet < handle
    %TwoLayerNet 2層ニューラルネットワーククラス

    properties
        params      % 各層のパラメータ
        layers      % 各レイヤ
        last_layer  % 最終層の関数ハンドル
    end

    methods
        function obj = TwoLayerNet(input_size, hidden_size, output_size, weight_init_std)
            % コンストラクタ

            % デフォルト引数設定
            if ~exist('weight_init_std', 'var')
                weight_init_std = 0.01;
            end

            % 重みの初期化
            obj.params = struct();
            obj.params.W1 = weight_init_std .* randn(input_size, hidden_size);
            obj.params.b1 = zeros(1, hidden_size);
            obj.params.W2 = weight_init_std .* randn(hidden_size, output_size);
            obj.params.b2 = zeros(1, output_size);

            % レイヤの生成
            obj.layers = struct();
            obj.layers.Affine1 = layers.Affine(obj.params.W1, obj.params.b1);
            obj.layers.Relu1 = layers.Relu();
            obj.layers.Affine2 = layers.Affine(obj.params.W2, obj.params.b2);
            obj.last_layer = layers.SoftmaxWithLoss();
        end

        function y = predict(obj, x)
            % 推論

            obj.layers.Affine1.W = obj.params.W1;
            obj.layers.Affine1.b = obj.params.b1;
            obj.layers.Affine2.W = obj.params.W2;
            obj.layers.Affine2.b = obj.params.b2;

            layer_names = fieldnames(obj.layers);
            for i_layer = 1:length(layer_names)
                layer = getfield(obj.layers, layer_names{i_layer});
                x = layer.forward(x);
            end
            y = x;
        end

        function ret = loss(obj, x, t)
            % 損失
            y = obj.predict(x);
            ret = obj.last_layer.forward(y, t);
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
            dout = obj.last_layer.backward(dout);
            layer_names = fieldnames(obj.layers);
            for i_layer = length(layer_names):-1:1
                layer = getfield(obj.layers, layer_names{i_layer});
                dout = layer.backward(dout);
            end

            % 勾配設定
            grads = struct();
            grads.W1 = obj.layers.Affine1.dW;
            grads.b1 = obj.layers.Affine1.db;
            grads.W2 = obj.layers.Affine2.dW;
            grads.b2 = obj.layers.Affine2.db;
        end
    end
end