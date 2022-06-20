classdef TwoLayerNet < handle
    %TwoLayerNet 2層ニューラルネットワーククラス

    properties
        params      % 各層のパラメータ
        layers      % 各レイヤ
        last_layer  % 最終層の関数ハンドル
    end

    methods
        function self = TwoLayerNet(input_size, hidden_size, output_size, weight_init_std)
            % コンストラクタ

            % デフォルト引数設定
            if ~exist('weight_init_std', 'var')
                weight_init_std = 0.01;
            end

            % 重みの初期化
            W1 = weight_init_std .* randn(hidden_size, input_size);
            b1 = zeros(hidden_size, 1);
            W2 = weight_init_std .* randn(output_size, hidden_size);
            b2 = zeros(output_size, 1);
            self.params = struct('W1', W1, 'b1', b1, 'W2', W2, 'b2', b2);

            % レイヤの生成
            self.layers = struct('Affine1', layers.Affine(W1, b1), ...
                                 'Relu1',   layers.Relu(), ...
                                 'Affine2', layers.Affine(W2, b2));
            self.last_layer = layers.SoftmaxWithLoss();
        end

        function y = predict(self, x)
            % 推論

            self.layers.Affine1.W = self.params.W1;
            self.layers.Affine1.b = self.params.b1;
            self.layers.Affine2.W = self.params.W2;
            self.layers.Affine2.b = self.params.b2;

            for layer_name = fieldnames(self.layers)'
                x = self.layers.(layer_name{1}).forward(x);
            end
            y = x;
        end

        function ret = loss(self, x, t)
            % 損失
            y = self.predict(x);
            ret = self.last_layer.forward(y, t);
        end

        function ret = accuracy(self, x, t)
            % 正解率

            y = self.predict(x);
            [~, y] = max(y, [], 1);
            if ~isvector(t)
                [~, t] = max(t, [], 1);
            end

            ret = sum(y==t, 'all') ./ size(x, ndims(x));
        end

        function grads = gradient(self, x, t)
            % 勾配計算

            % 順伝播計算
            self.loss(x, t);

            % 逆伝播計算
            dout = 1;
            dout = self.last_layer.backward(dout);
            for layer_name = flip(fieldnames(self.layers)')
                dout = self.layers.(layer_name{1}).backward(dout);
            end

            % 勾配設定
            grads = struct('W1', self.layers.Affine1.dW, ...
                           'b1', self.layers.Affine1.db, ...
                           'W2', self.layers.Affine2.dW, ...
                           'b2', self.layers.Affine2.db);
        end
    end
end
