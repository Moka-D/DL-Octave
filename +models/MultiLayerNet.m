classdef MultiLayerNet < handle
    %MultiLayerNet 全結合による多層ニューラルネットワーク

    properties
        input_size
        output_size
        hidden_size_list
        hidden_layer_num
        weight_decay_lambda
        params
        layers
        last_layer
    end

    methods
        function self = MultiLayerNet(input_size, hidden_size_list, output_size, varargin)
            % コンストラクタ

            % 引数の検証
            activation_names = {'sigmoid', 'relu'};
            valid_weight_init_std = [activation_names, {'he', 'xavier'}];

            p = inputParser;
            addRequired(p, 'input_size', @isscalar);
            addRequired(p, 'hidden_size_list', @isrow);
            addRequired(p, 'output_size', @isscalar);
            addParameter(p, 'activation', 'relu', ...
                @(x) validatestring(x, activation_names));
            addParameter(p, 'weight_init_std', 'relu', ...
                @(x) isscalar(x) || validatestring(x, valid_weight_init_std));
            addParameter(p, 'weight_decay_lambda', 0, @isscalar);
            parse(p, input_size, hidden_size_list, output_size, varargin{:});

            % プロパティ設定
            self.input_size = p.Results.input_size;
            self.output_size = p.Results.output_size;
            self.hidden_size_list = p.Results.hidden_size_list;
            self.hidden_layer_num = length(self.hidden_size_list);
            self.weight_decay_lambda = p.Results.weight_decay_lambda;
            self.params = containers.Map;

            % 重みの初期化
            self.init_weight(p.Results.weight_init_std);

            % レイヤの生成
            activation_layer = containers.Map({'sigmoid', 'relu'}, ...
                                              {layers.Sigmoid(), layers.Relu()});
            self.layers = struct();
            for idx = 1:self.hidden_layer_num
                self.layers.(strcat('Affine', num2str(idx))) ...
                    = layers.Affine(self.params(strcat('W', num2str(idx))), ...
                                    self.params(strcat('b', num2str(idx))));
                self.layers.(strcat('Activation_function', num2str(idx))) ...
                    = activation_layer(p.Results.activation);
            end

            idx = self.hidden_layer_num + 1;
            self.layers.(strcat('Affine', num2str(idx))) ...
                = layers.Affine(self.params(strcat('W', num2str(idx))), ...
                                self.params(strcat('b', num2str(idx))));

            self.last_layer = layers.SoftmaxWithLoss();
        end

        function y = predict(self, x)

            % Affine層のパラメータ更新
            self.update_weight();

            for key = fieldnames(self.layers)'
                x = self.layers.(key{1}).forward(x);
            end
            y = x;
        end

        function ret = loss(self, x, t)
            % 損失関数を求める

            y = self.predict(x);

            weight_decay = 0;
            for idx = 1:self.hidden_layer_num + 1
                W = self.params(strcat('W', num2str(idx)));
                weight_decay = weight_decay + 0.5 .* self.weight_decay_lambda .* sum(W.^2, 'all');
            end

            ret = self.last_layer.forward(y, t) + weight_decay;
        end

        function ret = accuracy(self, x, t)
            y = self.predict(x);
            [~, y] = max(y, [], 1);
            if ~isvector(t)
                [~, t] = max(t, [], 1);
            end

            ret = sum(y==t, 'all') ./ size(x, ndims(x));
        end

        function grads = numerical_gradient(self, x, t)
            % 数値微分による勾配計算

            loss_W = @(W) self.loss(x, t);

            grads = containers.Map;
            for idx = 1:self.hidden_layer_num + 1
                grads.(strcat('W', num2str(idx))) ...
                    = gradient.numerical_gradient(loss_W, self.params(strcat('W', num2str(idx))));
                grads.(strcat('b', num2str(idx))) ...
                    = gradient.numerical_gradient(loss_W, self.params(strcat('b', num2str(idx))));
            end
        end

        function grads = gradient(self, x, t)
            % 誤差逆伝播法による勾配計算

            % forward
            self.loss(x, t);

            % backward
            dout = 1;
            dout = self.last_layer.backward(dout);
            for key = flip(fieldnames(self.layers)')
                dout = self.layers.(key{1}).backward(dout);
            end

            % 設定
            grads = containers.Map;
            for idx = 1:self.hidden_layer_num + 1
                grads(strcat('W', num2str(idx))) ...
                    = self.layers.(strcat('Affine', num2str(idx))).dW ...
                      + self.weight_decay_lambda .* self.layers.(strcat('Affine', num2str(idx))).W;
                grads(strcat('b', num2str(idx))) = self.layers.(strcat('Affine', num2str(idx))).db;
            end
        end
    end

    methods (Access = private)
        function init_weight(self, weight_init_std)
            % 重みの初期値設定

            all_size_list = cat(2, self.input_size, self.hidden_size_list, self.output_size);

            for idx = 1:length(all_size_list) - 1
                if ischar(weight_init_std) || isstring(weight_init_std)
                    if strcmpi(weight_init_std, 'relu') || strcmpi(weight_init_std, 'he')
                        scale = sqrt(2 / all_size_list(idx));   % ReLUの推奨初期値
                    elseif strcmpi(weight_init_std, 'sigmoid') || strcmpi(weight_init_std, 'xavier')
                        scale = sqrt(1 / all_size_list(idx));   % Sigmoidの推奨初期値
                    end
                else
                    scale = weight_init_std;
                end

                self.params(strcat('W', num2str(idx))) ...
                    = scale .* randn(all_size_list(idx + 1), all_size_list(idx));
                self.params(strcat('b', num2str(idx))) = zeros(all_size_list(idx + 1), 1);
            end
        end

        function update_weight(self)
            % Affine層のパラメータ更新
            for idx = 1:self.hidden_layer_num + 1
                self.layers.(strcat('Affine', num2str(idx))).W ...
                    = self.params(strcat('W', num2str(idx)));
                self.layers.(strcat('Affine', num2str(idx))).b ...
                    = self.params(strcat('b', num2str(idx)));
            end
        end
    end
end
