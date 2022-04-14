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
        function obj = MultiLayerNet(input_size, hidden_size_list, output_size, ...
                                     activation, weight_init_std, weight_decay_lambda)
            % コンストラクタ

            % デフォルト引数の設定
            if ~exist('activation', 'var')
                activation = 'relu';
            end
            if ~exist('weight_init_std', 'var')
                weight_init_std = 'relu';
            end
            if ~exist('weight_decay_lambda', 'var')
                weight_decay_lambda = 0;
            end

            % プロパティ設定
            obj.input_size = input_size;
            obj.output_size = output_size;
            obj.hidden_size_list = hidden_size_list;
            obj.hidden_layer_num = length(hidden_size_list);
            obj.weight_decay_lambda = weight_decay_lambda;
            obj.params = struct();

            % 重みの初期化
            obj.init_weight(weight_init_std);

            % レイヤの生成
            obj.layers = struct();
            for idx = 1:obj.hidden_layer_num
                obj.layers.(strcat('Affine', num2str(idx))) = layers.Affine(obj.params.(strcat('W', num2str(idx))), ...
                                                                            obj.params.(strcat('b', num2str(idx))));
                if strcmp(activation, 'relu')
                    activation_layer = layers.Relu();
                elseif strcmp(activation, 'sigmoid')
                    activation_layer = layers.Sigmoid();
                else
                    error('Argument ''activation'' must be ''relu'' or ''sigmoid''');
                end
                obj.layers.(strcat('Activation_function', num2str(idx))) = activation_layer;
            end

            idx = obj.hidden_layer_num + 1;
            obj.layers.(strcat('Affine', num2str(idx))) = layers.Affine(obj.params.(strcat('W', num2str(idx))), ...
                                                                        obj.params.(strcat('b', num2str(idx))));

            obj.last_layer = layers.SoftmaxWithLoss();
        end

        function y = predict(obj, x)
        end

        function ret = loss(obj, x, t)
            % 損失関数を求める

            y = obj.predict(x);

            weight_decay = 0;
            for idx = 1:obj.hidden_layer_num+1
                W_2 = obj.params.(strcat('W', num2str(idx))).^2;
                weight_decay = weight_decay + 0.5 .* obj.weight_decay_lambda .* sum(W_2(:));
            end

            ret = obj.last_layer.forward(y, t) + weight_decay;
        end

        function ret = accuracy(obj, x, t)
        end

        function grads = numerical_gradient(obj, x, y)
            % 数値微分による勾配計算
        end

        function grads = gradient(obj, x, t)
            % 誤差逆伝播法による勾配計算
        end
    end

    methods (Access = private, Hidden = true)
        function init_weight(obj, weight_init_std)
            % 重みの初期値設定

            all_size_list = cat(2, obj.input_size, obj.hidden_size_list, obj.output_size);

            for idx = 1:length(all_size_list)-1
                if ischar(weight_init_std)
                    if strcmp(lower(weight_init_std), 'relu') || ...
                        strcmp(lower(weight_init_std), 'he')
                        scale = sqrt(2 / all_size_list(idx));   % ReLUの推奨初期値
                    elseif strcmp(lower(weight_init_std), 'sigmoid') || ...
                            strcmp(lower(weight_init_std), 'xavier')
                        scale = sqrt(1 / all_size_list(idx));   % Sigmoidの推奨初期値
                    end
                else
                    scale = weight_init_std;
                end

                W = scale .* randn(all_size_list(idx), all_size_list(idx + 1));
                b = zeros(1, all_size_list(idx + 1));
                obj.params.(strcat('W', num2str(idx))) = W;
                obj.params.(strcat('b', num2str(idx))) = b;
            end
        end
    end
end
