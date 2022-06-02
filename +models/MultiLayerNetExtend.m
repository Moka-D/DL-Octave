classdef MultiLayerNetExtend < handle
    %MultiLayerNetExtend 拡張版の全結合による多層ニューラルネットワーク
    %
    % Weight Decay, Dropout, Batch Normalizationの機能を持つ

    properties
        input_size          % 入力サイズ (MNISTの場合は784)
        hidden_size_list    % 隠れ層のニューロンの数のリスト
        hidden_layer_num
        output_size         % 出力サイズ (MNISTの場合は10)
        weight_decay_lambda % Weight Decay(L2ノルム)の強さ
        use_dropout         % Dropoutを使用するかどうか
        dropout_ration      % Droupoutの割合
        use_batchnorm       % Batch Normalizationを使用するかどうか
        params
        layers
        last_layer
    end

    methods
        function obj = MultiLayerNetExtend(input_size, hidden_size_list, output_size, ...
                                           activation, weight_init_std, weight_decay_lambda, ...
                                           use_dropout, dropout_ration, use_batchnorm)
            % デフォルト引数
            if ~exist('activation', 'var')
                activation = 'relu';
            end
            if ~exist('weight_init_std', 'var')
                weight_init_std = 'relu';
            end
            if ~exist('weight_decay_lambda', 'var')
                weight_decay_lambda = 0;
            end
            if ~exist('use_dropout', 'var')
                use_dropout = false;
            end
            if ~exist('dropout_ration', 'var')
                dropout_ration = 0.5;
            end
            if ~exist('use_batchnorm', 'var')
                use_batchnorm = false;
            end

            obj.input_size = input_size;
            obj.output_size = output_size;
            obj.hidden_size_list = hidden_size_list;
            obj.hidden_layer_num = length(hidden_size_list);
            obj.use_dropout = use_dropout;
            obj.weight_decay_lambda = weight_decay_lambda;
            obj.use_batchnorm = use_batchnorm;
            obj.params = struct();

            % 重みの初期化
            obj.init_weight(weight_init_std);

            % レイヤの生成
            obj.layers = struct();
            for idx = 1:obj.hidden_layer_num
                obj.layers.(strcat('Affine', num2str(idx))) = layers.Affine(obj.params.(strcat('W', num2str(idx))), ...
                                                                            obj.params.(strcat('b', num2str(idx))));

                if obj.use_batchnorm
                    obj.params.(strcat('gamma', num2str(idx))) = ones(1, hidden_size_list(idx));
                    obj.params.(strcat('beta', num2str(idx))) = zeros(1, hidden_size_list(idx));
                    obj.layers.(strcat('BatchNorm', num2str(idx))) = layers.BatchNormalization(obj.params.(strcat('gamma', num2str(idx))), ...
                                                                                               obj.params.(strcat('beta', num2str(idx))));
                end

                if strcmp(activation, 'relu')
                    obj.layers.(strcat('Activation_function', num2str(idx))) = layers.Relu();
                elseif strcmp(activation, 'sigmoid')
                    obj.layers.(strcat('Activation_function', num2str(idx))) = layers.Sigmoid();
                else
                    error('Argument ''activation'' must be ''relu'' or ''sigmoid''');
                end

                if obj.use_dropout
                    obj.layers.(strcat('Dropout', num2str(idx))) = layers.Dropout(dropout_ration);
                end
            end

            idx = obj.hidden_layer_num + 1;
            obj.layers.(strcat('Affine', num2str(idx))) = layers.Affine(obj.params.(strcat('W', num2str(idx))), ...
                                                                        obj.params.(strcat('b', num2str(idx))));

            obj.last_layer = layers.SoftmaxWithLoss();
        end

        function y = predict(obj, x, train_flg)
            if ~exist('train_flg', 'var')
                train_flg = false;
            end

            % Affine層のパラメータ更新
            obj.update_weight();

            keys = fieldnames(obj.layers);
            for idx = 1:length(keys)
                key = keys{idx};
                if regexp(key, 'Dropout\d+', 'once') || regexp(key, 'BatchNorm\d+', 'once')
                    x = obj.layers.(key).forward(x, train_flg);
                else
                    x = obj.layers.(key).forward(x);
                end
            end
            y = x;
        end

        function ret = loss(obj, x, t, train_flg)
            % 損失関数を求める
            %
            % 引数のxは入力データ、tは教師ラベル

            if ~exist('train_flg', 'var')
                train_flg = false;
            end

            y = obj.predict(x, train_flg);

            weight_decay = 0;
            for idx = 1:obj.hidden_layer_num+1
                W_2 = obj.params.(strcat('W', num2str(idx))).^2;
                weight_decay = weight_decay + 0.5 .* obj.weight_decay_lambda .* sum(W_2(:));
            end

            ret = obj.last_layer.forward(y, t) + weight_decay;
        end

        function ret = accuracy(obj, x, t)
            y = obj.predict(x, false);
            [~, y] = max(y, [], 2);
            if size(t, 2) ~=1
                [~, t] = max(t, [], 2);
            end

            ret = sum(y==t, 'all') ./ size(x, 1);
        end

        function grads = numerical_gradient(obj, x, t)
            % 数値微分による勾配計算

            loss_W = @(W) obj.loss(x, t);

            grads = struct();
            for idx = 1:obj.hidden_layer_num+1
                grads.(strcat('W', num2str(idx))) = gradient.numerical_gradient(loss_W, obj.params.(strcat('W', num2str(idx))));
                grads.(strcat('b', num2str(idx))) = gradient.numerical_gradient(loss_W, obj.params.(strcat('b', num2str(idx))));

                if obj.use_batchnorm && idx ~= (obj.hidden_layer_num + 1)
                    grads.(strcat('gamma', num2str(idx))) = gradient.numerical_gradient(loss_W, obj.params.(strcat('gamma', num2str(idx))));
                    grads.(strcat('beta', num2str(idx))) = gradient.numerical_gradient(loss_W, obj.params.(strcat('beta', num2str(idx))));
                end
            end
        end

        function grads = gradient(obj, x, t)
            % 誤差逆伝播法による勾配計算

            % forward
            obj.loss(x, t, true);

            % backward
            dout = 1;
            dout = obj.last_layer.backward(dout);

            layer_names = fieldnames(obj.layers);
            for idx = length(layer_names):-1:1
                dout = obj.layers.(layer_names{idx}).backward(dout);
            end

            % 設定
            grads = struct();
            for idx = 1:obj.hidden_layer_num+1
                layer = obj.layers.(strcat('Affine', num2str(idx)));
                grads.(strcat('W', num2str(idx))) = layer.dW + obj.weight_decay_lambda .* layer.W;
                grads.(strcat('b', num2str(idx))) = layer.db;

                if obj.use_batchnorm && idx ~= (obj.hidden_layer_num + 1)
                    layer = obj.layers.(strcat('BatchNorm', num2str(idx)));
                    grads.(strcat('gamma', num2str(idx))) = layer.dgamma;
                    grads.(strcat('beta', num2str(idx))) = layer.dbeta;
                end
            end
        end
    end

    methods(Access = private, Hidden = true)
        function init_weight(obj, weight_init_std)
            % 重みの初期値設定

            all_size_list = cat(2, obj.input_size, obj.hidden_size_list, obj.output_size);

            for idx = 1:length(all_size_list)-1
                if ischar(weight_init_std)
                    if strcmpi(weight_init_std, 'relu') || ...
                       strcmpi(weight_init_std, 'he')
                        scale = sqrt(2 / all_size_list(idx));   % ReLUの推奨初期値
                    elseif strcmpi(weight_init_std, 'sigmoid') || ...
                           strcmpi(weight_init_std, 'xavier')
                        scale = sqrt(1 / all_size_list(idx));   % Sigmoidの推奨初期値
                    end
                else
                    scale = weight_init_std;
                end

                obj.params.(strcat('W', num2str(idx))) = scale .* randn(all_size_list(idx), all_size_list(idx + 1));
                obj.params.(strcat('b', num2str(idx))) = zeros(1, all_size_list(idx + 1));
            end
        end

        function update_weight(obj)
            % Affine層のパラメータ更新
            for idx = 1:obj.hidden_layer_num+1
                obj.layers.(strcat('Affine', num2str(idx))).W = obj.params.(strcat('W', num2str(idx)));
                obj.layers.(strcat('Affine', num2str(idx))).b = obj.params.(strcat('b', num2str(idx)));

                if obj.use_batchnorm && idx ~= (obj.hidden_layer_num + 1)
                    obj.layers.(strcat('BatchNorm', num2str(idx))).gamma_ = obj.params.(strcat('gamma', num2str(idx)));
                    obj.layers.(strcat('BatchNorm', num2str(idx))).beta_ = obj.params.(strcat('beta', num2str(idx)));
                end
            end
        end
    end
end
