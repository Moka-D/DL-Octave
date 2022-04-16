classdef BatchNormalization < handle
    %BatchNormalization Batch Normalizationクラス
    %
    % http://arxiv.org/abs/1502.03167

    properties
        gamma_
        beta_
        momentum
        input_shape
        running_mean
        running_var
        batch_size
        xc
        std_
        dgamma
        dbeta
    end

    methods
        function obj = BatchNormalization(gamma_, beta_, momentum, running_mean, running_var)
            % デフォルト引数
            if ~exist('momentum', 'var')
                momentum = 0.9;
            end
            if ~exist('running_mean', 'var')
                running_mean = [];
            end
            if ~exist('running_var', 'var')
                running_var = [];
            end

            obj.gamma_ = gamma_;
            obj.beta_ = beta_;
            obj.momentum = momentum;
            obj.input_shape = [];   % Conv層の場合は4次元、全結合層の場合は2次元

            % テスト時に使用する平均と分散
            obj.running_mean = running_mean;
            obj.running_var = running_var;

            % backward時に使用する中間データ
            obj.batch_size =[];
            obj.xc = [];
            obj.std_ = [];
            obj.dgamma = [];
            obj.dbeta = [];
        end

        function out = forward(obj, x, train_flg)
            % デフォルト引数
            if ~exist('train_flg', 'var')
                train_flg = true;
            end

            obj.input_shape = size(x);
            if ndims(x) ~= 2
                N = size(x, 1);
                x = reshape(x, N, []);
            end

            out = obj.forward_(x, train_flg);
            out = reshape(out, obj.input_shape);
        end

        function dx = backward(obj, dout)
            if ndims(dout) ~= 2
                N = size(dout, 1);
                dout = reshape(dout, N, []);
            end

            dx = obj.backward_(dout);

            dx = reshape(dx, obj.input_shape);
        end
    end

    methods (Access = private, Hidden = true)
        function out = forward_(obj, x, train_flg)
            if isempty(obj.running_mean)
                D = size(x, 2);
                obj.running_mean = zeros(1, D);
                obj.running_var = zeros(1, D);
            end

            if train_flg
                mu = mean(x, 1);
                xc = x - mu;
                var_ = mean(xc.^2, 1);
                std_ = sqrt(var_ + 10e-7);
                xn = xc / std_;

                obj.batch_size = size(x, 1);
                obj.xc = xc;
                obj.xn = xn;
                obj.std_ = std_;
                obj.running_mean = obj.momentum .* obj.running_mean + (1-obj.momentum) .* mu;
                obj.running_var = obj.momentum .* obj.running_var + (1-obj.momentum) .* var_;
            else
                xc = x - obj.running_mean;
                xn = xc ./ sqrt(obj.running_var + 10e-7);
            end

            out = obj.gamma_ .* xn + obj.beta_;
        end

        function dx = backward_(obj, dout)
            dbeta = sum(dout, 1);
            dgamma = sum(obj.xn .* dout, 1);
            dxn = obj.gamma_ .* dout;
            dxc = dxn ./ obj.std_;
            dstd = -sum((dxn .* obj.xc) ./ (obj.std_ .* obj.std_), 1);
            dvar = 0.5 .* dstd ./ obj.std_;
            dxc = dxc + (2.0 ./ obj.batch_size) .* obj.xc .* dvar;
            dmu = sum(dxc, 1);
            dx = dxc - dmu ./ obj.batch_size;

            obj.dgamma = dgamma;
            obj.dbeta = dbeta;
        end
    end
end
