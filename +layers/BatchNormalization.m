classdef BatchNormalization < handle
    %BatchNormalization Batch Normalizationクラス
    %
    % http://arxiv.org/abs/1502.03167

    properties
        gamma
        beta
        momentum
        input_sz        % Conv層の場合は4次元、全結合層の場合は2次元
        running_mean
        running_var
        xn

        % backward時に使用する中間データ
        batch_size
        xc
        std
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

            obj.gamma = gamma_;
            obj.beta = beta_;
            obj.momentum = momentum;

            % テスト時に使用する平均と分散
            obj.running_mean = running_mean;
            obj.running_var = running_var;
        end

        function out = forward(self, x, train_flg)
            % デフォルト引数
            if ~exist('train_flg', 'var')
                train_flg = true;
            end

            self.input_sz = size(x);
            if ~ismatrix(x)
                [~, ~, ~, N] = size(x);
                x = reshape(x, [], N);
            end

            out = self.forward_(x, train_flg);
            out = reshape(out, self.input_sz);
        end

        function dx = backward(obj, dout)
            if ~ismatrix(dout)
                [~, ~, ~, N] = size(dout);
                dout = reshape(dout, [], N);
            end

            dx = obj.backward_(dout);
            dx = reshape(dx, obj.input_sz);
        end
    end

    methods (Access = private, Hidden = true)
        function out = forward_(self, x, train_flg)
            if isempty(self.running_mean)
                [D, ~] = size(x);
                self.running_mean = zeros(D, 1);
                self.running_var = zeros(D, 1);
            end

            if train_flg
                mu = mean(x, 2);
                xc_ = x - mu;
                var_ = mean(xc_.^2, 2);
                std_ = sqrt(var_ + 10e-7);
                xn_ = xc_ ./ std_;

                self.batch_size = size(x, 1);
                self.xc = xc_;
                self.xn = xn_;
                self.std = std_;
                self.running_mean = self.momentum .* self.running_mean + (1 - self.momentum) .* mu;
                self.running_var = self.momentum .* self.running_var + (1 - self.momentum) .* var_;
            else
                xc_ = x - self.running_mean;
                xn_ = xc_ ./ sqrt(self.running_var + 10e-7);
            end

            out = self.gamma .* xn_ + self.beta;
        end

        function dx = backward_(self, dout)
            dbeta_ = sum(dout, 2);
            dgamma_ = sum(self.xn .* dout, 2);
            dxn = self.gamma .* dout;
            dxc = dxn ./ self.std;
            dstd = -sum((dxn .* self.xc) ./ (self.std .* self.std), 2);
            dvar = 0.5 .* dstd ./ self.std;
            dxc = dxc + (2.0 ./ self.batch_size) .* self.xc .* dvar;
            dmu = sum(dxc, 1);
            dx = dxc - dmu ./ self.batch_size;

            self.dgamma = dgamma_;
            self.dbeta = dbeta_;
        end
    end
end
