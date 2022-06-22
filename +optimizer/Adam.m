classdef Adam < handle
    %Adam Adam
    % http://arxiv.org/abs/1412.6980v8

    properties
        lr  % 学習率
        beta1
        beta2
        iter
        m
        v
    end

    methods
        function self = Adam(lr, beta1, beta2)
            % コンストラクタ

            if ~exist('lr', 'var')
                lr = 0.001;
            end
            if ~exist('beta1', 'var')
                beta1 = 0.9;
            end
            if ~exist('beta2', 'var')
                beta2 = 0.999;
            end

            self.lr = lr;
            self.beta1 = beta1;
            self.beta2 = beta2;
            self.iter = 0;
        end

        function update(self, params, grads)
            % パラメータの更新

            if isempty(self.m)
                self.m = containers.Map();
                self.v = containers.Map();
                for key = keys(params)
                    self.m(key{1}) = zeros(size(params(key{1})));
                    self.v(key{1}) = zeros(size(params(key{1})));
                end
            end

            self.iter = self.iter + 1;
            lr_t = self.lr .* sqrt(1 - self.beta2.^self.iter) ./ (1 - self.beta1.^self.iter);

            for key = keys(params)
                self.m(key{1}) = self.m(key{1}) + (1 - self.beta1) .* (grads(key{1}) - self.m(key{1}));
                self.v(key{1}) = self.v(key{1}) + (1 - self.beta2) .* (grads(key{1}).^2 - self.v(key{1}));
                params(key{1}) = params(key{1}) - lr_t .* self.m(key{1}) ./ (sqrt(self.v(key{1})) + 1e-7);
            end
        end
    end
end
