classdef Momentum < handle
    %Momentum MomentumSGD

    properties
        lr          % 学習率
        momentum    % モーメンタム
        v           % 更新速度
    end

    methods
        function self = Momentum(lr, momentum)
            % コンストラクタ

            if ~exist('lr', 'var')
                lr = 0.01;
            end
            if ~exist('momentum', 'var')
                momentum = 0.9;
            end

            self.lr = lr;
            self.momentum = momentum;
        end

        function update(self, params, grads)
            % パラメータの更新

            if isempty(self.v)
                self.v = containers.Map();
                for key = keys(params)
                    self.v(key{1}) = zeros(size(params(key{1})));
                end
            end

            for key = keys(params)
                self.v(key{1}) = self.momentum .* self.v(key{1}) - self.lr .* grads(key{1});
                params(key{1}) = params(key{1}) + self.v(key{1});
            end
        end
    end
end
