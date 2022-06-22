classdef AdaGrad < handle
    %AdaGrad AdaGrad

    properties
        lr  % 学習率
        h
    end

    methods
        function self = AdaGrad(lr)
            % コンストラクタ
            if ~exist('lr', 'var')
                lr = 0.01;
            end
            self.lr = lr;
        end

        function update(self, params, grads)
            % パラメータの更新

            if isempty(self.h)
                self.h = containers.Map;
                for key = keys(params)
                    self.h(key{1}) = zeros(size(params(key{1})));
                end
            end

            for key = keys(params)
                self.h(key{1}) = self.h(key{1}) + grads(key{1}) .* grads(key{1});
                params(key{1}) = params(key{1}) - self.lr .* grads(key{1}) ./ (sqrt(self.h(key{1})) + 1e-7);
            end
        end
    end
end
