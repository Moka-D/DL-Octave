classdef SGD < handle
    %SGD 確率的勾配降下法

    properties
        lr  % 学習率
    end

    methods
        function self = SGD(lr)
            % コンストラクタ

            if ~exist('lr', 'var')
                lr = 0.01;
            end
            self.lr = lr;
        end

        function params = update(self, params, grads)
            % パラメータの更新
            for key = fieldnames(params)'
                params.(key{1}) = params.(key{1}) - self.lr .* grads.(key{1});
            end
        end
    end
end
