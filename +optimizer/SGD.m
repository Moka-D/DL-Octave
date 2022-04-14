classdef SGD < handle
    %SGD 確率的勾配降下法

    properties
        lr  % 学習率
    end

    methods
        function obj = SGD(lr)
            % コンストラクタ

            if ~exist('lr', 'var')
                lr = 0.01;
            end
            obj.lr = lr;
        end

        function params = update(obj, params, grads)
            % パラメータの更新
            keys = fieldnames(params);
            for idx = 1:length(keys)
                key = keys{idx};
                params.(key) = params.(key) - obj.lr .* grads.(key);
            end
        end
    end
end
