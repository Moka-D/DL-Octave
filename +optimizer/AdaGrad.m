classdef AdaGrad < handle
    %AdaGrad AdaGrad

    properties
        lr  % 学習率
        h
    end

    methods
        function obj = AdaGrad(lr)
            % コンストラクタ
            if ~exist('lr', 'var')
                lr = 0.01;
            end
            obj.lr = lr;
            obj.h = [];
        end

        function params = update(obj, params, grads)
            % パラメータの更新

            keys = fieldnames(params);

            if isempty(obj.h)
                obj.h = struct();
                for idx = 1:length(keys)
                    key = keys{idx};
                    obj.h.(key) = zeros(size(params.(key)));
                end
            end

            for idx = 1:length(keys)
                key = keys{idx};
                obj.h.(key) = obj.h.(key) + grads.(key) .* grads.(key);
                params.(key) = params.(key) - obj.lr .* grads.(key) ./ (sqrt(obj.h.(key)) + 1e-7);
            end
        end
    end
end
