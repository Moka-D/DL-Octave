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

            fields = fieldnames(params);
            for iField = 1:length(fields)
                param = getfield(params, fields{iField});
                grad = getfield(grads, fields{iField});
                param = param - obj.lr .* grad;
                params = setfield(params, fields{iField}, param);
            end
        end
    end
end

