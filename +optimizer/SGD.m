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
            for i_field = 1:length(fields)
                field_name = fields{i_field};
                param = getfield(params, field_name);
                grad = getfield(grads, field_name);

                param = param - obj.lr .* grad;
                params = setfield(params, field_name, param);
            end
        end
    end
end
