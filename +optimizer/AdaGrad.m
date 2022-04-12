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

            fields = fieldnames(params);

            if isempty(obj.h)
                obj.h = struct();
                for i_field = 1:length(fields)
                    val = getfield(params, fields{i_field});
                    h = zeros(size(val));
                    obj.h = setfield(obj.h, fields{i_field}, h);
                end
            end

            for i_field = 1:length(fields)
                field_name = fields{i_field};
                param = getfield(params, field_name);
                grad = getfield(grads, field_name);
                h = getfield(obj.h, field_name);

                h = h + grad .* grad;
                param = param - obj.lr .* grad ./ (sqrt(h) + 1e-7);

                obj.h = setfield(obj.h, field_name, h);
                params = setfield(params, field_name, param);
            end
        end
    end
end
