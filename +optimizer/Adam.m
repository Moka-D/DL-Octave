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
        function obj = Adam(lr, beta1, beta2)
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

            obj.lr = lr;
            obj.beta1 = beta1;
            obj.beta2 = beta2;
            obj.iter = 0;
            obj.m = [];
            obj.v = [];
        end

        function params = update(obj, params, grads)
            % パラメータの更新

            fields = fieldnames(params);

            if isempty(obj.m)
                obj.m = struct();
                obj.v = struct();
                for i_field = 1:length(fields)
                    field_name = fields{i_field};
                    val = getfield(params, field_name);
                    val = zeros(size(val));
                    obj.m = setfield(obj.m, field_name, val);
                    obj.v = setfield(obj.v, field_name, val);
                end
            end

            obj.iter = obj.iter + 1;
            lr_t = obj.lr .* sqrt(1 - obj.beta2.^obj.iter) ./ (1 - obj.beta1.^obj.iter);

            for i_field = 1:length(fields)
                field_name = fields{i_field};
                param = getfield(params, field_name);
                grad = getfield(grads, field_name);
                m = getfield(obj.m, field_name);
                v = getfield(obj.v, field_name);

                m = m + (1 - obj.beta1) .* (grad - m);
                v = v + (1 - obj.beta2) .* (grad.^2 - v);
                param = param - lr_t .* m ./ (sqrt(v) + 1e-7);

                obj.m = setfield(obj.m, field_name, m);
                obj.v = setfield(obj.v, field_name, v);
                params = setfield(params, field_name, param);
            end
        end
    end
end
