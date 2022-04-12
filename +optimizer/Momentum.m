classdef Momentum < handle
    %Momentum MomentumSGD

    properties
        lr          % 学習率
        momentum    % モーメンタム
        v           % 更新速度
    end

    methods
        function obj = Momentum(lr, momentum)
            % コンストラクタ

            if ~exist('lr', 'var')
                lr = 0.01;
            end
            if ~exist('momentum', 'var')
                momentum = 0.9;
            end

            obj.lr = lr;
            obj.momentum = momentum;
            obj.v = [];
        end

        function params = update(obj, params, grads)
            % パラメータの更新

            fields = fieldnames(params);

            if isempty(obj.v)
                obj.v = struct();
                for i_field = 1:length(fields)
                    field_name = fields{i_field};
                    val = getfield(params, field_name);
                    v = zeros(size(val));
                    obj.v = setfield(obj.v, field_name, v);
                end
            end

            for i_field = 1:length(fields)
                field_name = fields{i_field};
                param = getfield(params, field_name);
                grad = getfield(grads, field_name);
                v = getfield(obj.v, field_name);

                v = obj.momentum .* v - obj.lr .* grad;
                param = param + v;

                obj.v = setfield(obj.v, field_name, v);
                params = setfield(params, field_name, param);
            end
        end
    end
end
