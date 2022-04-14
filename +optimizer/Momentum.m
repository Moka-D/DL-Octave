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

            keys = fieldnames(params);

            if isempty(obj.v)
                obj.v = struct();
                for idx = 1:length(keys)
                    key = keys{idx};
                    obj.v.(key) = zeros(size(params.(key)));
                end
            end

            for idx = 1:length(keys)
                key = keys{idx};
                obj.v.(key) = obj.momentum .* obj.v.(key) - obj.lr .* grads.(key);
                params.(key) = params.(key) + obj.v.(key);
            end
        end
    end
end
