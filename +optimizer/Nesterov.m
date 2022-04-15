classdef Nesterov < handle
    %Nesterov Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)

    properties
        lr
        momentum
        v
    end

    methods
        function obj = Nesterov(lr, momentum)
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
            keys = fieldnames(params);
            if isempty(obj.v)
                obj.v = struct();
                for idx = 1:length(keys)
                    key = keys{idx};
                    obj.v.(key) = zeros(size(params.(key)));
                end
            end

            for idx = 1:length(key)
                key = keys{idx};
                params.(key) = params.(key) + obj.momentum .* obj.momentum .* obj.v.(key);
                params.(key) = params.(key) - (1 + obj.momentum) .* obj.lr .* grads.(key);
                obj.v.(key) = obj.v.(key) .* obj.momentum;
                obj.v.(key) = obj.v.(key) - obj.lr .* grads.(key);
            end
        end
    end
end
