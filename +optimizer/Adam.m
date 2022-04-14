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

            keys = fieldnames(params);

            if isempty(obj.m)
                obj.m = struct();
                obj.v = struct();
                for idx = 1:length(keys)
                    key = keys{idx};
                    obj.m.(key) = zeros(size(params.(key)));
                    obj.v.(key) = zeros(size(params.(key)));
                end
            end

            obj.iter = obj.iter + 1;
            lr_t = obj.lr .* sqrt(1 - obj.beta2.^obj.iter) ./ (1 - obj.beta1.^obj.iter);

            for idx = 1:length(keys)
                key = keys{idx};
                obj.m.(key) = obj.m.(key) + (1 - obj.beta1) .* (grad - obj.m.(key));
                obj.v.(key) = obj.v.(key) + (1 - obj.beta2) .* (grad.^2 - obj.v.(key));
                params.(key) = params.(key) - lr_t .* obj.m.(key) ./ (sqrt(obj.v.(key)) + 1e-7);
            end
        end
    end
end
