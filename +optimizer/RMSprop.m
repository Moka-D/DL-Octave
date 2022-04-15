classdef RMSprop < handle
    %RMSprop

    properties
        lr
        decay_rate
        h
    end

    methods
        function obj = RMSprop(lr, decay_rate)
            % コンストラクタ

            if ~exist('lr', 'var')
                lr = 0.01;
            end
            if ~exist('decay_rate', 'var')
                decay_rate = 0.99;
            end

            obj.lr = lr;
            obj.decay_rate = decay_rate;
            obj.h = [];
        end

        function params = update(obj, params, grads)
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
                obj.h.(key) = obj.h.(key) .* obj.decay_rate;
                obj.h.(key) = obj.h.(key) + (1 - obj.decay_rate) .* grads.(key) .* grads.(key);
                params.(key) = params.(key) - obj.lr .* grads.(key) ./ (sqrt(obj.h.(key)) + 1e-7);
            end
        end
    end
end
