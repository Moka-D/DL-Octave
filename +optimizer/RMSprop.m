classdef RMSprop < handle
    %RMSprop

    properties
        lr
        decay_rate
        h
    end

    methods
        function self = RMSprop(lr, decay_rate)
            % コンストラクタ

            if ~exist('lr', 'var')
                lr = 0.01;
            end
            if ~exist('decay_rate', 'var')
                decay_rate = 0.99;
            end

            self.lr = lr;
            self.decay_rate = decay_rate;
        end

        function update(self, params, grads)
            if isempty(self.h)
                self.h = containers.Map;
                for key = keys(params)
                    self.h(key{1}) = zeros(size(params(key{1})));
                end
            end

            for key = keys(params)
                self.h(key{1}) = self.h(key{1}) .* self.decay_rate;
                self.h(key{1}) = self.h(key{1}) + (1 - self.decay_rate) .* grads(key{1}) .* grads(key{1});
                params(key{1}) = params(key{1}) - self.lr .* grads(key{1}) ./ (sqrt(self.h(key{1})) + 1e-7);
            end
        end
    end
end
