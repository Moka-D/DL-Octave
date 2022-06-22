classdef Nesterov < handle
    %Nesterov Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)

    properties
        lr
        momentum
        v
    end

    methods
        function self = Nesterov(lr, momentum)
            % コンストラクタ

            if ~exist('lr', 'var')
                lr = 0.01;
            end
            if ~exist('momentum', 'var')
                momentum = 0.9;
            end

            self.lr = lr;
            self.momentum = momentum;
        end

        function update(self, params, grads)
            if isempty(self.v)
                self.v = containers.Map;
                for key = keys(params)
                    self.v(key{1}) = zeros(size(params(key{1})));
                end
            end

            for key = keys(params)
                params(key{1}) = params(key{1}) + self.momentum .* self.momentum .* self.v(key{1});
                params(key{1}) = params(key{1}) - (1 + self.momentum) .* self.lr .* grads(key{1});
                self.v(key{1}) = self.v(key{1}) .* self.momentum;
                self.v(key{1}) = self.v(key{1}) - self.lr .* grads(key{1});
            end
        end
    end
end
