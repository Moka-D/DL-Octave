classdef SoftmaxWithLoss < handle
    %SoftmaxWithLoss Softmax-with-LossCNX
    properties
        loss    % ΉΈ
        y       % softmaxΜoΝ
        t       % ³tf[^ (one-hot vector)
    end


    methods
        % RXgN^
        function obj = SoftmaxWithLoss()
            obj.loss = NaN;
            obj.y    = NaN;
            obj.t    = NaN;
        end


        % `d
        function loss = forward(obj, x, t)
            obj.t = t;
            obj.y = softmax(x);
            obj.loss = cross_entropy_error(obj.y, obj.t);
            loss = obj.loss;
        end


        % t`d
        function dx = backward(obj, dout)
            if ~exist('dout', 'var')
                dout = 1;
            end

            batch_size = size(obj.t, 1);

            if (size(obj.t, 1) == size(obj.y, 1)) ...   % one-hot-labelΜκ
                && (size(obj.t, 2) == size(obj.y, 2))
                dx = (obj.y - obj.t) ./ batch_size;
            else
                dx = obj.y;
                tmp = zeros(size(dx));
                tmp(1:batch_size, obj.t) = 1;
                dx = (dx - tmp) ./ batch_size;
            end
        end
    end
end
