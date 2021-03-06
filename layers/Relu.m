classdef Relu < handle
    %Relu ReluCNX
    properties
        mask    % logicalzρ
    end


    methods
        % RXgN^
        function obj = Relu
            obj.mask = NaN;
        end


        % `d
        function out = forward(obj, x)
            obj.mask = (x <= 0);
            out = x;
            out(obj.mask) = 0;
        end


        % t`d
        function dx = backward(obj, dout)
            dx = dout;
            dx(obj.mask) = 0;
        end
    end
end
