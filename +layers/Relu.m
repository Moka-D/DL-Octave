classdef Relu < handle
    %Relu ReluƒŒƒCƒ„ƒNƒ‰ƒX

    properties
        mask    % logical”z—ñ
    end

    methods
        function obj = Relu
            % ƒRƒ“ƒXƒgƒ‰ƒNƒ^
            obj.mask = NaN;
        end

        function out = forward(obj, x)
            % ‡“`”d
            obj.mask = (x <= 0);
            out = x;
            out(obj.mask) = 0;
        end

        function dx = backward(obj, dout)
            % ‹t“`”d
            dx = dout;
            dx(obj.mask) = 0;
        end
    end
end
