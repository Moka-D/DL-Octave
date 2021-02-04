classdef Relu < handle
    %Relu ReluƒŒƒCƒ„ƒNƒ‰ƒX
    properties
        mask    % logical”z—ñ
    end


    methods
        % ƒRƒ“ƒXƒgƒ‰ƒNƒ^
        function obj = Relu
            obj.mask = NaN;
        end


        % ‡“`”d
        function out = forward(obj, x)
            obj.mask = (x <= 0);
            out = x;
            out(obj.mask) = 0;
        end


        % ‹t“`”d
        function dx = backward(obj, dout)
            dout(obj.mask) = 0;
            dx = dout;
        end
    end
end
