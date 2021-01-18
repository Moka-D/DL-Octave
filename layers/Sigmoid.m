classdef Sigmoid < handle
    %Sigmoid SigmoidƒŒƒCƒ„ƒNƒ‰ƒX
    properties
        out % o—Í
    end


    methods
        % ƒRƒ“ƒXƒgƒ‰ƒNƒ^
        function obj = Sigmoid()
            obj.out = NaN;
        end


        % ‡“`”d
        function y = forward(obj, x)
            y = 1 ./ (1 + exp(-x));
            obj.out = y;
        end


        % ‹t“`”d
        function dx = backward(obj, dy)
            dx = dout .* (1.0 - obj.out) .* obj.out;
        end
    end
end
