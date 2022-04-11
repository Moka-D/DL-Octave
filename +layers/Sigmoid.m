classdef Sigmoid < handle
    %Sigmoid SigmoidƒŒƒCƒ„ƒNƒ‰ƒX

    properties
        out % o—Í
    end

    methods
        function obj = Sigmoid()
            % ƒRƒ“ƒXƒgƒ‰ƒNƒ^
            obj.out = [];
        end

        function y = forward(obj, x)
            % ‡“`”d
            obj.out = 1 ./ (1 + exp(-x));
            y = obj.out;
        end

        function dx = backward(obj, dy)
            % ‹t“`”d
            dx = dout .* (1.0 - obj.out) .* obj.out;
        end
    end
end
