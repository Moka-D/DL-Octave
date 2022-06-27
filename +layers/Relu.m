classdef Relu < handle
    %Relu Reluレイヤクラス

    properties
        mask    % logical配列
    end

    methods
        function out = forward(self, x)
            % 順伝播
            self.mask = (x <= 0);
            out = x;
            out(self.mask) = 0;
        end

        function dx = backward(self, dout)
            % 逆伝播
            dx = dout;
            dx(self.mask) = 0;
        end
    end
end
