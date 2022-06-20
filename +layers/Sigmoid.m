classdef Sigmoid < handle
    %Sigmoid Sigmoidレイヤクラス

    properties
        out % 出力
    end

    methods
        function out = forward(self, x)
            % 順伝播
            out = functions.sigmoid(x);
            self.out = out;
        end

        function dx = backward(self, dout)
            % 逆伝播
            dx = dout .* (1 - self.out) .* self.out;
        end
    end
end
