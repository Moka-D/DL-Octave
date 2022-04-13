classdef Relu < handle
    %Relu Reluレイヤクラス

    properties
        mask    % logical配列
    end

    methods
        function obj = Relu()
            % コンストラクタ
            obj.mask = [];
        end

        function out = forward(obj, x)
            % 順伝播
            obj.mask = (x <= 0);
            out = x;
            out(obj.mask) = 0;
        end

        function dx = backward(obj, dout)
            % 逆伝播
            dx = dout;
            dx(obj.mask) = 0;
        end
    end
end
