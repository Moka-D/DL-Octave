classdef Sigmoid < handle
    %Sigmoid Sigmoidレイヤクラス

    properties
        out % 出力
    end

    methods
        function obj = Sigmoid()
            % コンストラクタ
            obj.out = [];
        end

        function y = forward(obj, x)
            % 順伝播
            obj.out = 1 ./ (1 + exp(-x));
            y = obj.out;
        end

        function dx = backward(obj, dy)
            % 逆伝播
            dx = dout .* (1.0 - obj.out) .* obj.out;
        end
    end
end
