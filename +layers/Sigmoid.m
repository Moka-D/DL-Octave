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

        function out = forward(obj, x)
            % 順伝播
            obj.out = functions.sigmoid(x);
            out = obj.out;
        end

        function dx = backward(obj, dy)
            % 逆伝播
            dx = dout .* (1.0 - obj.out) .* obj.out;
        end
    end
end
