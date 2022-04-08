classdef Affine < handle
    %Affine Affineレイヤクラス

    properties
        W   % 重み
        b   % バイアス
        x   % 入力
        dW  % 重みの微分
        db  % バイアスの微分
    end

    methods
        function obj = Affine(W, b)
            % コンストラクタ
            obj.W = W;
            obj.b = b;
            obj.x = NaN;
            obj.dW = NaN;
            obj.db = NaN;
        end

        function out = forward(obj, x)
            % 順伝播
            obj.x = x;
            out = obj.x * obj.W + obj.b;
        end

        function dx = backward(obj, dout)
            % 逆伝播
            dx = dout * obj.W.';
            obj.dW = obj.x.' * dout;
            obj.db = sum(dout, 1);
        end
    end
end
