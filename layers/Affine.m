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
        % コンストラクタ
        function obj = Affine(W, b)
            obj.W  = W;
            obj.b  = b;
            obj.x  = NaN;
            obj.dW = NaN;
            obj.db = NaN;
        end


        % 順伝播
        function out = forward(obj, x)
            obj.x = x;
            out = obj.x * obj.W + obj.b;
        end


        % 逆伝播
        function dx = backward(obj, dout)
            dx = dout * obj.W.';
            obj.dW = obj.x.' * dout;
            obj.db = sum(dout, 1);
        end
    end
end
