classdef Affine < handle
    %Affine Affineレイヤクラス

    properties
        W   % 重み
        b   % バイアス
        x   % 入力
        dW  % 重みの微分
        db  % バイアスの微分
        original_x_shape
    end

    methods
        function obj = Affine(W, b)
            % コンストラクタ
            obj.W = W;
            obj.b = b;
            obj.x = [];
            obj.dW = [];
            obj.db = [];
            obj.original_x_shape = [];
        end

        function out = forward(obj, x)
            % 順伝播

            % テンソル対応
            obj.original_x_shape = size(x);
            x = reshape(x, size(x, 1), []);

            obj.x = x;
            out = obj.x * obj.W + obj.b;
        end

        function dx = backward(obj, dout)
            % 逆伝播
            dx = dout * obj.W.';
            obj.dW = obj.x.' * dout;
            obj.db = sum(dout, 1);

            dx = reshape(dx, obj.original_x_shape); % テンソル対応
        end
    end
end
