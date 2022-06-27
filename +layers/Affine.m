classdef Affine < handle
    %Affine Affineレイヤクラス

    properties
        W   % 重み
        b   % バイアス
        x   % 入力
        dW  % 重みの微分
        db  % バイアスの微分
        original_x_sz
    end

    methods
        function self = Affine(W, b)
            % コンストラクタ
            self.W = W;
            self.b = b;
        end

        function out = forward(self, x)
            % 順伝播

            % テンソル対応
            self.original_x_sz = size(x);
            x = reshape(x, [], size(x, ndims(x)));

            self.x = x;
            out = self.W * self.x + self.b;
        end

        function dx = backward(self, dout)
            % 逆伝播
            dx = self.W.' * dout;
            self.dW = dout * self.x.';
            self.db = sum(dout, 2);

            % テンソル対応
            dx = reshape(dx, self.original_x_sz);
        end
    end
end
