classdef SoftmaxWithLoss < handle
    %SoftmaxWithLoss Softmax-with-Lossレイヤクラス

    properties
        loss    % 損失
        y       % softmaxの出力
        t       % 教師データ (one-hot vector)
    end

    methods
        function loss = forward(self, x, t)
            % 順伝播
            self.t = t;
            self.y = functions.softmax(x);
            self.loss = functions.cross_entropy_error(self.y, self.t);
            loss = self.loss;
        end

        function dx = backward(self, dout)
            % 逆伝播
            if ~exist('dout', 'var')
                dout = 1;
            end

            batch_size = size(self.t, ndims(self.t));

            if isequal(size(self.t), size(self.y))  % one-hot-labelの場合
                dx = (self.y - cast(self.t, 'like', self.y)) ./ batch_size;
            else
                dx = self.y;
                ind = sub2ind(size(dx), self.t, 1:batch_size);
                dx(ind) = dx(ind) - 1;
                dx = dx ./ batch_size;
            end
        end
    end
end
