classdef Dropout < handle
    %Dropout Dropout層クラス
    %
    % http://arxiv.org/abs/1207.0580

    properties
        dropout_ratio
        mask
    end

    methods
        function self = Dropout(dropout_ratio)
            if ~exist('dropout_ratio', 'var')
                dropout_ratio = 0.5;
            end
            self.dropout_ratio = dropout_ratio;
        end

        function out = forward(self, x, train_flg)
            if ~exist('train_flg', 'var')
                train_flg = true;
            end

            if train_flg
                self.mask = rand(size(x)) > self.dropout_ratio;
                out = x .* self.mask;
            else
                out = x .* (1 - self.dropout_ratio);
            end
        end

        function dx = backward(self, dout)
            dx = dout .* self.mask;
        end
    end
end
