classdef Dropout < handle
    %Dropout Dropout層クラス
    %
    % http://arxiv.org/abs/1207.0580

    properties
        dropout_ratio
        mask
    end

    methods
        function obj = Dropout(dropout_ratio)
            if ~exist('dropout_ratio', 'var')
                dropout_ratio = 0.5;
            end

            obj.dropout_ratio = dropout_ratio;
            obj.mask = [];
        end

        function out = forward(x, train_flg)
            if ~exist('train_flg', 'var')
                train_flg = true;
            end

            if train_flg
                obj.mask = randn(size(x)) > obj.dropout_ratio;
                out = x .* obj.mask;
            else
                out = x .* (1 - obj.dropout_ratio);
            end
        end

        function dx = backward(obj, dout)
            dx = dout .* obj.mask;
        end
    end
end
