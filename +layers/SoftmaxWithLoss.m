classdef SoftmaxWithLoss < handle
    %SoftmaxWithLoss Softmax-with-Loss���C���N���X

    properties
        loss    % ����
        y       % softmax�̏o��
        t       % ���t�f�[�^ (one-hot vector)
    end

    methods
        function obj = SoftmaxWithLoss()
            % �R���X�g���N�^
            obj.loss = NaN;
            obj.y = NaN;
            obj.t = NaN;
        end

        function loss = forward(obj, x, t)
            % ���`�d
            obj.t = t;
            obj.y = functions.softmax(x);
            obj.loss = functions.cross_entropy_error(obj.y, obj.t);
            loss = obj.loss;
        end

        function dx = backward(obj, dout)
            % �t�`�d
            if ~exist('dout', 'var')
                dout = 1;
            end

            batch_size = size(obj.t, 1);

            if (size(obj.t, 1) == size(obj.y, 1)) ...   % one-hot-label�̏ꍇ
                && (size(obj.t, 2) == size(obj.y, 2))
                dx = (obj.y - obj.t) ./ batch_size;
            else
                dx = obj.y;
                tmp = zeros(size(dx));
                tmp(1:batch_size, obj.t) = 1;
                dx = (dx - tmp) ./ batch_size;
            end
        end
    end
end
