classdef Relu < handle
    %Relu Relu���C���N���X

    properties
        mask    % logical�z��
    end

    methods
        function obj = Relu
            % �R���X�g���N�^
            obj.mask = NaN;
        end

        function out = forward(obj, x)
            % ���`�d
            obj.mask = (x <= 0);
            out = x;
            out(obj.mask) = 0;
        end

        function dx = backward(obj, dout)
            % �t�`�d
            dx = dout;
            dx(obj.mask) = 0;
        end
    end
end
