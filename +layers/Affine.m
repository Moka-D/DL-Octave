classdef Affine < handle
    %Affine Affine���C���N���X

    properties
        W   % �d��
        b   % �o�C�A�X
        x   % ����
        dW  % �d�݂̔���
        db  % �o�C�A�X�̔���
    end

    methods
        function obj = Affine(W, b)
            % �R���X�g���N�^
            obj.W = W;
            obj.b = b;
            obj.x = NaN;
            obj.dW = NaN;
            obj.db = NaN;
        end

        function out = forward(obj, x)
            % ���`�d
            obj.x = x;
            out = obj.x * obj.W + obj.b;
        end

        function dx = backward(obj, dout)
            % �t�`�d
            dx = dout * obj.W.';
            obj.dW = obj.x.' * dout;
            obj.db = sum(dout, 1);
        end
    end
end
