classdef Sigmoid < handle
    %Sigmoid Sigmoid���C���N���X

    properties
        out % �o��
    end

    methods
        function obj = Sigmoid()
            % �R���X�g���N�^
            obj.out = [];
        end

        function y = forward(obj, x)
            % ���`�d
            obj.out = 1 ./ (1 + exp(-x));
            y = obj.out;
        end

        function dx = backward(obj, dy)
            % �t�`�d
            dx = dout .* (1.0 - obj.out) .* obj.out;
        end
    end
end
