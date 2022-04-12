classdef SGD < handle
    %SGD �m���I���z�~���@
    properties
        lr  % �w�K��
    end

    methods
        function obj = SGD(lr)
            % �R���X�g���N�^

            if ~exist('lr', 'var')
                lr = 0.01;
            end
            obj.lr = lr;
        end

        function params = update(obj, params, grads)
            % �p�����[�^�̍X�V

            fields = fieldnames(params);
            for i_field = 1:length(fields)
                param = getfield(params, fields{i_field});
                grad = getfield(grads, fields{i_field});
                param = param - obj.lr .* grad;
                params = setfield(params, fields{i_field}, param);
            end
        end
    end
end

