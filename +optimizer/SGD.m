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
            for iField = 1:length(fields)
                param = getfield(params, fields{iField});
                grad = getfield(grads, fields{iField});
                param = param - obj.lr .* grad;
                params = setfield(params, fields{iField}, param);
            end
        end
    end
end

