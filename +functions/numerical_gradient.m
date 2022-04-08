function grad = numerical_gradient(f, x)
    %numerical_gradient ���l���z�v�Z�֐�

    h = 1e-4;   % 0.0001
    grad = zeros(size(x));  % x�Ɠ����`���0�s����쐬

    for idx = 1:numel(x)
        tmp = x(idx);
        % f(x+h)�̌v�Z
        x(idx) = tmp + h;
        fxh1 = f(x);

        % f(x-h)�̌v�Z
        x(idx) = tmp - h;
        fxh2 = f(x);

        grad(idx) = (fxh1 - fxh2) ./ (2 .* h);
        x(idx) = tmp;
    end
end
