function loss = mean_square_error(y, t)
    %mean_square_error ���ϓ��덷
    loss = 0.5 .* sum((y - t).^2);
end
