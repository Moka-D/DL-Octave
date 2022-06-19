function loss = sum_square_error(y, t)
    %mean_square_error 平均二乗誤差
    loss = 0.5 .* sum((y - t).^2, 'all');
end
