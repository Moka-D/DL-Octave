function loss = mean_square_error(y, t)
    %mean_square_error •½‹Ï“ñæŒë·
    loss = 0.5 .* sum((y - t).^2);
end
