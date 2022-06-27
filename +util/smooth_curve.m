function y = smooth_curve(x)
    %smooth_curve 損失関数のグラフを滑らかにする
    %
    % http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html

    window_len = 11;
    s = horzcat(x(window_len:-1:2), x, x(end:-1:end-(window_len-2)));
    w = kaiser(window_len, 2);
    y = conv(w./sum(w(:)), s)(window_len:end-(window_len-1));
    y = y(6:end-5);
end

