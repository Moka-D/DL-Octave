function col = im2col(input_data, filter_h, filter_w, stride, pad)
    %im2col 画像を行列に変換

    if ~exist('stride', 'var')
        stride = 1;
    end
    if ~exist('pad', 'var')
        pad = 0;
    end

    [H, W, C, N] = size(input_data);
    out_h = fix((H + 2 .* pad - filter_h) / stride) + 1;
    out_w = fix((W + 2 .* pad - filter_w) / stride) + 1;

    img = padarray(input_data, [pad pad], 0);
    col = zeros([out_h, out_w, filter_h, filter_w, C, N]);

    for y = 1:filter_h
        y_max = (y - 1) + stride .* out_h;
        for x = 1:filter_w
            x_max = (x - 1) + stride .* out_w;
            col(:, :, y, x, :, :) = img(y:stride:y_max, x:stride:x_max, :, :);
        end
    end

    col = permute(col, [3 4 5 1 2 6]);
    col = reshape(col, [], N .* out_h .* out_w);
end
