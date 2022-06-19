function [x_train, t_train, x_test, t_test] = load_mnist_data(varargin)
    %load_mnist_data MNISTデータのロード関数
    %
    % [x_train, t_train, x_test, t_test] = dataset.load_mnist_data()
    % [x_train, t_train, x_test, t_test] = dataset.load_mnist_data(Name, Value)
    %
    % Parameters
    % ----------
    % normalize : logical (parameter, default true)
    %   画像のピクセル値を0.0～0.1に正規化する
    % flatten : logical (parameter, default true)
    %   画像を1次元配列に平らにするかどうか
    % one_hot_label : logical (parameter, default false)
    %   trueなら、ラベルは0-1の配列として返す
    %
    % Returns
    % -------
    % x_train : matrix(numeric)
    %   学習データ画像
    % t_train : matrix(numeric)
    %   学習データラベル
    % x_test : matrix(numeric)
    %   推論データ画像
    % t_test : matrix(numeric)
    %   推論データラベル

    p = inputParser;
    addParameter(p, 'normalize',     true,  @islogical);
    addParameter(p, 'flatten',       true,  @islogical);
    addParameter(p, 'one_hot_label', false, @islogical);
    parse(p, varargin{:});

    [x_train, t_train] = prepare(true);
    [x_test,  t_test] = prepare(false);

    if p.Results.normalize
        x_train = single(x_train) ./ 255;
        x_test = single(x_test) ./ 255;
    end

    if p.Results.flatten
        x_train = flatten(x_train);
        x_test = flatten(x_test);
    end

    if p.Results.one_hot_label
        t_train = change_one_hot_label(t_train);
        t_test = change_one_hot_label(t_test);
    end
end

function [data, label] = prepare(train)
    url = 'http://yann.lecun.com/exdb/mnist/';
    train_files = containers.Map({'target', 'label'}, ...
        {'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'});
    test_files = containers.Map({'target', 'label'}, ...
        {'t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'});

    if train
        files = train_files;
    else
        files = test_files;
    end
    data_path = util.get_file(strcat(url, files('target')));
    label_path = util.get_file(strcat(url, files('label')));

    data = load_data(data_path);
    label = load_label(label_path);
end

function data = load_data(filepath)
    unzipped_filepath = char(gunzip(filepath));
    [fid, msg] = fopen(unzipped_filepath, 'r', 'b');

    ME = [];
    try
        if fid < 0
            error(msg);
        end

        % Read the magic number.
        magic_num = fread(fid, 1, 'int32', 0, 'b');
        assert(magic_num == 2051);

        % Read the number of images, rows, cols.
        num_imgs = fread(fid, 1, 'int32', 0, 'b');
        num_rows = fread(fid, 1, 'int32', 0, 'b');
        num_cols = fread(fid, 1, 'int32', 0, 'b');

        % Read the image data.
        data = fread(fid, inf, 'unsigned char=>uint8');

        % Reshape the data
        data = reshape(data, num_cols, num_rows, num_imgs);
        data = permute(data, [2 1 3]);
        [H, W, ~] = size(data);
        data = reshape(data, [H, W, 1, num_imgs]);
    catch ME
        % NOP
    end

    % Close the file
    fclose(fid);
    delete(unzipped_filepath);

    if ~isempty(ME)
        rethrow(ME);
    end
end

function labels = load_label(filepath)
    unzipped_filepath = char(gunzip(filepath));
    [fid, msg] = fopen(unzipped_filepath, 'r', 'b');

    ME = [];
    try
        if fid < 0
            error(msg);
        end

        % Read the magic number.
        magic_num = fread(fid, 1, 'int32', 0, 'b');
        assert(magic_num == 2049);

        num_items = fread(fid, 1, 'int32', 0, 'b');

        labels = fread(fid, inf, 'unsigned char=>uint8');
        labels = reshape(labels, num_items, 1);
        labels = labels + 1;
    catch ME
        % NOP
    end

    % Close the file
    fclose(fid);
    delete(unzipped_filepath);

    if ~isempty(ME)
        rethrow(ME);
    end
end

function out = change_one_hot_label(labels)
    N = length(labels);
    out = zeros([N 10], 'uint8');
    ind = sub2ind([N 10], 1:N, labels.');
    out(ind) = uint8(1);
end

function out = flatten(data)
    N = size(data, 4);
    out = reshape(data, [], N).';
end
