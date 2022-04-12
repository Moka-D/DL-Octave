function [x_train, t_train, x_test, t_test] = load_mnist_data(normalize, flatten, one_hot_label)
    %load_mnist_data MNISTデータのロード関数
    %
    % Parameters
    % ----------
    % normalize : logical
    %   画像のピクセル値を0.0～0.1に正規化する
    % flatten : logical
    %   画像を1次元配列に平らにするかどうか
    % one_hot_label : logical
    %   trueなら、ラベルはlogical配列として返す
    %
    % Returns
    % -------
    % x_train : vector/matrix(numeric)
    %   学習データ画像
    % t_train : matrix(numeric/logical)
    %   学習データラベル
    % x_test : vector/matrix(numeric)
    %   推論データ画像
    % t_test : matrix(numeric/logical)
    %   推論データラベル

    % デフォルト引数設定
    if ~exist('normalize', 'var')
        normalize = true;
    end
    if ~exist('flatten', 'var')
        flatten = true;
    end
    if ~exist('one_hot_label', 'var')
        one_hot_label = false;
    end

    set_global_var();
    save_file = get_save_file();

    if ~exist(save_file, 'file')
        dataset = initialize_mnist();
    else
        fprintf('Loading mat file ...\n');
        dataset = load(save_file);
        fprintf('Done.\n');
    end

    x_train = dataset.train_img;
    t_train = dataset.train_label;
    x_test  = dataset.test_img;
    t_test  = dataset.test_label;

    if normalize
        x_train = x_train ./ 255.0;
        x_test  = x_test ./ 255.0;
    end

    if one_hot_label
        t_train = change_one_hot_label(t_train);
        t_test  = change_one_hot_label(t_test);
    end

    if ~flatten
        x_train = reshape(x_train.', 28, 28, []);
        x_test  = reshape(x_test.',  28, 28, []);
    end
end


function set_global_var()
    global files
    files = { ...
        'train-images-idx3-ubyte.gz'; ...
        'train-labels-idx1-ubyte.gz'; ...
        't10k-images-idx3-ubyte.gz'; ...
        't10k-labels-idx1-ubyte.gz'; ...
    };

    global save_file
    save_file = char(fullfile(get_script_dir(), 'mnist.mat'));
end


function out = get_files()
    global files
    out = files;
end


function out = get_save_file()
    global save_file
    out = save_file;
end


function dataset = initialize_mnist()
    download_mnist();
    dataset = convert_mat();
    fprintf('Saving mat file ...\n');
    save_file = get_save_file();
    save(save_file, '-struct', 'dataset');
    fprintf('Done.\n');
end


function download_mnist()
    files = get_files();
    for i_file = 1:length(files)
        download_file(char(files(i_file)));
    end
end


function download_file(file_name)
    url_base = 'http://yann.lecun.com/exdb/mnist/';
    file_path = char(fullfile(get_script_dir(), file_name));

    if exist(file_path, 'file')
        return;
    end

    fprintf('Downloading %s ...\n', file_name);
    urlwrite(strcat(url_base, file_name), file_path);
    fprintf('Done.\n');
end


function dataset = convert_mat()
    files = get_files();
    dataset.train_img   = load_img(char(files(1)));
    dataset.train_label = load_label(char(files(2)));
    dataset.test_img    = load_img(char(files(3)));
    dataset.test_label  = load_label(char(files(4)));
end


function data = load_img(file_name)
    file_name = char(fullfile(get_script_dir(), file_name));
    img_file = char(gunzip(file_name));
    [file_id, msg] = fopen(img_file, 'r', 'b');
    if file_id < 0
        fclose(file_id);
        delte(img_file);
        error(msg);
    end

    % Read the magic number.
    magic_num = fread(file_id, 1, 'int32', 0, 'b');
    if magic_num == 2051
        fprintf('Read MNIST image data ...\n');
    end

    % Read the number of images, rows, cols.
    num_imgs = fread(file_id, 1, 'int32', 0, 'b');
    fprintf('Number of images in the dataset: %6d ...\n', num_imgs);
    num_rows = fread(file_id, 1, 'int32', 0, 'b');
    num_cols = fread(file_id, 1, 'int32', 0, 'b');
    fprintf('Each image is of %2d by %2d pixels ...\n', num_rows, num_cols);

    % Read the image data.
    data = fread(file_id, inf, 'unsigned char');

    % Reshape the data
    data = reshape(data, num_cols, num_rows, num_imgs);
    data = permute(data, [2 1 3]);

    % flatten
    data = reshape(data, num_cols * num_rows, num_imgs).';
    fprintf(['The image data is read to a matrix of dimensions: %6d by %4d...\n',...
    'End of reading image data.\n'], size(data, 1), size(data, 2));

    % Close the file
    fclose(file_id);
    delete(img_file);
end


function labels = load_label(file_name)
    file_name = char(fullfile(get_script_dir(), file_name));
    label_file = char(gunzip(file_name));
    [file_id, msg] = fopen(label_file, 'r', 'b');
    if file_id < 0
        fclose(file_id);
        delte(label_file);
        error(msg);
    end

    % Read the magic number.
    magic_num = fread(file_id, 1, 'int32', 0, 'b');
    if magic_num == 2049
        fprintf('Read MNIST label data ...\n');
    end

    num_items = fread(file_id, 1, 'int32', 0, 'b');
    fprintf('Number of labels in the dataset: %6d ...\n', num_items);

    labels = fread(file_id, inf, 'unsigned char');
    fprintf(['The label data is read to a matrix of dimensions: %6d by %2d...\n', ...
        'End of reading label data.\n'], size(labels, 1), size(labels, 2));

    % Close the file
    fclose(file_id);
    delete(label_file);
end


function out = change_one_hot_label(X)
    out = zeros(size(X, 1), 10);
    for ri = 1:size(X, 1)
        out(ri, X(ri, :) + 1) = 1;
    end
end


function path = get_script_dir()
    path = mfilename('fullpath');
    [path, ~, ~] = fileparts(path);
    path = char(path);
end
