function file_path = get_file(url, file_name)
    % file_path = mdl.utils.get_file(url, file_names)
    %
    % Download a file from the 'url' if it is not in the cache.
    %
    % The file at the 'url' is downloaded to the "./.tmp".
    %
    % Parameters
    % ----------
    % url (str) : URL of the file.
    % file_name (str) : Name of the file. It "" is specified the original
    %     file name is used.
    %
    % Returns
    % -------
    % file_path (str) : Absolute path to the saved file.

    if ~exist('file_name', 'var')
        file_name = '';
    end

    cache_dir = '.tmp';

    if isempty(file_name)
        splitted_url = split(url, '/');
        file_name = splitted_url{end};
    end
    file_path = fullfile(cache_dir, file_name);

    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end

    if exist(file_path, 'file')
        return
    end

    fprintf('Downloading: %s\n', file_name);
    try
        websave(file_path, url);
    catch ME
        if exist(file_path, 'file')
            delete(file_path);
        end
        rethrow(ME);
    end
    disp(" Done")
end
