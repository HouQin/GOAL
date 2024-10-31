function V = func_insertImage(V, rows, cols, image_path)
    % 读取图片
    img = imread(image_path);

    % 转换为灰度图
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    % resize图片
    img = imresize(img, [20, 20]);

    % 对每一列向量进行操作
    for i = 1:size(V, 2)
        % reshape成指定大小的图
        img_large = reshape(V(:, i), [rows, cols]);

        % 随机选择一个位置插入图片
        row = randi([1, rows-19]);
        col = randi([1, cols-19]);
        img_large(row:row+19, col:col+19) = img;

        % 将修改后的图像保存回矩阵中
        V(:, i) = img_large(:);
    end
end
