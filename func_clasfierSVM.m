function [label] = func_clasfierSVM(data, data_ori, label_ori, para, method)
%% SVM 分类器
% input data : test data without label [d*n]
%       data_ori : train data [d*n]
%       label_ori : train label n*1
%       para : is a vector. Basically, para = [alpha, beta, gamma, ...]
% output label : test label

%% function body
[~, num_c] = size(unique(label_ori'));
[d, test_n] = size(data);
list(num_c * (num_c - 1) / 2) = struct('alpha', [],...
                                       'w', [],...
                                       'b', [],...
                                 'i', [], 'j', []);
k = 1;
for i = 1:num_c
    for j = (i+1):num_c
            list(k).i = i;
            list(k).j = j;
            k = k + 1;
    end
end

if method == 1
    % hard margin SVM
    parfor c = 1:(num_c * (num_c - 1) / 2)
        this_i = list(c).i;
        this_j = list(c).j;
        ind_pos = find(label_ori == this_i);
        ind_neg = find(label_ori == this_j);
        this_data = [data_ori(:, ind_pos),data_ori(:, ind_neg)];
        thislabel = [label_ori(ind_pos);label_ori(ind_neg)];
        thislabel(thislabel == this_j) = -1;
        thislabel(thislabel == this_i) = 1;
        
        [alpha, w, b] = func_HSVM(this_data, thislabel);
        list(c).alpha = alpha;
        list(c).w = w;
        list(c).b = b;
    end

    w_matrix = cat(2, list.w)';
    b_vector = [list.b]; % 创建一个包含所有b值的向量
    b_vector = repmat(b_vector, test_n, 1)';
    
    func_value_matrix = w_matrix * data + b_vector;
    
    % 创建一个新的矩阵，其大小与func_value_matrix相同
    new_matrix = zeros(size(func_value_matrix));
    
    % 对于func_value_matrix中大于0的元素，将对应的new_matrix元素设置为list(j).i
    for j = 1:num_c * (num_c - 1) / 2
        new_matrix(j, func_value_matrix(j,:) > 0) = list(j).i;
    end
    
    % 对于func_value_matrix中小于等于0的元素，将对应的new_matrix元素设置为list(j).j
    for j = 1:num_c * (num_c - 1) / 2
        new_matrix(j, func_value_matrix(j,:) <= 0) = list(j).j;
    end

    label = mode(new_matrix);
else
    % least square SVM
    parfor c = 1:(num_c * (num_c - 1) / 2)
        this_i = list(c).i;
        this_j = list(c).j;
        ind_pos = find(label_ori == this_i);
        ind_neg = find(label_ori == this_j);
        this_data = [data_ori(:, ind_pos),data_ori(:, ind_neg)];
        thislabel = [label_ori(ind_pos);label_ori(ind_neg)];
        thislabel(thislabel == this_j) = -1;
        thislabel(thislabel == this_i) = 1;
        
        [alpha, w, b] = func_LSSVM(this_data, thislabel, para(3));
        list(c).alpha = alpha;
        list(c).w = w;
        list(c).b = b;
    end
    w_matrix = cat(2, list.w)';
    b_vector = [list.b]; % 创建一个包含所有b值的向量
    b_vector = repmat(b_vector, test_n, 1)';
    
    func_value_matrix = w_matrix * data + b_vector;

    % 创建一个新的矩阵，其大小与func_value_matrix相同
    new_matrix = zeros(size(func_value_matrix));
    
    % 对于func_value_matrix中大于0的元素，将对应的new_matrix元素设置为list(j).i
    for j = 1:num_c * (num_c - 1) / 2
        new_matrix(j, func_value_matrix(j,:) > 0) = list(j).i;
    end
    
    % 对于func_value_matrix中小于等于0的元素，将对应的new_matrix元素设置为list(j).j
    for j = 1:num_c * (num_c - 1) / 2
        new_matrix(j, func_value_matrix(j,:) <= 0) = list(j).j;
    end

    label = mode(new_matrix);

end
end