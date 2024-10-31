% function [out_mat] = func_getKNNMat(data,k)
% %% 获得数据的K近邻矩阵
% % Input：data，维度d*n
% %        k,取k个近邻点
% % Output：knn_mat，维度k*n，其值代表第i个样本的k近邻的数据下标
% 
% %% function body
%     [~,n] = size(data);
%     knn_mat = zeros(k,n);
%     out_mat = zeros(n,n);
%     for i = 1:n
%         norm_k = zeros(1,k);
%         index_k = zeros(1,k);
%         for j = 1:k
%             norm_k(j) = Inf;
%         end
%         for j = 1:n
%             if (i~=j)
%                 eij = norm(data(:,i)-data(:,j));
%                 if eij < norm_k(k)
%                     norm_k(k) = eij;
%                     index_k(k) = j;
%                 end
%                 [~,sort_ind] = sort(norm_k);
%                 sort_norm_k = norm_k(sort_ind);
%                 sort_index_k = index_k(sort_ind);
%                 norm_k = sort_norm_k;
%                 index_k = sort_index_k;
%             end
%         end
%         knn_mat(:,i) = index_k';
%     end
%     
%     for i = 1:n
%         out_mat(knn_mat(:,i),i) = 1;
%     end
% end

function [out_mat] = func_getKNNMat(data,k)
%% 获得数据的K近邻矩阵
% Input：data，维度d*n
%        k,取k个近邻点
% Output：knn_mat，维度k*n，其值代表第i个样本的k近邻的数据下标

%% function body
    [~,n] = size(data);
    knn_mat = zeros(k,n);
    out_mat = zeros(n,n);

    % 计算所有样本之间的欧氏距离
    distances = pdist2(data', data');

    for i = 1:n
        % 对于每个样本，找到k个最近邻的样本
        [~, indices] = sort(distances(i, :));
        nearest_neighbors = indices(2:k+1);
        
        % 将样本的k个最近邻的数据下标存储在knn_mat中
        knn_mat(:,i) = nearest_neighbors;
    end
    
    for i = 1:n
        out_mat(knn_mat(:,i),i) = 1;
    end
end
