% function [out_mat] = func_getInterKNN(X, Y, k)
% %% as MFA, get the L_p and so on.
% % input X : the sample points d*n
% %       Y : the label matrix  c*n
% %       k : k nearest neighbors scale
% 
% %% function body
%     [~,n] = size(X);
%     knn_mat = zeros(k,n);
%     out_mat = zeros(n,n);
%     for i = 1:n
%         norm_k = zeros(1,k);
%         index_k = zeros(1,k);
%         for j = 1:k
%             norm_k(j) = Inf;
%         end
%         for j = 1:n
%             a = find(Y(:, i));
%             b = find(Y(:, j));
%             if (find(Y(:, i))~=find(Y(:, j)))
%                 eij = norm(X(:,i)-X(:,j));
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

function [out_mat] = func_getInterKNN(X, Y, k)
%% as MFA, get the L_p and so on.
% input X : the sample points d*n
%       Y : the label matrix  c*n
%       k : k nearest neighbors scale

%% function body
    [~,n] = size(X);
    knn_mat = zeros(k,n);
    out_mat = zeros(n,n);

    % 计算所有样本之间的欧氏距离
    distances = pdist2(X', X');

    for i = 1:n
        % 对于每个样本，找到k个最近邻的样本
        [~, indices] = sort(distances(i, :));
        nearest_neighbors = indices(vecnorm(Y(:,i)-Y(:,indices), 2) > 1e-3);
        
        % 将样本的k个最近邻的数据下标存储在knn_mat中
        knn_mat(:,i) = nearest_neighbors(1:k);
    end
    
    for i = 1:n
        out_mat(knn_mat(:,i),i) = 1;
    end
end

