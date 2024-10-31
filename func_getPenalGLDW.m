function [L_p,D_p,W_p] = func_getPenalGLDW(data,knnmat)
%% 获得数据的邻接矩阵
% Input
%       data：数据
%       knnmat：k nearest neighbors information
% Output
    [~,n] = size(data);
    D_p = zeros(n,n);
    W_p = zeros(n,n);
    for i = 1:n
        for j = 1:n
            if knnmat(i, j) == 1 || knnmat(j, i) == 1
                W_p(i, j) = 1;
            end
        end
        D_p(i,i) = sum(W_p(i,:));
    end
    L_p = D_p - W_p;
end