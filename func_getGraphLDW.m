function [L,D,W] = func_getGraphLDW(data,knnmat)
%% ������ݵ��ڽӾ���
% Input
%       data������
% Output
    [~,n] = size(data);
    D = zeros(n,n);
    W = zeros(n,n);
    for i = 1:n
        for j = 1:n
            if knnmat(i, j) == 1 || knnmat(j, i) == 1
                W(i, j) = 1;
            end
        end
        D(i,i) = sum(W(i,:));
    end
    L = D - W;
end