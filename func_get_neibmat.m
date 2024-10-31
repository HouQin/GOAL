function [L,D,W] = func_get_neibmat(data,delta,knnmat)
%% ������ݵ��ڽӾ���
% Ĭ�ϸ�˹����
% Input
%       data������
%       delta����˹��������
% Output
    [~,n] = size(data);
    D = zeros(n,n);
    W = zeros(n,n);
    for i = 1:n
        for j = 1:n
            x = norm(data(:,i) - data(:,j));
            W(i,j) = exp(-(x * x) / (2 * delta * delta));
        end
        D(i,i) = sum(W(i,:));
    end
    L = D - W;
end