function [out] = func_thrsinval(data, thr)
%% threshold the singular value with PCA
% input: data \in \mathbb{R}^{n \times d}
% output: out \in \mathbb{R}^{n \times r}
% where r << d

%% function body
    S = cov(data);
    [V, D] = eig(S);
    [D_sorted,ind] = sort(diag(D),'descend');
    Ds = D(ind,ind);
    Vs = V(:,ind);

    % 计算总和和98%的总和
    total = sum(D_sorted);
    threshold = thr * total;
    
    % 找到累积和达到阈值的位置
    cumulative_sum = cumsum(D_sorted);
    idx = find(cumulative_sum >= threshold, 1);
    
    % 保留前idx个特征值和特征向量
    D_filtered = D_sorted(1:idx);
    V_filtered = Vs(:, 1:idx);

    out = data * V_filtered;

end 