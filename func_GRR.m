function [B] = func_GRR(X, Y, W, k, beta, gamma, lamda, max_step)
%% GRR
% input X : the training data n*d
%       Y : training label    n*c
%       W : neighbor matrix   n*n
%       k : the objective dimension
%       beta, gamma, lamda are the parameters
%       max_step : for iteration
% output B : low dimensional discriminative subspace

%% function body
[n, d] = size(X);
[~, c] = size(Y);
A = randn(c, k);
B = randn(d, k);
D_wav = eye(d);
D_hat = eye(n);
D = zeros(n, n);
vect_one = ones(n, 1);
G = eye(n);

% 预先计算 X * B * A'
XBA = X * B * A';

% 创建一个三维矩阵，其中每个切片都是 XBA
XBA_repeated = repmat(XBA, [1, 1, n]);

% 计算每两行之间的差异
diffs = permute(XBA_repeated, [3, 2, 1]) - XBA_repeated;

% 计算范数
G = sqrt(sum(diffs .^ 2, 2));

% 将结果转换回二维矩阵
G = squeeze(G);

% 将对角线元素设置为0
G(1:n+1:end) = 1e-5;

ele_wise_mat = W ./ G;
for i = 1:n
    D(i, i) = sum(ele_wise_mat(i, :));
end
step = 1;
converged = false;

while ~converged && step <= max_step
    s = vect_one' * D_hat * vect_one + lamda;
    h = (Y' * D_hat * vect_one - A * B' * X' * D_hat * vect_one) / s;
    B_former = B;
    B = gamma *( (beta * D_wav + X'* ((D - ele_wise_mat) + gamma * D_hat) * X + 1e-3*eye(size(D_wav))) \...
        X' * D_hat * (Y - vect_one * h') * A);
    S_mat = (h * vect_one' - Y') * D_hat * X * B;
    [U,~,V] = svd((S_mat + 1e-5 * eye(size(S_mat))),'econ');
    A = U * V';
    % 预先计算 X * B * A'
    XBA = X * B * A';
    
    % 创建一个三维矩阵，其中每个切片都是 XBA
    XBA_repeated = repmat(XBA, [1, 1, n]);
    
    % 计算每两行之间的差异
    diffs = permute(XBA_repeated, [3, 2, 1]) - XBA_repeated;
    
    % 计算范数
    G = sqrt(sum(diffs .^ 2, 2));
    
    % 将结果转换回二维矩阵
    G = squeeze(G);
    
    % 将对角线元素设置为0
    G(1:n+1:end) = 1e-5;

    ele_wise_mat = W ./ G;
    for i = 1:n
        D(i, i) = sum(ele_wise_mat(i, :));
    end
    regs_mat = X * B * A' + vect_one * h' - Y;
    for i = 1:d
        D_wav(i, i) = 1 / (2 * norm(B(i, :)) + 1e-5);
    end
    for i = 1:n
        D_hat(i, i) = 1 / (2 * norm(regs_mat(i, :)) + 1e-5);
    end
    step = step + 1;
    if norm(B - B_former) < 1e-3
        converged = true;
    end
end
end