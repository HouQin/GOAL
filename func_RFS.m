function [W] = func_RFS(X, Y, alpha, maxStep)
%% RFS by feiping nie
% input X : training sample d*n
%       Y : training label c*n, namely, onehot
%       alpha : balanced parameter
% output W : projection matrix d*c

%% function body
    [d, n] = size(X);
    [c, ~] = size(Y);
    D_rr = eye(n);
    D_w = eye(d);
    W = randn(d, c);
    for step = 1:maxStep
        Y_XW = Y' - X' * W;
        for i = 1:n
            D_rr(i, i) = 1 / (2 * norm(Y_XW(i, :)) + 1e-5);
        end
        for i = 1:d
            D_w(i, i) = 1 / (2 * norm(W(i, :)) + 1e-5);
        end
        W = inv(X * D_rr * X' + alpha * D_w) * X * D_rr * Y';
    end
end