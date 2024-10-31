function [W] = func_RR(X, Y, alpha)
%% ridge regression
% input X : training sample d*n
%       Y : training label c*n, namely, onehot
%       alpha : balanced parameter
% output W : projection matrix d*c

%% function body
    [d, ~] = size(X);
    I_d = eye(d);
    W = inv(X * X' + alpha * I_d) * X * Y';
end