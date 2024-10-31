function [alpha, w, b, ei] = func_LSSVM(x, y, gamma)
%% 最小均方SVM
% input x : the original data
%       y : the original label
%       gamma : balanced parameter
% output alpha : support vector
%        w : weight
%        b : bias
%        ei : alpha = gamma * ei

%% function body
[d, n] = size(x);
Q = zeros(n, n);

for i = 1:n
    for j = 1:n
        Q(i, j) = x(:, i)' * x(:, j);
        Q(i, j) = y(i) * y(j) * Q(i, j);
    end
end

neg_one = ones(n, 1);
neg_one = -1 * neg_one;
w = zeros(d, 1);
b = 0;
D = eye(n);
D = (1 / gamma) * D;
Q = Q + D;

alpha = quadprog(Q, neg_one, [], [], y', 0);

parfor i = 1:n
    w = w + alpha(i) * y(i) * x(:, i);
end

inter_sum = 0;
for i = 1:n
    intra_sum = 0;
    for j = 1:n
        intra_sum = intra_sum + alpha(j) * y(j) * x(:, j)' * x(:, i);
    end
    inter_sum = inter_sum + (1 - alpha(i)/gamma) / y(i) - intra_sum;
end
b = inter_sum / n;
end