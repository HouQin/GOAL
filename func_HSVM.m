function [alpha, w, b] = func_HSVM(x, y)
%% ”≤º‰∏ÙSVM
% input: x - the original data
%        y - the original label
% output: alpha - the support vector
%         w - weight
%         b - bias

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
z = zeros(n, 1);
w = zeros(d, 1);

alpha = quadprog(Q, neg_one, [], [], y', 0, z, []);

parfor i = 1:n
    w = w + alpha(i) * y(i) * x(:, i);
end

zero_norm = 0;
inter_mul = 0;
for i = 1:n
    if alpha(i) ~= 0
        intra_mul = 0;
        zero_norm = zero_norm + 1;
        for j = 1:n
            intra_mul = intra_mul + alpha(j) * y(j) * x(:, j)' * x(:, i);
        end
        inter_mul = inter_mul + y(i) - intra_mul;
    else
        continue;
    end
end

b = inter_mul / zero_norm;
end

