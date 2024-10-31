function [W] = func_SCM(X, Y, gamma, max_iter)
%% robust feature selection via similtaneous capped...(SCM)
% input X : training data, d*n
%       Y : training label c*n
% output W : projection matrix d*c

%% function body
    [d, n] = size(X);
    F = eye(n);
    D = eye(d);
    for i = 1:max_iter
        W = inv(X * F * X' + gamma * D) * X * F * Y';
        norm_list = zeros(1, n);
        for j = 1:n
            norm_list(j) = norm(W' * X(:, j) - Y(:, j));
        end
        norm_mean = mean(norm_list);
        norm_std = std(norm_list);
        for j = 1:n
            if norm_list(j) < norm_mean + 2 * norm_std
                F(j, j) = 1 / (2 * norm_list(j));
            end
        end
        for j = 1:d
            D(j, j) = 1 / (2 * norm(W(j, :)));
        end
    end
end