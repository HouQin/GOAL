function [out_mat, train_feature, test_feature] = func_KDAr(X,...
            Y, test_X, k, sigma, epsilon)
%% kernel discriminant analysis for regression
% input X : the data matrix, d*n
%       Y : the label matrix 1*n
%       test_X : the test data, d * n'
%       k : the low dimension
%       sigma : for gaussion kernel
%       epsilon : for close-far matrix Aw and Ab, simply, it can be 1.
% output out_mat : d * k
%% function body
    % training
    [~, n] = size(X);
    unc_kmat = zeros(n, n);
    % simply apply gaussian kernel
    for i = 1:n
        for j = 1:n
            this_norm = norm(X(:, i) - X(:, j));
            unc_kmat(i, j) = exp(- this_norm * this_norm / (sigma));
        end
    end
    e = ones(n, 1);
    one_n = (1/n) * (e * e');
    ce_kmat = unc_kmat - one_n * unc_kmat - unc_kmat * one_n +...
        one_n * unc_kmat * one_n;
    Aw = zeros(n, n);
    for i = 1:n
        for j = 1:n
            this_card = abs(Y(i) - Y(j));
            if this_card <= epsilon
                Aw(i, j) = 1;
            end
        end
    end
    W_w = zeros(n, n);
    W_b = zeros(n, n);
    D_w = zeros(n, n);
    D_b = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if Aw(i, j) == 1
                W_w(i, j) = 1;
            else
                W_b(i, j) = 1;
            end
        end
    end
    for i = 1:n
        D_w(i, i) = sum(W_w(i, :));
        D_b(i, i) = sum(W_b(i, :));
    end
    L_w = D_w - W_w;
    L_b = D_b - W_b;
    
    S_bk = ce_kmat * L_b * ce_kmat;
    S_wk = ce_kmat * L_w * ce_kmat;
    S_wk = S_wk + 1e-5 * eye(size(S_wk));
    [V, D] = eig(S_bk, S_wk);
    [~, ind] = sort(diag(D), 'descend');
    Ds = D(ind, ind);
    Vs = V(:, ind);
    out_mat = Vs(:, 1:k);
    train_feature = out_mat' * ce_kmat;
    
    %test
    [~, test_n] = size(test_X);
    test_feature = zeros(k, test_n);
    for i = 1:test_n
        this_kx = zeros(n, 1);
        for j = 1:n
            this_norm = norm(test_X(:, i) - X(:, j));
            this_kx(j) = exp(- this_norm * this_norm / (sigma));
        end
        test_feature(:, i) = out_mat' * this_kx;
    end
end