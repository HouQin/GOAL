function [Y, B, A] = func_JSER(X, Z, c, intraK, interK, alpha, beta, lamd, max_step)
%% JSER under the GER framework£¬by Jianglin Lu
% input X : traing samples, n*d
%       Z : label matrix n*k, where k is the number of classes
%       c : the dimension of subspace
%       intraK : similar to LPP's L matrix. we need the KNN information
%                 in whole samples group
%       interK : similar to MFA. we need to get the KNN information 
%                   in different class
%       alpha, beta, lamd are set for balanced parameters.
%       max_step : need to say?

%% function body
    % step 1
    [n, d] = size(X);
    [~, k] = size(Z);
    Y = randn(n, c);
    A = zeros(k, c);
    B = zeros(d, c);
    U_0 = eye(d);
    U_1 = eye(n);
    U_2 = eye(n);
    
    % step 2
    graphknn = func_getKNNMat(X', intraK);
    [L, ~, ~] = func_getGraphLDW(X', graphknn);
    pegraphknn = func_getInterKNN(X', Z', interK);
    [L_p, ~, ~] = func_getPenalGLDW(X', pegraphknn);
    
    % step 3
    [K, E, ~] = svd(L_p);
    wait_demat = inv(K * E^(1/2)) * (L + alpha * U_1) * inv(E^(1/2) * K');
    [P, Q, ~] = svd(wait_demat);
    
    for step = 1:max_step
        B = inv(X' * (alpha * U_1 + lamd * U_2) * X + beta * U_0) * X' *...
            (alpha * U_1 * Y + lamd * U_2 * Z * A);
        wait_demat = 2 * lamd * Z' * U_2 * X * B;
        [XiTa, ~, GaMma] = svd(wait_demat, 'econ');
        A = XiTa * GaMma';
        A_1 = Q^(1/2) * P';
        A_2 = inv((Q^(1/2) * P')') * inv(K * E^(1/2)) * alpha * U_1 *...
            X * B;
        [U, ~, V] = svd(A_1' * A_2, 'econ');
        M = U * V';
        Y = inv(E^(1/2) * K') * M;
        for i = 1:d
            U_0(i, i) = 1 / (2 * norm(B(i, :)) + 1e-5);
        end
        ob_Znorm = Z - X * B * A';
        ob_Ynorm = Y - X * B;
        for i = 1:n
            U_1(i, i) = 1 / (2 * norm(ob_Ynorm(i, :)) + 1e-5);
            U_2(i, i) = 1 / (2 * norm(ob_Znorm(i, :)) + 1e-5);
        end
    end
end