function [A] = func_kernelLPP(X,L,option)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%   X \in R^{d \times n} is the data matrix
%   L \in R^{n \times n} is a normalized Laplacian matrix

% function body
    ker_option = [];
    ker_option.KernelType = 'Gaussian';
    ker_option.t = 1e+3; 
    ker_option.d = 1;
    K = constructKernel(X', X', ker_option);
    K_T_LK = K' * L * K;
    
    switch lower(option.orthoType)
        case {lower('PPI')}
            [eig_vec, eig_val] = eig(K_T_LK + 1e-5 * eye(size(K_T_LK)));
        otherwise
            [eig_vec, eig_val] = eig(K_T_LK + 1e-5 * eye(size(K_T_LK)), K^2);
    end

    % 过滤值小于1e-3的元素
    eigIdx = find(diag(eig_val) < 1e-3);
    eig_val(eigIdx, :) = [];
    eig_val(:, eigIdx) = [];
    eig_vec(:,eigIdx) = [];

    [~, ind] = sort(diag(eig_val), 'ascend');
    eig_vec_Sorted = eig_vec(:, ind);
    
    A = eig_vec_Sorted(:, 1:option.r);

end