function [U, V, loss] = func_discrNMF(X, Y, k, eta, alpha, max_iter)
%% the method proposed in 2022/5/30
% It is a method 
% \begin{equation}
% 	\begin{aligned}
% 		\min_{U, V} &\left \| X-UV^{T} \right \|_{2,1}+\eta \left \| 
% 		X_{WI}^{T}U \right \|_{2,1}+\alpha 
% 		\left \| U \right \|_{2,1} \\
% 		&\mathrm{s.t.}~U\ge 0, V\ge 0  
% 		\end{aligned}
% \end{equation}
% input X : data d*n
%       Y : label c*n
%       k : low dimension
%       eta, alpha : para. for model
% Output U: d*k, V:n*k

%% function body
    [d,n] = size(X);
    [c,n] = size(Y);
    loss = zeros(1, max_iter);

    U = rand(d,k);
    V = rand(n,k);
    
    % 根据标签信息获得类内样本矩阵以及类间样本矩阵
    X_wi = zeros(d,n);
    X_bt = zeros(d,c);
    x_c_mean = zeros(d,c);
    x_mean = mean(X,2);
    for i = 1:c
        ind = find(Y(i,:));
        x_c_mean(:,i) = mean(X(:,ind),2);
    end
    
    for i = 1:n
        c_ind = find(Y(:,i));
        X_wi(:,i) = X(:,i) - x_c_mean(:,c_ind);
    end
    
    for i = 1:c
        X_bt(:,i) = x_c_mean(:,i) - x_mean;
    end
    
    D_uv = eye(d);
    D_wi = eye(n);
    D_u = eye(d);
    D_bt = eye(c);
    
    for t = 1:max_iter
        loss(t) = norm(X-U*V');
        mat_XsubUV = X - U * V';
        for i = 1:d
            D_uv(i, i) = 1 / (2 * norm(mat_XsubUV(i, :)));
        end
        mat_XwiU = X_wi' * U;
        for i = 1:n
            D_wi(i, i) = 1 / (2 * norm(mat_XwiU(i, :)));
        end
        mat_XbtU = X_bt' * U;
        for i = 1:c
            D_bt(i, i) = 1 / (2 * norm(mat_XbtU(i, :)));
        end
        for i = 1:d
            D_u(i, i) = 1 / (2 * norm(U(i, :)));
        end
        upp_2_u = D_uv * X * V + eta * X_bt * D_bt * X_bt' * U;
        low_2_u = (D_uv * U) * (V') * V + eta * X_wi * D_wi * X_wi' * U +...
            alpha * D_u * U;
        for i = 1:d
            for kk = 1:k
                U(i, kk) = U(i, kk) * (upp_2_u(i, kk) / low_2_u(i, kk));
            end
        end
        upp_2_v = X' * D_uv * U;
        low_2_v = V * U' * D_uv * U;
        for j = 1:n
            for kk = 1:k
                V(j, kk) = V(j, kk) * (upp_2_v(j, kk) / low_2_v(j, kk));
            end
        end
    end
end