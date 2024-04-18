function [B_mat,A_mat,h_vec,E] = func_GOAL(X,Y,d,W,...
    alpha,beta,eta,gamma,mu,sig_mul,max_Iter)
%% GOAL
    %{
    Input: data matrix [X] \in \mathbb{R}^{m \times n}
            m features, n samples, dtype=double;
           label matrix [Y] \in \mathbb{R}^{c \times n}
            c class, dtype=double;
           dimention of subspace [d], e.g., 50, 100,...;
           adjacency [W] \in \mathbb{R}^{n \times n};
           parameters [alpha],[beta],[eta],[gamma],[mu] dtype=double;
           paramter [sig_mul] dtype=int;
           #Iteration [max_Iter], (optimal)>=4;
    Output: projection matrix [B_mat] \in \mathbb{R}^{m \times d};
            projection matrix [A_mat] \in \mathbb{R}^{c \times d};
            bias vector [h_vec] \in \mathbb{R}^c
            embedding in latent space [E] \in \mathbb{R}^{n \times d};
    Usage: 
        [B, A, h, E] = func_GOAL(X, Y, d, W, alpha, beta, eta, gamma,
                mu, sig_mul, 4);
        low_dimentional_feature = B' * data_matrix;
    %}
    [m,~] = size(X);
    [c,n] = size(Y);

    B_mat = randn(m,d);
    A_mat = eye(c,d);
    h_vec = zeros(c,1);
    e_vec = ones(n,1);
    epsilon = zeros(1, max_Iter);

    L = diag(sum(W)) - W;
    E = eye(n, d);
    
    % construct intra- & inter- class sample matrices
    X_wi = zeros(m,n);
    X_bt = zeros(m,c);
    x_c_mean = zeros(m,c);
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
    
    %% Train iteration
    if c < d
        for t = 1:max_Iter
            % initial diagonal matrices
            D_wi = zeros(n,n);
            D_bt = zeros(c,c);
            D_rr = zeros(n,n);
            D_b = zeros(m,m);
            norm_wi = zeros(1, n);
            for i = 1:n
                norm_wi(i) = norm(X_wi(:,i)' * B_mat * A_mat');
            end
            norm_wi_m = mean(norm_wi);
            norm_wi_std = std(norm_wi);
            epsilon(t) = norm_wi_m + sig_mul * norm_wi_std;
            for i = 1:n
                if norm_wi(i) < epsilon(t)
                    D_wi(i,i) = 1 / (2 * norm_wi(i));
                end
            end
            for i = 1:c
                norm_bt = norm(X_bt(:,i)' * B_mat * A_mat');
                D_bt(i,i) = 1 / (2 * norm_bt);
            end
            rr_mat = X' * B_mat * A_mat' + e_vec * h_vec' - Y';
            for i = 1:n
                norm_rr = norm(rr_mat(i,:));
                D_rr(i,i) = 1 / (2 * norm_rr);
            end
            for i = 1:m
                norm_b = norm(B_mat(i,:));
                D_b(i,i) = 1 / (2 * norm_b);
            end

            % update h
            h_vec = gamma * inv(gamma * e_vec' * D_rr * e_vec + beta) * (Y - A_mat * B_mat' * X) * D_rr * e_vec;

            % update B
            B_mat_cop = zeros(size(B_mat));

            S_cal = X_wi * D_wi * X_wi' - eta * X_bt * D_bt * X_bt' + gamma * X * D_rr * X';
            Q_cal = gamma * X * D_rr * e_vec * h_vec' * A_mat - gamma * X * D_rr * Y'* A_mat + mu * X * L * E;
            B_mat_cop(:, 1:c) = (S_cal + alpha * D_b + 1e-3 * eye(size(D_b))) \ (- Q_cal(:, 1:c));
            B_mat_cop(:, c+1:d) = (alpha * D_b + 1e-3 * eye(size(D_b))) \ (- Q_cal(:, c+1:d));

            B_mat = B_mat_cop;

            % update E
            Se_mat = L' * X' * B_mat;
            [Ue, ~, Ve] = svd(Se_mat, 'econ');
            E = Ue * Ve';

            % update A
            S_mat = (Y - h_vec * e_vec') * D_rr * X' * B_mat;
            [U,~,V] = svd(S_mat,'econ');

            A_mat = U * V';
            
            % according to coro
            diag_elements = diag(A_mat'*A_mat);
            [~, sorted_indices] = sort(diag_elements, 'descend');
            A_mat = A_mat(:, sorted_indices);
            zero_mat = zeros(c, d);
            zero_mat(:, 1:c) = A_mat(:, 1:c);
            A_mat = zero_mat;

        end
    else
        for t = 1:max_Iter
            % initial diagonal matrices
            D_wi = zeros(n,n);
            D_bt = zeros(c,c);
            D_rr = zeros(n,n);
            D_b = zeros(m,m);
            norm_wi = zeros(1, n);
            for i = 1:n
                norm_wi(i) = norm(X_wi(:,i)' * B_mat * A_mat');
            end
            norm_wi_m = mean(norm_wi);
            norm_wi_std = std(norm_wi);
            epsilon(t) = norm_wi_m + sig_mul * norm_wi_std;
            for i = 1:n
                if norm_wi(i) < epsilon(t)
                    D_wi(i,i) = 1 / (2 * norm_wi(i));
                end
            end
            for i = 1:c
                norm_bt = norm(X_bt(:,i)' * B_mat * A_mat');
                D_bt(i,i) = 1 / (2 * norm_bt);
            end
            rr_mat = X' * B_mat * A_mat' + e_vec * h_vec' - Y';
            for i = 1:n
                norm_rr = norm(rr_mat(i,:));
                D_rr(i,i) = 1 / (2 * norm_rr);
            end
            for i = 1:m
                norm_b = norm(B_mat(i,:));
                D_b(i,i) = 1 / (2 * norm_b);
            end

            % update h
            h_vec = gamma * inv(gamma * e_vec' * D_rr * e_vec + beta) * (Y - A_mat * B_mat' * X) * D_rr * e_vec;

            % update B
            inv_b = inv(X_wi * D_wi * X_wi' - eta * X_bt * D_bt * X_bt' + gamma * X * D_rr * X' + alpha * D_b + 1e-3 * eye(size(D_b)));
            B_mat = gamma * inv_b * ((-X * D_rr * e_vec * h_vec' + X * D_rr * Y') * A_mat - (mu * X * L * E));

            % update E
            Se_mat = L * X' * B_mat;
            [Ue, ~, Ve] = svd(Se_mat, 'econ');
            E = Ue * Ve';

            % update A
            S_mat = (Y - h_vec * e_vec') * D_rr * X' * B_mat;
            [U,~,V] = svd(S_mat,'econ');
            A_mat = U * V';
        end
    end
    
end

