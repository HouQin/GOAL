function [pj_mat,acc] = func_SLDA(train_data,train_label,test_data,test_label,...
    pca_d,dim,maxIter,lambda_reg,k)
% 样本x维度
    %% 特征分解
    S_cov = cov(train_data);
    [pca_V,pca_D] = eig(S_cov);
    [~,pca_ind] = sort(diag(pca_D),'descend');
    pca_Ds = pca_D(pca_ind,pca_ind);
    pca_Vs = pca_V(:,pca_ind);

    %% 降维
    pca_value = pca_Ds(1:pca_d,1:pca_d);
    pca_mat = pca_Vs(:,1:pca_d);

    train_data_dd = train_data * pca_mat;
    ck_laInf = tabulate(train_label);

    %% 分类计算均值向量
    x_mean = mean(train_data_dd,1);
    xc_mean = [];
    for i = 1:ck_laInf(end,1)
        ind = find(train_label == i);
        data_part_mat = train_data_dd(ind,:);
        ci_mean = mean(data_part_mat,1);
        xc_mean = [xc_mean;ci_mean];
    end

    %% 计算Hw，Hb，其中S=H'H，Hw in n * d,Hb in k * d
    enX = [];
    for i = 1:ck_laInf(end,1)
        e_n = ones(ck_laInf(i,2),1);
        enX = [enX;e_n * xc_mean(i,:)];
    end
    H_w = train_data_dd - enX;
    H_b = [];
    for i = 1:ck_laInf(end,1)
        H_b = [H_b;sqrt(ck_laInf(i,2)) * (xc_mean(i,:) - x_mean)];
    end
    S_w = H_w' * H_w;
    S_b = H_b' * H_b;

    %% Cholesky分解得到上三角矩阵R_w, H_w'H_w = R_w'R_w
    R_w = chol(S_w + 1e-5 * eye(size(S_w)));

    %% 迭代
    % initialize A B
    A = eye(pca_d,dim);
    B = eye(pca_d,dim);
    for t = 1:maxIter
%         fprintf('[t = %d]\n',t);
        for i = 1:dim
            MSerror = 0;
            y_head = [H_b * (R_w \ A(:,i));zeros(pca_d,1)];
            W_head = [H_b;sqrt(lambda_reg) * R_w];
%             [beta,fitinfo] = lasso(W_head,y_head,'CV',10);
            [beta,fitinfo] = lasso(W_head,y_head,'lambda',0.0037);
            [~,sort_mse_ind] = sort(fitinfo.MSE);
            MSerror = MSerror + fitinfo.MSE(sort_mse_ind(1));
            B(:,i) = beta(:,sort_mse_ind(1));
        end 
        MSerror = MSerror / dim;
%         fprintf('MSE in [t = %d] is %f\n',t,MSerror);
        [U,~,V] = svd(inv(R_w)' * S_b * B,'econ');
        A = U * V';
    end

    test_data_dd = test_data * pca_mat * B;
    pj_mat = pca_mat * B;                     %d * k

%     %% k_NN分类
%     re_label = func_kNN(test_data_dd',(train_data_dd * B)',train_label,k);

    %% Accuracy
    right_num = 0;
    test_num = size(test_label);
    test_num = test_num(2);
%     for i =1:test_num
%         if test_label(i) == re_label(i)
%             right_num = right_num + 1;
%         end
%     end
    acc = right_num / test_num * 100;
end