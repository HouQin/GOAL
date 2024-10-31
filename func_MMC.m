function [proj_mat] = func_MMC(train_data,train_label,dim)
    %% Introduction
%     输入数据的维度应该是 样本x维度
    train_data_dd = train_data;
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
    S_w = S_w + 1e-5 * eye(size(S_w));
    S_b = H_b' * H_b;

    [lda_V,lda_D] = eig(S_b - S_w);
    [~,lda_ind] = sort(diag(lda_D),'descend');
    lda_Ds = lda_D(lda_ind,lda_ind);
    lda_Vs = lda_V(:,lda_ind);

    lda_mat = lda_Vs(:,1:dim);
    proj_mat = lda_mat;

end