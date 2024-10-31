function [pca_mat,acc] = func_PCA(train_data,train_label,test_data,test_label,...
    pca_d,k)
    %% Introduction
%     �������ݵĳߴ�Ӧ��Ϊ ά��x����
    %% �����ֽ�
    S_cov = cov(train_data');
    [V,D] = eig(S_cov);
    [~,ind] = sort(diag(D),'descend');
    Ds = D(ind,ind);
    Vs = V(:,ind);

    %% ��ά
    pca_value = Ds(1:pca_d,1:pca_d);
    pca_mat = Vs(:,1:pca_d);

    %% �ع�
    test_data_y = pca_mat' * test_data;

    %% k_NN����
%     re_label = func_kNN(test_data_y,pca_mat' * train_data,train_label,k);

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