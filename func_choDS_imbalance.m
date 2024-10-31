function [train_data,train_label,test_data,test_label,data_d,data_n] = func_choDS_imbalance(apart_r,varargin)
%% 函数获取数据集
% 通过给定的文件路径和相应的数据库名称，加载相应的数据集
% 默认的文件夹是当前路径，默认的数据集是ORL数据集
% 默认的划分参数是0.7
% 输出训练集、训练集标签、测试集、测试集标签、样本维度、样本总数
% 输出的训练矩阵的维度是 d * n
    defaultapart_r = 0.7;
    defaultpath = './';
    defaultsign = 'ORL';
    p = inputParser;
    addOptional(p,'apart_r',defaultapart_r);
    addParameter(p,'path',defaultpath);
    addParameter(p,'sign',defaultsign);
    parse(p,apart_r,varargin{:});
    
    apart_parm = p.Results.apart_r;
    file_path = p.Results.path;
    package_sign = p.Results.sign;

    if strcmp(package_sign,'COIL100')
        database = load(strcat(file_path,'COIL100.mat'));
        data = database.COIL100;
        label = database.gnd;
        label = label';
        [data_d,data_n] = size(data);
        
        %% train_test data
        train_data = [];
        test_data = [];
        train_label = [];
        test_label = [];
        
        for i = 1:100
            ind_list = find(label==i);                        % 满足该类的下标（在数据中的位置）
            totl_ind = 1:72;                                  % 该类下标表的总下标  
            part_ind = randperm(72,floor(72 * apart_parm));          % 取出该类下标中的一部分作为训练集的下标表的下标
            not_part_ind = setdiff(totl_ind,part_ind);        % 下标表中作为测试集的表的下标
            
            train_data = [train_data,data(:,ind_list(part_ind))];
            train_label = [train_label,label(:,ind_list(part_ind))];
            test_data = [test_data,data(:,ind_list(not_part_ind))];
            test_label = [test_label,label(:,ind_list(not_part_ind))];
        end

        class_order = randperm(100);

        new_train_data = [];
        new_train_label = [];

        for i = 1:100
            ind_list = find(train_label == class_order(i));
            num_samples = length(ind_list);
            
            if i <= 33
                ratio = 1;
            elseif i <=66
                ratio = 2/3;
            else
                ratio = 1/3;
            end

            num_train = floor(num_samples * ratio);
            train_ind = randperm(num_samples, num_train);
            new_train_data = [new_train_data, train_data(:, ind_list(train_ind))];
            new_train_label = [new_train_label, train_label(:, ind_list(train_ind))];
        end
        train_data = new_train_data;
        train_label = new_train_label;
    end
end