close all;
clear;
clc;

%% L20RDR parameter search demo with GA
apart_parm = 0.4;
holesize = 15;
sign = 'AR';
addpath("utils");
addpath("BenfordRNG-master");

dim = [60, 180];
alpha = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4];
beta = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4];
eta = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4];
gamma = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4];
theta = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4];
sigma_mul = [1, 2, 3, 4, 5];

para_mat = cell(length(dim), length(sigma_mul));

%% train/test data
[train_data,train_label,test_data,test_label,data_d,...
        data_n] = func_choDS(apart_parm,'sign',sign);
train_data = double(train_data);
test_data = double(test_data);

% % impulse noise
% train_data = func_impulsenoise(train_data);

% % insert image
% train_data = func_insertImage(train_data, 50, 40, 'monkey.png');

train_data = train_data - mean(train_data,2);
test_data = test_data - mean(test_data,2);
train_data = normalize(train_data,2);
test_data = normalize(test_data,2);
c = train_label(end);
train_n = length(train_label);
train_onehot = zeros(c,train_n);
for i = 1:train_n
    train_onehot(train_label(i),i) = 1;
end

% dig a hole
train_data = reshape(train_data,[50 40 train_n]);
train_data = func_digahole(train_data,holesize);
%             if necessary, figure can be applied here.
train_data = reshape(train_data,[2000,train_n]);

for i=1:length(dim)
    for j = 1: length(sigma_mul)

        opts = [];
        opts.PopulationSize = 200;

        % 生成一个100x5的矩阵，每个元素都是1到7的随机整数
        random_matrix = randi([1, 17], opts.PopulationSize, 5);
        score_list = zeros(opts.PopulationSize, 1);

        for p = 1:opts.PopulationSize
            [B_mat,A_mat,h_vec,loss,cap_num,epsilon] = func_GOALfinal(train_data,...
                            train_onehot,dim(i),alpha(random_matrix(p, 1)),...
                            beta(random_matrix(p, 2)),...
                            eta(random_matrix(p, 3)),...
                            gamma(random_matrix(p, 4)),...
                            theta(random_matrix(p, 5)),...
                            sigma_mul(j),4);

            % 1-nn classifier
            nn1_label = func_kNN(B_mat' * test_data,B_mat' * train_data,train_label,1);
            acc = func_getRecogAcc(test_label, nn1_label);

            score_list(p, 1) = acc;
        end

        [score_list, sort_ind] = sort(score_list, 'descend');
        random_matrix = random_matrix(sort_ind, :);

        for t = 1:200
%             % 生成两个不重复的随机整数
%             num1 = 0;
%             num2 = 0;
%             while num1 == num2
%                 % 生成服从指数分布的随机数
%                 randNum1 = -log(rand()) * opts.PopulationSize;
%                 randNum2 = -log(rand()) * opts.PopulationSize;
%             
%                 % 将随机数四舍五入为最接近的整数
%                 num1 = round(randNum1);
%                 num2 = round(randNum2);
%             
%                 % 确保生成的数在1到100的范围内
%                 if num1 > opts.PopulationSize
%                     num1 = opts.PopulationSize;
%                 end
%                 if num1 < 1
%                     num1 = 1;
%                 end
%                 if num2 > opts.PopulationSize
%                     num2 = opts.PopulationSize;
%                 end
%                 if num2 < 1
%                     num2 = 1;
%                 end
%             end
%             new_baby = zeros(1, 5);
%             father = random_matrix(num1, :);
%             mather = random_matrix(num2, :);

            % 生成两个不重复的随机整数
            random_numbers = randbenford(2, 200, 1, 0, 1);
            new_baby = zeros(1, 5);
            father = random_matrix(random_numbers(1), :);
            mather = random_matrix(random_numbers(2), :);

            % 对比两个向量
            for k = 1:length(father)
                if father(k) == mather(k)
                    new_baby(k) = father(k);
                else
                    flag = randi([1, 2]);
                    if flag == 1
                        new_baby(k) = father(k);
                    else
                        new_baby(k) = mather(k);
                    end
                end
            end

            [B_mat,A_mat,h_vec,loss,cap_num,epsilon] = func_GOALfinal(train_data,...
                            train_onehot,dim(i),alpha(new_baby(1, 1)),...
                            beta(new_baby(1, 2)),...
                            eta(new_baby(1, 3)),...
                            gamma(new_baby(1, 4)),...
                            theta(new_baby(1, 5)),...
                            sigma_mul(j),4);

            % 1-nn classifier
            nn1_label = func_kNN(B_mat' * test_data,B_mat' * train_data,train_label,1);
            acc = func_getRecogAcc(test_label, nn1_label);

            if acc > score_list(end)
                score_list = [score_list; acc];
                random_matrix = [random_matrix; new_baby];
            end 

            [score_list, sort_ind] = sort(score_list, 'descend');
            score_list = score_list(1:opts.PopulationSize);
            random_matrix = random_matrix(sort_ind(1:opts.PopulationSize), :);

        end 

        para_mat{i, j} = [score_list(1), random_matrix(1, :)];

    end
end
