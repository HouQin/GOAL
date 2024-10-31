close all;
clear;
clc;

%% GOAL parameter search demo with beyas
import statistics.*

apart_parm = 0.3;
holesize = 15;
sign = 'COIL100';
addpath("utils");

%% train/test data
% [train_data,train_label,test_data,test_label,data_d,...
%         data_n] = func_choDS(apart_parm,'sign',sign);
[train_data,train_label,test_data,test_label,data_d,...
        data_n] = func_choDS_imbalance(apart_parm,'sign',sign);
train_data = double(train_data);
test_data = double(test_data);
% % impulse noise
% train_data = func_impulsenoise(train_data);

% % insert image
% train_data = func_insertImage(train_data, 32, 32, 'monkey.png');

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
train_data = reshape(train_data,[32 32 train_n]);
train_data = func_digahole(train_data,holesize);
%             if necessary, figure can be applied here.
train_data = reshape(train_data,[1024,train_n]);

%% baseline
rr_alpha = 1e3;
rr_mat = func_RR(train_data,train_onehot,rr_alpha);

% svm classifier
svm_label = func_clasfierSVM(rr_mat' * test_data,...
    rr_mat' * train_data,train_label',[],1);
acc = func_getRecogAcc(test_label, svm_label);
  
fprintf('[Baseline Acc %.4f]\n', acc);

%% bayes loc

dim = optimizableVariable('dim', [60, 200],'Type','integer');
alpha = optimizableVariable('alpha', [1, 1e8]);
beta = optimizableVariable('beta', [1, 1e8]);
eta = optimizableVariable('eta', [1, 1e8]);
gamma = optimizableVariable('gamma', [1, 1e8]);
theta = optimizableVariable('theta', [1, 1e8]);
sigma_mul = optimizableVariable('sigma_mul', [1, 5],'Type','integer');

vars = [dim, alpha, beta, eta, gamma, theta, sigma_mul];

fun = @(x) objective(test_data, test_label, train_label, train_data, train_onehot, ...
                        x.dim, x.alpha/1e-4, x.beta/1e-4, x.eta/1e-4, x.gamma/1e-4, x.theta/1e-4, x.sigma_mul);

results = bayesopt(fun,vars,'MaxObjectiveEvaluations',500,'Verbose',1,'PlotFcn',[]);

function loss = objective(test_data,test_label,train_label,train_data,...
    train_onehot,dim,alpha,beta,eta,gamma,theta,sigma_mul)

    [B_mat,~,~,~,~,~] = func_GOALfinal(train_data,...
                            train_onehot,dim,alpha,beta,eta,gamma,theta,sigma_mul,4);

%     % 1-nn classifier
%     nn1_label = func_kNN(B_mat' * test_data,B_mat' * train_data,train_label,1);
%     acc = func_getRecogAcc(test_label, nn1_label);
    % svm classifier
    svm_label = func_clasfierSVM(B_mat' * test_data,...
        B_mat' * train_data,train_label',[],1);
    acc = func_getRecogAcc(test_label, svm_label);

    loss = 100 - acc;
end
