clc; clear; close all;
addpath(genpath('models'));
addpath(genpath('utils'));

% 加载数据
load('features_iRFPCA.mat');  % 包含 X_feat 和 Y
[N, D] = size(X_feat);
fprintf('共有 %d 条轨迹，每条 %d 维特征\n', N, D);

% 30 次实验
repeat_times = 30;
model_funcs = {@run_tree, @run_rf, @run_nb, @run_knn, @run_svm, ...
               @run_adaboost, @run_gbt, @run_mlp, @run_ann};
model_names = {'决策树','随机森林','朴素贝叶斯','KNN','SVM','AdaBoost','GBT','MLP','ANN'};

metrics = struct();

for i = 1:length(model_names)
    accs = zeros(repeat_times,1);
    f1s  = zeros(repeat_times,1);
    aucs = zeros(repeat_times,1);
    for k = 1:repeat_times
        cv = cvpartition(Y, 'HoldOut', 0.2);
        X_train = X_feat(cv.training, :);
        Y_train = Y(cv.training);
        X_test  = X_feat(cv.test, :);
        Y_test  = Y(cv.test);
        try
            [Y_pred, ~, scores] = model_funcs{i}(X_train, Y_train, X_test, Y_test);
            accs(k) = mean(Y_pred == Y_test);
            f1s(k) = f1_score_macro(Y_test, Y_pred);
            aucs(k) = auc_macro(Y_test, scores);
        catch
            accs(k) = NaN;
            f1s(k) = NaN;
            aucs(k) = NaN;
        end
    end
    metrics(i).name = model_names{i};
    metrics(i).acc_mean = mean(accs, 'omitnan');
    metrics(i).acc_std  = std(accs, 'omitnan');
    metrics(i).f1_mean  = mean(f1s, 'omitnan');
    metrics(i).f1_std   = std(f1s, 'omitnan');
    metrics(i).auc_mean = mean(aucs, 'omitnan');
    metrics(i).auc_std  = std(aucs, 'omitnan');
end

% 输出结果表格
T = struct2table(metrics);
disp(T(:, {'name', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std', 'auc_mean', 'auc_std'}));
