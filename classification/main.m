clc; clear; close all;
rng(2025);  % 固定随机种子

%% 1. 加载特征
load('features_iRFPCA.mat');  % 包含 X_feat, Y
[N, D] = size(X_feat);
fprintf('共有 %d 条轨迹，每条 %d 维特征\n', N, D);

%% 2. 数据可视化
fprintf('▶ 特征均值 + 标准差：\n');
disp(array2table([mean(X_feat); std(X_feat)], ...
    'VariableNames', strcat('F', string(1:D)), ...
    'RowNames', {'均值','标准差'}));

figure;
imagesc(corr(X_feat)); colorbar;
title('特征相关性热力图'); xlabel('特征'); ylabel('特征');

figure;
histogram(categorical(Y, 0:3, {'正常','偏移','振动','失控'}));
title('标签分布'); xlabel('类别'); ylabel('样本数');

%% 3. 划分训练 / 测试集
cv = cvpartition(Y, 'HoldOut', 0.2);
X_train = X_feat(cv.training, :);
Y_train = Y(cv.training);
X_test  = X_feat(cv.test, :);
Y_test  = Y(cv.test);
fprintf('\n训练集：%d 条，测试集：%d 条\n', sum(cv.training), sum(cv.test));

%% 4+. 多次评估：准确率 / F1 / AUC 多指标对比
repeat_times = 10;
metrics = struct();  % 保存所有模型的多次指标
model_funcs = {@run_tree, @run_rf, @run_nb, @run_knn, @run_svm, ...
               @run_adaboost, @run_gbt, @run_mlp, @run_ann};
model_names = {'决策树','随机森林','朴素贝叶斯','KNN','SVM', ...
               'AdaBoost','GBT','MLP','ANN'};

for i = 1:length(model_names)
    accs = zeros(repeat_times,1);
    f1s  = zeros(repeat_times,1);
    aucs = zeros(repeat_times,1);
    for k = 1:repeat_times
        % 划分数据
        cv = cvpartition(Y, 'HoldOut', 0.2);
        X_train = X_feat(cv.training, :);
        Y_train = Y(cv.training);
        X_test  = X_feat(cv.test, :);
        Y_test  = Y(cv.test);
        
        % 运行模型
        try
            [Y_pred, ~, scores] = model_funcs{i}(X_train, Y_train, X_test, Y_test);
            accs(k) = mean(Y_pred == Y_test);
            
            % F1 score（宏平均）
            f1s(k) = f1_score_macro(Y_test, Y_pred);
            
            % AUC（宏平均）
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
% 显示结果表格
T = struct2table(metrics);
disp(T(:, {'name', 'f1_mean', 'f1_std', 'auc_mean', 'auc_std'}));


%% 5. 可视化 - 准确率柱状图（含颜色区分和数值标签）
accs = [results.acc];
colors = lines(numel(model_names));  % 获取互异颜色
figure;
hold on;
for i = 1:numel(model_names)
    bar(i, accs(i)*100, 'FaceColor', colors(i,:), 'EdgeColor', 'k');
    text(i, accs(i)*100 + 1, sprintf('%.1f%%', accs(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end
hold off;
xlim([0.5, numel(model_names)+0.5]);
ylim([0, 105]);
set(gca, 'XTickLabel', model_names, 'XTick', 1:numel(model_names));
xtickangle(45); ylabel('准确率 (%)'); title('分类模型准确率对比');
grid on;
%% 6. 可视化 - 混淆矩阵（最佳模型）
[~, best_idx] = max(accs);
best_pred = results(best_idx).Y_pred;
figure;
cm = confusionchart(Y_test, best_pred, ...
    'Title', ['混淆矩阵（最佳模型：', results(best_idx).name, '）'], ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
cm.FontName = '微软雅黑'; % 设置中文字体避免乱码
%% 7. 可视化 - 特征降维后可视化（t-SNE）
mappedX = tsne(X_feat, 'NumDimensions', 2, 'Perplexity', 30, 'Standardize', true);
figure;
gscatter(mappedX(:,1), mappedX(:,2), Y, lines(4), 'osd^', 8);
title('iRFPCA 特征的 t-SNE 可视化');
xlabel('降维维度1'); ylabel('降维维度2');
legend({'正常','偏移','振动','失控'});

%% 8. 特征重要性排名图（以随机森林为例）
% 重训练随机森林模型（获取变量重要性）
rf_model = fitcensemble(X_train, Y_train, 'Method', 'Bag');

% 获取重要性分数
imp = predictorImportance(rf_model);  % 1×20 数组

% 可视化柱状图
figure;
bar(imp, 'FaceColor', [0.2 0.6 0.8]);
xlabel('特征编号'); ylabel('重要性分数');
title('iRFPCA 特征维度的重要性排名');
xticks(1:length(imp)); xticklabels(strcat('F', string(1:length(imp))));
xtickangle(45); grid on;

%% 9. 特征选择并重训随机森林
K = 10;
[~, idx] = maxk(imp, K);
X_sel = X_feat(:, sort(idx));
X_train_sel = X_sel(cv.training, :);
X_test_sel = X_sel(cv.test, :);

rf_model_sel = fitcensemble(X_train_sel, Y_train, 'Method', 'Bag');
Y_pred_sel = predict(rf_model_sel, X_test_sel);
acc_sel = mean(Y_pred_sel == Y_test);
fprintf('\n精简特征后的随机森林准确率：%.2f%%\n', acc_sel * 100);

%% 10. 可视化②：精简随机森林结果 + t-SNE
figure;
cm2 = confusionchart(Y_test, Y_pred_sel, ...
    'Title', '精简特征后的混淆矩阵（随机森林）', ...
    'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
cm2.FontName = '微软雅黑';

mappedX_sel = tsne(X_sel, 'NumDimensions', 2, 'Perplexity', 30, 'Standardize', true);
figure;
gscatter(mappedX_sel(:,1), mappedX_sel(:,2), Y, lines(4), 'osd^', 8);
title('精简特征下的 t-SNE 可视化');
xlabel('降维维度1'); ylabel('降维维度2');
legend({'正常','偏移','振动','失控'});
