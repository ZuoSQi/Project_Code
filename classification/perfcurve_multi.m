function [mean_auc, aucs] = perfcurve_multi(Y_true, scores, class_labels)
% 输入:
%   Y_true: 测试集真实标签
%   scores: 预测概率，大小为 N x C（每列为每类的概率）
%   class_labels: 类别标签数组，如 0:3

aucs = zeros(length(class_labels), 1);
for i = 1:length(class_labels)
    c = class_labels(i);
    y_true_bin = (Y_true == c);
    [~, ~, ~, auc] = perfcurve(y_true_bin, scores(:,i), true);
    aucs(i) = auc;
end
mean_auc = mean(aucs, 'omitnan');
end
