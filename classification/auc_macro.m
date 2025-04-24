function auc = auc_macro(y_true, scores)
classes = unique(y_true);
n_class = length(classes);
aucs = zeros(n_class, 1);
for i = 1:n_class
    label = (y_true == classes(i));
    try
        [~,~,~,auc_score] = perfcurve(label, scores(:,i), 1);
        aucs(i) = auc_score;
    catch
        aucs(i) = NaN;
    end
end
auc = mean(aucs, 'omitnan');
end
