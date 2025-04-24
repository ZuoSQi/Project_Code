function f1 = f1_score_macro(y_true, y_pred)
classes = unique(y_true);
f1s = zeros(length(classes), 1);
for i = 1:length(classes)
    c = classes(i);
    tp = sum((y_true == c) & (y_pred == c));
    fp = sum((y_true ~= c) & (y_pred == c));
    fn = sum((y_true == c) & (y_pred ~= c));
    p = tp / (tp + fp + eps);
    r = tp / (tp + fn + eps);
    f1s(i) = 2 * p * r / (p + r + eps);
end
f1 = mean(f1s);
end