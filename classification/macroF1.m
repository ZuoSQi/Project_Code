function score = macroF1(y_true, y_pred)
    classes = unique(y_true);
    f1s = zeros(length(classes), 1);
    for i = 1:length(classes)
        c = classes(i);
        tp = sum((y_pred == c) & (y_true == c));
        fp = sum((y_pred == c) & (y_true ~= c));
        fn = sum((y_pred ~= c) & (y_true == c));
        if tp + fp == 0 || tp + fn == 0
            f1s(i) = 0;
        else
            prec = tp / (tp + fp);
            rec  = tp / (tp + fn);
            f1s(i) = 2 * prec * rec / (prec + rec);
        end
    end
    score = mean(f1s, 'omitnan');
end
