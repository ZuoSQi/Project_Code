%随机森林
function [Y_pred, acc, model] = run_rf(X_train, Y_train, X_test, Y_test)
    model = TreeBagger(100, X_train, Y_train, ...
        'Method','classification', 'OOBPrediction','on', ...
        'OOBPredictorImportance','on');
    Y_pred = str2double(predict(model, X_test));
    acc = mean(Y_pred == Y_test);
end