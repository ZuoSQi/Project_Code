%自适应提升树
function [Y_pred, acc, model] = run_adaboost(X_train, Y_train, X_test, Y_test)
    % 使用 AdaBoostM2 方法，适用于多分类问题
    model = fitcensemble(X_train, Y_train, 'Method', 'AdaBoostM2', 'NumLearningCycles', 100);
    Y_pred = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end
