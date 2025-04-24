%梯度提升树
function [Y_pred, acc, model] = run_gbt(X_train, Y_train, X_test, Y_test)
    % 使用 AdaBoostM2，支持多分类的梯度提升模型
    t = templateTree('MaxNumSplits', 10);  % 基学习器为浅层决策树
    model = fitcensemble(X_train, Y_train, ...
        'Method', 'AdaBoostM2', ...
        'NumLearningCycles', 100, ...
        'Learners', t);
    Y_pred = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end
