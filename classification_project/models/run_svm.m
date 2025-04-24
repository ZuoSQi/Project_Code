function [Y_pred, acc, scores] = run_svm(X_train, Y_train, X_test, Y_test)
    % ✅ 1. 手动标准化
    mu = mean(X_train);
    sigma = std(X_train) + eps;
    X_train = (X_train - mu) ./ sigma;
    X_test = (X_test - mu) ./ sigma;

    % ✅ 2. 使用 SVM 模板（RBF 核或线性核都可尝试）
    t = templateSVM('KernelFunction', 'rbf', 'Standardize', false);

    % ✅ 3. fitcecoc 加上 'Prior' 设置
    model = fitcecoc(X_train, Y_train, ...
        'Learners', t, ...
        'Coding', 'onevsall', ...
        'Prior', 'uniform');

    % ✅ 4. 明确进行概率后验拟合
    model = fitPosterior(model, X_train, Y_train);

    % ✅ 5. 预测
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end
