function [Y_pred, acc, scores] = run_svm(X_train, Y_train, X_test, Y_test)
    % 手动标准化输入
    mu = mean(X_train); sigma = std(X_train) + eps;
    X_train = (X_train - mu) ./ sigma;
    X_test = (X_test - mu) ./ sigma;

    % 创建 SVM 模型（线性核，禁用自动标准化）
    t = templateSVM('KernelFunction','linear','Standardize',false);

    % 多分类 + 拟合后验概率
    model = fitcecoc(X_train, Y_train, 'Learners', t, 'Coding','onevsall');
    model = fitPosterior(model, X_train, Y_train);  % 避免 NaN scores

    % 预测
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end