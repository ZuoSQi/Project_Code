function [Y_pred, acc, scores] = run_ann(X_train, Y_train, X_test, Y_test)
    % 标准化
    mu = mean(X_train);
    sigma = std(X_train) + eps;
    X_train = (X_train - mu) ./ sigma;
    X_test  = (X_test - mu) ./ sigma;

    % 网络训练
    model = fitcnet(X_train, Y_train, 'LayerSizes', [10 10], ...
        'Activations', 'relu', 'Standardize', false);

    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end
