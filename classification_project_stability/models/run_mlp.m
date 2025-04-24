function [Y_pred, acc, scores] = run_mlp(X_train, Y_train, X_test, Y_test)
    model = fitcnet(X_train, Y_train, 'LayerSizes', 20);
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end