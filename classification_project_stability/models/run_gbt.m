function [Y_pred, acc, scores] = run_gbt(X_train, Y_train, X_test, Y_test)
    model = fitcensemble(X_train, Y_train, 'Method', 'LogitBoost');
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end