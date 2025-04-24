function [Y_pred, acc, scores] = run_rf(X_train, Y_train, X_test, Y_test)
    model = fitcensemble(X_train, Y_train, 'Method', 'Bag');
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end