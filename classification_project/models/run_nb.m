function [Y_pred, acc, scores] = run_nb(X_train, Y_train, X_test, Y_test)
    model = fitcnb(X_train, Y_train);
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end