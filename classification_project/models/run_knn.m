function [Y_pred, acc, scores] = run_knn(X_train, Y_train, X_test, Y_test)
    model = fitcknn(X_train, Y_train, 'NumNeighbors', 5, 'Standardize', 1);
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end