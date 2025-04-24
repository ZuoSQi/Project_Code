%K近邻
function [Y_pred, acc, model] = run_knn(X_train, Y_train, X_test, Y_test)
    model = fitcknn(X_train, Y_train, 'NumNeighbors', 5);
    Y_pred = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end
