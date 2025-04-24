function [Y_pred, acc, scores] = run_adaboost(X_train, Y_train, X_test, Y_test)
    model = fitcecoc(X_train, Y_train, 'Learners', templateTree(), 'Coding', 'onevsall');
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end
