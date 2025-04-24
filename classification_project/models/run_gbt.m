function [Y_pred, acc, scores] = run_gbt(X_train, Y_train, X_test, Y_test)
    t = templateTree('MaxNumSplits', 10);
    model = fitcecoc(X_train, Y_train, 'Learners', t, 'Coding', 'onevsall');
    [Y_pred, scores] = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end
