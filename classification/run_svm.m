%支持向量机（多分类 ECOC）
function [Y_pred, acc] = run_svm(X_train, Y_train, X_test, Y_test)
    t = templateSVM('KernelFunction','rbf');
    model = fitcecoc(X_train, Y_train, 'Learners', t);
    Y_pred = predict(model, X_test);
    acc = mean(Y_pred == Y_test);
end