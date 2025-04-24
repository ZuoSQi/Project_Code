%多层神经网络
function [Y_pred, acc] = run_mlp(X_train, Y_train, X_test, Y_test)
    Y_train_oh = full(ind2vec(Y_train'+1));
    net = patternnet([64, 32, 16]);
    net.performParam.regularization = 0.1;
    net.trainParam.showWindow = false;
    net = train(net, X_train', Y_train_oh);
    Y_pred_probs = net(X_test');
    [~, Y_pred] = max(Y_pred_probs);
    Y_pred = Y_pred' - 1;
    acc = mean(Y_pred == Y_test);
end