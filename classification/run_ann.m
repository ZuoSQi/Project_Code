%单隐层神经网络
function [Y_pred, acc] = run_ann(X_train, Y_train, X_test, Y_test)
    % One-hot 编码：保证标签从 1 开始
    % 对标签进行转换
    Y_train_oh = full(ind2vec(Y_train'+1));
    % 采用单隐藏层，隐含层节点数为50
    net = patternnet(50);
    % 可调整正则化参数以控制过拟合
    net.performParam.regularization = 0.05;
    net.trainParam.showWindow = false;
    net = train(net, X_train', Y_train_oh);
    Y_pred_probs = net(X_test');
    [~, Y_pred] = max(Y_pred_probs);
    Y_pred = Y_pred' - 1;
    acc = mean(Y_pred == Y_test);
end
