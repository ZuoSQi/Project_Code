% 数据读取 + SPD 对数映射 + GP 模型训练 + 外推预测 + 可视化 + MSE评估

%% Step 1: 加载 SPD 数据并做对数映射
load('CShape_SPD.mat');
data = demoSPD{1};
spd_seq = data.spd;  % 2x2x1000

% 对数映射函数（需单独保存为 spd_log_map.m）
logvecs = spd_log_map(spd_seq);  % 得到 1000x3 切向量

%% Step 2: 拆分时间序列
t = (0:size(logvecs,1)-1)';
y1 = logvecs(:,1); y2 = logvecs(:,2); y3 = logvecs(:,3);

%% Step 3: 拆分训练和测试集（优化为仅预测最后30帧）
train_idx = 1:970; test_idx = 971:1000;
t_train = t(train_idx); t_test = t(test_idx);
y1_train = y1(train_idx); y1_test = y1(test_idx);
y2_train = y2(train_idx); y2_test = y2(test_idx);
y3_train = y3(train_idx); y3_test = y3(test_idx);

%% Step 4: 使用优化核函数结构训练 GP 模型（matern52 + 二次基）
gprMdl1 = fitrgp(t_train, y1_train, 'Basis','pureQuadratic', 'KernelFunction','matern52', 'Standardize', true);
gprMdl2 = fitrgp(t_train, y2_train, 'Basis','pureQuadratic', 'KernelFunction','matern52', 'Standardize', true);
gprMdl3 = fitrgp(t_train, y3_train, 'Basis','pureQuadratic', 'KernelFunction','matern52', 'Standardize', true);

%% Step 5: 预测 + 置信区间输出
[y1_pred, y1_std] = predict(gprMdl1, t_test);
[y2_pred, y2_std] = predict(gprMdl2, t_test);
[y3_pred, y3_std] = predict(gprMdl3, t_test);

%% Step 6: 拼接向量并指数映射还原 SPD
log_pred = [y1_pred, y2_pred, y3_pred];
pred_spd = zeros(2,2,length(t_test));
for i = 1:length(t_test)
    L = [log_pred(i,1), log_pred(i,2); log_pred(i,2), log_pred(i,3)];
    pred_spd(:,:,i) = expm(L);
end

%% Step 7: 可视化预测效果（含置信区间）
figure;
subplot(3,1,1);
plot(t, y1, 'b'); hold on;
plot(t_test, y1_pred, 'r--');
plot(t_test, y1_pred + 2*y1_std, 'g--');
plot(t_test, y1_pred - 2*y1_std, 'g--');
title('logSPD(1,1)');

subplot(3,1,2);
plot(t, y2, 'b'); hold on;
plot(t_test, y2_pred, 'r--');
plot(t_test, y2_pred + 2*y2_std, 'g--');
plot(t_test, y2_pred - 2*y2_std, 'g--');
title('logSPD(1,2)');

subplot(3,1,3);
plot(t, y3, 'b'); hold on;
plot(t_test, y3_pred, 'r--');
plot(t_test, y3_pred + 2*y3_std, 'g--');
plot(t_test, y3_pred - 2*y3_std, 'g--');
title('logSPD(2,2)');
legend('真实值','GP预测','置信上界','置信下界');

%% Step 8: 评估 SPD Frobenius 误差
true_spd = spd_seq(:,:,test_idx);
mse = 0;
for i = 1:length(t_test)
    diff = pred_spd(:,:,i) - true_spd(:,:,i);
    mse = mse + norm(diff, 'fro')^2;
end
mse = mse / length(t_test);
fprintf('SPD Frobenius MSE (后 30 帧): %.6f\n', mse);

true_spd = spd_seq(:,:,test_idx);
geo_errs = geodesic_error_spd(true_spd, pred_spd);
geo_mse = mean(geo_errs.^2);
if geo_mse < 1e-10
    fprintf('测地线 MSE（挖点补全）: <1e-10（近似零）\n');
else
    fprintf('测地线 MSE（挖点补全）: %.10f\n', geo_mse);
end

