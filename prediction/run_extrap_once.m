function geo_mse = run_extrap_once()
load('CShape_SPD.mat');
data = demoSPD{1};
spd_seq = data.spd;

logvecs = spd_log_map(spd_seq);
t = (0:size(logvecs,1)-1)';
y1 = logvecs(:,1); y2 = logvecs(:,2); y3 = logvecs(:,3);

% 固定外拓：训练1~970, 测试971~1000
train_idx = 1:970; test_idx = 971:1000;
t_train = t(train_idx); t_test = t(test_idx);
y1_train = y1(train_idx); y2_train = y2(train_idx); y3_train = y3(train_idx);
y1_test = y1(test_idx); y2_test = y2(test_idx); y3_test = y3(test_idx);

% 训练 GP
gpr1 = fitrgp(t_train, y1_train, 'Basis','pureQuadratic','KernelFunction','matern52','Standardize',true);
gpr2 = fitrgp(t_train, y2_train, 'Basis','pureQuadratic','KernelFunction','matern52','Standardize',true);
gpr3 = fitrgp(t_train, y3_train, 'Basis','pureQuadratic','KernelFunction','matern52','Standardize',true);

% 预测
[y1_pred,~] = predict(gpr1, t_test);
[y2_pred,~] = predict(gpr2, t_test);
[y3_pred,~] = predict(gpr3, t_test);

log_pred = [y1_pred, y2_pred, y3_pred];
pred_spd = zeros(2,2,length(t_test));
for i = 1:length(t_test)
    L = [log_pred(i,1), log_pred(i,2); log_pred(i,2), log_pred(i,3)];
    pred_spd(:,:,i) = expm(L);
end

true_spd = spd_seq(:,:,test_idx);
geo_errs = geodesic_error_spd(true_spd, pred_spd);
geo_mse = mean(geo_errs.^2);
end
