clc; clear;

% ✅ 加载 Step1 中生成的轨迹数据（XT: 208×1 cell，每条是 2×2×1000）
load('XT_Y_for_iRFPCA.mat');  % 包含 XT, Y

% ✅ Step 1：将每条轨迹从 SPD → log → 展平为向量形式，拼成 [3 × T × 208]
N = length(XT);
T = size(XT{1}, 3);  % 时间步数
X_mat = zeros(3, T, N);

for i = 1:N
    for t = 1:T
        A = XT{i}(:,:,t);
        logA = logm(A);
        X_mat(:, t, i) = [logA(1,1); logA(1,2); logA(2,2)];
    end
end

% ✅ Step 2：构造 SPD 上的几何结构体 mfd
mfd.name = 'SPD';
mfd.d = 3;   % SPD(2) 对数映射后是 3 维
mfd.D = 2;

% 函数接口
mfd.intrinsic_mean    = @(X) frechet_mean_SPD(X);        % 支持 [3×T×N]
mfd.Log               = @(mu,X) log_map_SPD(mu,X);
mfd.Exp               = @(mu,V) exp_map_SPD(mu,V);
mfd.coef_process      = @(mu,V) coef_process_SPD(mu,V);
mfd.coef_to_log       = @(mu,phi) coef_to_log_SPD(mu,phi);
mfd.orthonormal_frame = @(mu) orthonormal_frame_SPD(mu);

% ✅ Step 3：执行 iRFPCA
K = 20;  % 主成分数（可调）
rslt = iRFPCA(X_mat, mfd, 'K', K);  % 返回结构体

% ✅ Step 4：提取 Xi（主成分得分）作为最终特征
X_feat = rslt.Xi;  % 大小 [208 × 10]

% ✅ Step 5：保存特征与标签
Y = Y(:);  % 确保列向量
save('features_iRFPCA.mat', 'X_feat', 'Y');
disp('✅ iRFPCA 提取成功！208条轨迹 → 每条20维特征');
