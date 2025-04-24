function logvecs = spd_log_map(spd_seq)
% 输入：spd_seq 是 2x2xN 的 SPD 矩阵序列
% 输出：logvecs 是 Nx3 的切空间向量序列

N = size(spd_seq, 3);           % 帧数
logvecs = zeros(N, 3);          % 初始化

for i = 1:N
    spd = spd_seq(:, :, i);     % 第i帧 SPD
    logm_spd = logm(spd);       % 对数映射
    logvecs(i, :) = [logm_spd(1,1), logm_spd(1,2), logm_spd(2,2)];
end
end
