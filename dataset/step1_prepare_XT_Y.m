%% Step 1: 构造 XT（轨迹样本）和 Y（标签），正常轨迹加入轻微扰动
clc; clear;

addpath(genpath('./dataset'));

load('CShape_SPD.mat');  % 包含 demoSPD{1~4}

XT = {};   % 存储 2×2×T 的轨迹样本
Y = [];    % 存储对应标签

samples_per_class = 13;  % 每类轨迹在每条原始轨迹中复制13条 → 208 条样本

for i = 1:4
    spd = demoSPD{i}.spd;

    for j = 1:samples_per_class
        % 类别 0：正常（加入非常轻微扰动）
        spd_normal = injectVibration(spd, 0.001);  % 模拟正常波动
        XT{end+1} = spd_normal;
        Y(end+1) = 0;

        % 类别 1：贴合偏移
        spd_offset = injectOffset(spd, 0.01);
        XT{end+1} = spd_offset;
        Y(end+1) = 1;

        % 类别 2：振动异常
        spd_vib = injectVibration(spd, 0.01);
        XT{end+1} = spd_vib;
        Y(end+1) = 2;

        % 类别 3：速度失控（打乱时间顺序）
        spd_shuffle = spd(:, :, randperm(size(spd, 3)));
        XT{end+1} = spd_shuffle;
        Y(end+1) = 3;
    end
end

disp(['XT样本数: ', num2str(length(XT))]);  % 应为 208
disp(['标签Y样本数: ', num2str(length(Y))]);

save('XT_Y_for_iRFPCA.mat', 'XT', 'Y');
disp('✅ XT + Y（含轻扰动正常轨迹）已保存');
