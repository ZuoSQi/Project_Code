% 清理工作区并关闭所有图形窗口
clear;
close all;

% 添加库路径，确保可以访问自定义函数和工具
addpath(genpath('libs'))

% 定义 LASA 数据集的路径
modelPath = 'LASA_dataset/';

%% 预处理演示数据
% 定义目标四元数为 [1, 0, 0, 0]，表示没有旋转
q_goal.s = 1;  % 四元数的实部
q_goal.v = [0; 0; 0];  % 四元数的虚部

% 设置平均采样时间
dt = 0.003;

% 设置缩放因子，将数据从像素单位转换为米
scale_ = 100;

% 遍历 LASA 数据集中的 30 个模型
for i = 1:30
    % 加载演示数据
    [demos, ~, name] = load_LASA_models(modelPath, i);
    
    % 将所有演示数据堆叠成一个矩阵
    demoMat = [];
    rangeUQ = [];
    for demoIt = 1:length(demos)
        demoMat = [demoMat; demos{demoIt}.pos];
    end
    
    % 初始化存储单位四元数演示数据的变量
    demoUQ = [];
    
    % 定义索引矩阵，用于从演示数据中提取特定的轨迹
    idx_ = [1:3; 5:7; 9:11; [13,14,1]];
    
    % 遍历索引矩阵，生成单位四元数轨迹
    for demoUQIt = 1:size(idx_, 1)
        % 从演示数据中提取特定的轨迹，生成 3D 运动
        demoUQ{demoUQIt}.tsPos = demoMat(idx_(demoUQIt, :), :);
        
        % 将轨迹从像素单位转换为米，以便获得合理的轨迹向量
        demoUQ{demoUQIt}.tsPos = demoUQ{demoUQIt}.tsPos ./ scale_;
        
        % 计算轨迹的速度
        demoUQ{demoUQIt}.tsVel = [diff(demoUQ{demoUQIt}.tsPos, [], 2) ./ dt, zeros(3, 1)];
        
        % 存储采样时间
        demoUQ{demoUQIt}.dt = dt;
        
        % 计算单位四元数轨迹
        for tt = 1:size(demoUQ{demoUQIt}.tsPos, 2)
            % 将轨迹点转换为四元数
            tmp = quat_exp(demoUQ{demoUQIt}.tsPos(:, tt));
            
            % 将生成的四元数与目标四元数相乘，得到最终的单位四元数
            demoUQ{demoUQIt}.quat(:, tt) = quat2array(quat_mult(tmp, q_goal));
        end
    end
    
    % 将生成的单位四元数轨迹保存到文件中
    filename = ['R_LASA_UQ/' name '_UQ.mat'];
    save(filename, 'demoUQ')
end