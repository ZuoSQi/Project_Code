# Riemannian Lasa (R-LASA) Dataset
A dataset containing Riemannian (unit quaternion (UQ) and symmetric and positive definite (SPD) matrix) motion profiles.

The R-LASA dataset is obtained with the procedure described in [(Saveriano et al., 2023)](https://www.sciencedirect.com/science/article/pii/S0921889023001495) to augment an Euclidean dataset with Riemannian motions.

## Script description
- `LASA_to_R_LASA_UQ.m`: a script to augment the LASA Handwritten dataset with Riemannian (unit quaternion (UQ)) motion profiles.
- `LASA_to_R_LASA_SPD.m`: a script to augment the LASA Handwritten dataset with Riemannian (symmetirc and positive definite (SPD) matrices) motion profiles.
- `extractSPDFeature.m`: 接收一个 SPD 矩阵序列作为输入，输出每个时间点上的 logm 特征向量。
- `extractUQFeature.m`: 提取UQ特征。
- `injectVibration.m`: 注入振动异常的函数
- `injectOffset.m`: 注入贴合偏移（平移扰动）
- `uq_shuffled.m`: 速度失控（打乱时序）
- `SPD_UQ_abnormal.m`: 主处理程序（SPD + UQ 合并 + 正常轨迹 + 3 类异常）

## Software Requirements
The code is developed and tested under `Ubuntu 18.04` and `Matlab2019b`.

## References
Please acknowledge the authors in any academic publication that used this dataset and/or the provided code.
```
@article{Saveriano2023Learning,
author = {Saveriano, Matteo and Abu-Dakka, Fares J. and Kyrki, Ville},
title = {Learning stable robotic skills on Riemannian manifolds},
journal = {Robotics and Autonomous Systems},
volume = {169},
pages = {104510},
year = {2023}
}

@article{Wang2022Learning,
  author={Wang, Weitao and Saveriano, Matteo and Abu-Dakka, Fares J.},
  title={Learning Deep Robotic Skills on Riemannian Manifolds}, 
  journal={IEEE Access},
  volume={10},
  pages={114143--114152},
  year={2022}
}

```

## Third-party material
Third-party code and dataset have been included in this repository for convenience.

- *LASA Handwritten dataset*: please acknowledge the authors in any academic publications that have made use of the LASA HandWritten dataset by citing: *S. M. Khansari-Zadeh and A. Billard, "Learning Stable Non-Linear Dynamical Systems with Gaussian Mixture Models", IEEE Transaction on Robotics, 2011*.

## Note
This source code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.

## extractSPDFeature用法
% 1. 加载数据
load('CShape_SPD.mat');

% 2. 拿出一条轨迹的 SPD 数据
spd_mats = demoSPD{1}.spd;

% 3. 提取特征
features = extractSPDFeature(spd_mats);

% 4. 查看结果
disp(size(features));  % 应该是 [T × 3]

## injectVibration用法
%% 步骤 1：加载数据
load('CShape_SPD.mat');

%% 步骤 2：拿出一条轨迹
spd_mats = demoSPD{1}.spd;  % 原始 SPD，2x2xT

%% 步骤 3：调用注入函数
noise_level = 0.01;  % 可调范围建议 0.005 ~ 0.05
spd_vibration = injectVibration(spd_mats, noise_level);  % 得到异常轨迹

%% 步骤 4：查看结果
disp(size(spd_vibration));  % 应该还是 2x2xT

% （可选）画个矩阵看看变化
disp('原始 SPD 第1帧:');
disp(spd_mats(:,:,1));

disp('注入振动后的 SPD 第1帧:');
disp(spd_vibration(:,:,1));

## 保存数据
save('full_trajectory_dataset.mat', 'X_all', 'y_all');
disp('✅ 已保存为 full_trajectory_dataset.mat');
