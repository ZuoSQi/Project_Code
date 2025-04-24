function mu = frechet_mean_SPD(X)
% 输入：X 为 [3 × T × N]
% 输出：mu 为 [3 × T]，表示平均轨迹

[~, T, N] = size(X);
mu = zeros(3, T);
for t = 1:T
    mu(:,t) = mean(X(:,t,:), 3);
end