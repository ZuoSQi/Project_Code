function V = log_map_SPD(mu, X)
% mu: [3 × T]，均值路径
% X : [3 × T × N]，样本数据
% 返回：V [3 × T × N]

[~, T, N] = size(X);
V = zeros(3, T, N);
for t = 1:T
    for i = 1:N
        V(:, t, i) = X(:, t, i) - mu(:, t);
    end
end