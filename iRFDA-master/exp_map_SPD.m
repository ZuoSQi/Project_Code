function X = exp_map_SPD(mu, V)
% mu: [3 × T]，均值路径
% V : [3 × T × N]，切空间向量
% 返回：X [3 × T × N]

[~, T, N] = size(V);
X = zeros(3, T, N);
for t = 1:T
    for i = 1:N
        X(:, t, i) = mu(:, t) + V(:, t, i);
    end
end