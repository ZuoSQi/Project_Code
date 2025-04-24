function err = geodesic_error_spd(P, Q)
% P, Q: 2x2xN 的 SPD 矩阵序列
N = size(P, 3);
err = zeros(N, 1);

for i = 1:N
    A = P(:,:,i);
    B = Q(:,:,i);
    sqrtA = sqrtm(A);
    invSqrtA = inv(sqrtA);
    C = invSqrtA * B * invSqrtA;
    eigvals = eig(C);
    err(i) = sqrt(sum(log(eigvals).^2));  % geodesic distance
end
end
