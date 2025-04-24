function frame = orthonormal_frame_SPD(mu)
% 返回单位基向量张量，大小 [3 × T × 3]
[~, T] = size(mu);
frame = zeros(3, T, 3);
for k = 1:3
    ek = zeros(3, 1); ek(k) = 1;
    frame(:,:,k) = repmat(ek, 1, T);
end