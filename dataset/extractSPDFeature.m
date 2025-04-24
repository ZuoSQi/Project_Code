function features = extractSPDFeature(spd_mats)
    T = size(spd_mats, 3);
    features = zeros(T, 3);  % 2x2 SPD矩阵 → 3维特征
    for t = 1:T
        mat = spd_mats(:, :, t);
        log_mat = logm(mat);
        features(t, :) = [log_mat(1,1), log_mat(1,2), log_mat(2,2)];
    end
end
