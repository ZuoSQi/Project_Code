function uq_feat = extractUQFeature(uq_seq)
    % uq_seq: 4×T 单位四元数序列
    % 返回：T × 4，每列是 [q0 q1 q2 q3]
    uq_feat = uq_seq';  % 直接转置为 T×4
end
