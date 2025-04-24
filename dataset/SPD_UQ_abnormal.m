X_all = {};
y_all = [];

% 每类13条 × 4条原始轨迹 ≈ 52 条 → 总共约 200 条
samples_per_type = 13;

for i = 1:4  % 遍历 demoSPD{1}~{4}
    spd = demoSPD{i}.spd;
    uq = demoUQ{i}.quat;  % 4×T 单位四元数

    for j = 1:samples_per_type
        % 类别0：正常轨迹
        feat_spd = extractSPDFeature(spd);
        feat_uq = extractUQFeature(uq);
        X_all{end+1} = [feat_spd, feat_uq];
        y_all(end+1) = 0;
    end

    for j = 1:samples_per_type
        % 类别1：贴合偏移
        spd_offset = injectOffset(spd, 0.01);
        feat_spd = extractSPDFeature(spd_offset);
        feat_uq = extractUQFeature(uq);
        X_all{end+1} = [feat_spd, feat_uq];
        y_all(end+1) = 1;
    end

    for j = 1:samples_per_type
        % 类别2：振动异常
        spd_vib = injectVibration(spd, 0.01);
        feat_spd = extractSPDFeature(spd_vib);
        feat_uq = extractUQFeature(uq);
        X_all{end+1} = [feat_spd, feat_uq];
        y_all(end+1) = 2;
    end

    for j = 1:samples_per_type
        % 类别3：速度失控
        uq_shuffle = injectSpeedLoss_UQ(uq);
        feat_spd = extractSPDFeature(spd);  % SPD 不打乱
        feat_uq = extractUQFeature(uq_shuffle);
        X_all{end+1} = [feat_spd, feat_uq];
        y_all(end+1) = 3;
    end
end

disp(['✅ 样本生成完毕，轨迹总数：', num2str(length(X_all))]);
