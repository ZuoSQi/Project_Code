function uq_shuffled = injectSpeedLoss_UQ(uq_seq)
    % uq_seq: 4×T
    idx = randperm(size(uq_seq, 2));
    uq_shuffled = uq_seq(:, idx);
end
