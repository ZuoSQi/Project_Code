function spd_shifted = injectOffset(spd_mats, shift_val)
    T = size(spd_mats, 3);
    spd_shifted = spd_mats;
    for t = 1:T
        spd_shifted(:,:,t) = spd_mats(:,:,t) + shift_val * eye(2);
    end
end
