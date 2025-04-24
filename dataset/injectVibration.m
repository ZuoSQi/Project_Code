function spd_noisy = injectVibration(spd_mats, noise_level)
    T = size(spd_mats, 3);
    spd_noisy = spd_mats;
    for t = 1:T
        noise = noise_level * randn(2,2);
        noise = 0.5 * (noise + noise');  % 保对称
        spd_noisy(:,:,t) = spd_mats(:,:,t) + noise;
    end
end
