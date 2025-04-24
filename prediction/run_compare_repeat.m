function run_compare_repeat()
repeat = 10;
extrap_errors = zeros(repeat,1);
interp_errors = zeros(repeat,1);

for i = 1:repeat
    extrap_errors(i) = run_extrap_once();
    interp_errors(i) = run_interp_once();
    fprintf('[%d] extrap: %.10f, interp: %.10f\n', i, extrap_errors(i), interp_errors(i));
end

fprintf('\n【统计结果 - 测地线 MSE】\n');
fprintf('外拓预测：均值 = %.10f, 标准差 = %.10f\n', mean(extrap_errors), std(extrap_errors));
fprintf('挖点补全：均值 = %.10f, 标准差 = %.10f\n', mean(interp_errors), std(interp_errors));

figure;
boxplot([extrap_errors, interp_errors], 'Labels', {'外拓预测', '挖点补全'});
ylabel('Geodesic MSE'); title('WGPR 多次运行测地线误差箱线图');
grid on;
end
