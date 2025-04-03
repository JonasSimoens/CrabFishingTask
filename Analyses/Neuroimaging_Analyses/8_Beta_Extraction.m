spm('defaults', 'fmri');
spm_jobman('initcfg');

cwd = 'C:/Users/.../Experiment 2/Analyses/';

output_dir = [cwd sprintf('beta_values/')];

subs = [1:11 13:19 21:25 27:51 53];

relevant_betas = [1, 8, 15, 22, 29, 36, 50, 57, 64, 71, 78, 85, 99, 106, 113, 120, 127, 134, 148, 155, 162, 169, 176, 183];

header = {'angle_1_run_1', 'angle_2_run_1', 'angle_3_run_1', 'angle_4_run_1', 'angle_5_run_1', 'angle_6_run_1', 'angle_1_run_2', 'angle_2_run_2', 'angle_3_run_2', 'angle_4_run_2', 'angle_5_run_2', 'angle_6_run_2', 'angle_1_run_3', 'angle_2_run_3', 'angle_3_run_3', 'angle_4_run_3', 'angle_5_run_3', 'angle_6_run_3', 'angle_1_run_4', 'angle_2_run_4', 'angle_3_run_4', 'angle_4_run_4', 'angle_5_run_4', 'angle_6_run_4'};

Y = spm_read_vols(spm_vol([cwd sprintf('rofc3.nii')]), 1);
indx = find(Y > 0);
[x,y,z] = ind2sub(size(Y), indx);

XYZ = [x y z]';

for sub = subs
    
    input_dir = [cwd sprintf('first_level/sub-%d/', sub)];

    data = [];

    for b = relevant_betas
        
        contrast = [input_dir sprintf('beta_%04d.nii', b)];

        ROI_data = spm_get_data(contrast, XYZ)';

        data = [data ROI_data];

    end

    data_table = array2table(data, "VariableNames", header);

    writetable(data_table, [output_dir sprintf('beta_values_part_%d.csv', sub)], Delimiter= ',');

end
