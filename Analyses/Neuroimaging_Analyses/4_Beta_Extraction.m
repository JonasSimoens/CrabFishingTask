spm('defaults', 'fmri');
spm_jobman('initcfg');

cwd = 'C:/Users/.../Experiment 2/Analyses/';
cd(cwd);

subs = [1:11 13:19 21:25 27:51 53];

output_dir = [cwd sprintf('beta_values/')];

relevant_betas = [1, 8, 15, 22, 29, 36, 50, 57, 64, 71, 78, 85, 99, 106, 113, 120, 127, 134, 148, 155, 162, 169, 176, 183];

header = {'x_coord', 'y_coord', 'z_coord', 'angle_1_run_1', 'angle_2_run_1', 'angle_3_run_1', 'angle_4_run_1', 'angle_5_run_1', 'angle_6_run_1', 'angle_1_run_2', 'angle_2_run_2', 'angle_3_run_2', 'angle_4_run_2', 'angle_5_run_2', 'angle_6_run_2', 'angle_1_run_3', 'angle_2_run_3', 'angle_3_run_3', 'angle_4_run_3', 'angle_5_run_3', 'angle_6_run_3', 'angle_1_run_4', 'angle_2_run_4', 'angle_3_run_4', 'angle_4_run_4', 'angle_5_run_4', 'angle_6_run_4'};

for sub = subs
    
    input_dir = [cwd sprintf('first_level/sub-%d/', sub)];

    for b = relevant_betas
        
        if b == 1

            Y = spm_read_vols(spm_vol([input_dir sprintf('beta_%04d.nii', b)]));

            indx = find(~isnan(Y));
            [x,y,z] = ind2sub(size(Y),indx);
            data = [x y z Y(indx)];

        else

            Y = spm_read_vols(spm_vol([input_dir sprintf('beta_%04d.nii', b)]));

            data = [data Y(indx)];

        end
    end

    data_table = array2table(data, "VariableNames", header);
    writetable(data_table, [output_dir sprintf('beta_values_part_%d.csv', sub)], Delimiter= ',');

end
