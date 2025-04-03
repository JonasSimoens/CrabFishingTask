spm('defaults', 'fmri');
spm_jobman('initcfg');

cwd = 'C:/Users/.../Experiment_2/Analyses/first_level/';
cd(cwd);

subs = [1:11 13:19 21:25 27:51 53];
nsubs = length(subs);
nruns = 4;
jobs = cell(nsubs, 1);

data_all = readtable('C:/Users/.../Experiment 2/Analyses/Event_File.csv');

conditions = table();
row = 1;
index = 1;

for sub = subs

    sub_index = (data_all.part==sub);
    data_sub = data_all(sub_index, :);

    clear matlabbatch;

    output_dir = [cwd sprintf('sub-%d/', sub)];

    matlabbatch{1}.spm.stats.fmri_spec.dir = {output_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 1.78;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

    for run = 1:nruns

        input_dir = sprintf('C:/Users/.../Experiment 2/Data/sub-%d/', sub);
        input = sprintf('smoothed_sub-%d_task-ep2dboldrun%d_space-MNI152NLin2009cAsym_desc-preproc_bold.nii', sub, run);
        input = cellstr(spm_select('ExtFPList', input_dir, input));
        input = input(6:end);

        matlabbatch{1}.spm.stats.fmri_spec.sess(run).scans = input;

        run_index = (data_sub.run==run);
        data_run = data_sub(run_index, :);

        glm_con = 1;

        for angle = 1:3

            angle_index = (data_run.angle==angle);
            data_angle = data_run(angle_index, :);

            if isempty(data_angle)

                disp('no such condition');
                continue

            end

            for event = 1:7

                event_index = (data_angle.event==event);
                data_event = data_angle(event_index, :);

                if isempty(data_event)

                    disp('no such condition');
                    continue

                end

                onset = data_event(:, 'onset');
                onset = table2array(onset);

                matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).name = [int2str(angle) '-' int2str(event)];
                matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).onset = onset;
                matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).duration = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).tmod = 0;

                if ismember(event, [3, 5])

                    error = data_event(:, 'error');
                    error = table2array(error);

                    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).pmod(1).name = ['error'];
                    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).pmod(1).param = error;
                    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).pmod(1).poly = 1;
                    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).orth = 0;

                    conditions(row, 'part') = {sub};
                    conditions(row, 'run') = {run};
                    conditions(row, 'con') = {glm_con};
                    conditions(row, 'angle') = {angle};
                    conditions(row, 'event') = {event};
                    conditions(row, 'error') = {0};
    
                    row = row + 1;

                    conditions(row, 'part') = {sub};
                    conditions(row, 'run') = {run};
                    conditions(row, 'con') = {glm_con};
                    conditions(row, 'angle') = {angle};
                    conditions(row, 'event') = {event};
                    conditions(row, 'error') = {1};
    
                    row = row + 1;
                    glm_con = glm_con + 1;

                else

                    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).pmod = struct('name', {}, 'param', {}, 'poly', {});
                    matlabbatch{1}.spm.stats.fmri_spec.sess(run).cond(glm_con).orth = 0;

                    conditions(row, 'part') = {sub};
                    conditions(row, 'run') = {run};
                    conditions(row, 'con') = {glm_con};
                    conditions(row, 'angle') = {angle};
                    conditions(row, 'event') = {event};
                    conditions(row, 'error') = {0};
    
                    row = row + 1;
                    glm_con = glm_con + 1;

                end
            end
        end
        
        nuisance_file = [input_dir sprintf('sub-%d_task-ep2dboldrun%d_desc-confounds.txt', sub, run)];
        nuisance_reg = cellstr(fullfile(nuisance_file));

        matlabbatch{1}.spm.stats.fmri_spec.sess(run).multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(run).regress = struct('name', {}, 'val', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(run).multi_reg = cellstr(nuisance_reg);
        matlabbatch{1}.spm.stats.fmri_spec.sess(run).hpf = 128;

    end

    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    jobs{index, 1} = matlabbatch;
    index = index + 1;

end

writetable(conditions, [cwd 'conditions.csv'], 'Delimiter', ',');

for i = 1:nsubs

    spm_jobman('run', jobs{i})

end

jobs = cell(nsubs, 1);

index = 1;

for sub = subs

    clear matlabbatch

    dir = [cwd sprintf('sub-%d/', sub)];
    file = cellstr(spm_select('FPList', dir, 'SPM*'));

    matlabbatch{1}.spm.stats.fmri_est.spmmat = file;
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;

    jobs{index, 1} = matlabbatch;

    index = index + 1;

end

for i = 1:nsubs

    spm_jobman('run', jobs{i});

end
