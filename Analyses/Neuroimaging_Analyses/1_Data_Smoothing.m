spm('defaults', 'fmri');
spm_jobman('initcfg');

subs = [1:11 13:19 21:25 27:51 53];
nsubs = length(subs);
nruns = 4;
jobs = cell(nsubs, 1);

for sub = subs

    mkdir(sprintf('C:/Users/.../Experiment 2/Analyses/first_level/sub-%d', sub))

end

for sub = subs

    dir = sprintf('C:/Users/...Experiment 2/Data/sub-%d/', sub);
    confounds_of_interest = {'^rot_[xyz]$', '^trans_[xyz]$', '^global_signal$'};

    for run = 1:nruns

        content = readtable([dir sprintf('sub-%d_task-ep2dboldrun%d_desc-confounds_timeseries.tsv', sub, run)], "FileType","text",'Delimiter', '\t');
        confounds_names = fieldnames(content);
        confounds_to_keep = regexp(confounds_names, strjoin(confounds_of_interest, '|'));
        confounds_to_keep = ~cellfun('isempty', confounds_to_keep);
        confounds_names = confounds_names(confounds_to_keep);

        for j = 1:numel(confounds_names)

            names{j} = confounds_names{j};
            R(:, j) = content.(confounds_names{j});

        end
        
        R = R(6:end,:);
        output_file_name = sprintf('sub-%d_task-ep2dboldrun%d_desc-confounds.txt', sub, run);
        output_file = array2table(R, 'VariableNames', names);
        writetable(output_file, [dir, output_file_name], 'WriteVariableNames', false);
        save([dir, sprintf('sub-%d_task-ep2dboldrun%d_desc-confounds.mat', sub, run)], 'R', 'names')

        clear R;
        clear names;
        clear output_file;

    end
end

for sub = subs

    input_dir = sprintf('C:/Users/.../Experiment 2/Data/sub-%d/func/', sub);
    output_dir = sprintf('C:/Users/.../Experiment 2/Analyses/first_level/sub-%d/', sub);

    for run = 1:nruns

        cd(input_dir);
        gunzip(sprintf('sub-%d_task-ep2dboldrun%d_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', sub, run), output_dir);

    end
end

for sub = subs

    dir = sprintf('C:/Users/...Experiment 2/Analyses/first_level/sub-%d', sub);
    files = cellstr(spm_select('FPList', dir, '\.nii$'));

    clear matlabbatch

    matlabbatch{1}.spm.spatial.smooth.data = files;
    matlabbatch{1}.spm.spatial.smooth.fwhm = [5 5 5];
    matlabbatch{1}.spm.spatial.smooth.dtype = 0;
    matlabbatch{1}.spm.spatial.smooth.im = 0;
    matlabbatch{1}.spm.spatial.smooth.prefix = 'smoothed_';
        
    spm_jobman('run', matlabbatch)

end
