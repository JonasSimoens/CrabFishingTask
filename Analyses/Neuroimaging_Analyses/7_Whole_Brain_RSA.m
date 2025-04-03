spm('defaults', 'fmri');
spm_jobman('initcfg');

cwd = 'C:/Users/.../Experiment 2/Analyses/';
cd(cwd);

subs = [1:11 13:19 21:25 27:51 53];

for sub = subs

    v = spm_vol([cwd sprintf('first_level/sub-%d/beta_0001.nii', sub)]);
    v.fname = [cwd sprintf('group_level/corr_%d.nii', sub)];
    data = nan(v.dim);

    betas = readtable([cwd sprintf('tau_values/tau_values_part_%d.csv', sub)]);

    for row = 1:height(betas)

        data(betas.x(row), betas.y(row), betas.z(row)) = betas.b2(row);

    end

    spm_write_vol(v, data);

end

cwd = 'C:/Users/.../Experiment 2/Analyses/group_level/';
cd(cwd);

files = cellstr(spm_select('FPList', cwd, 'corr*'));

clear matlabbatch

matlabbatch{1}.spm.stats.factorial_design.dir = {cwd};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = files;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

spm_jobman('run', matlabbatch)

clear matlabbatch

file = cellstr(spm_select('FPList', cwd, 'SPM*'));

matlabbatch{1}.spm.stats.fmri_est.spmmat = file;
matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;

spm_jobman('run', matlabbatch)

clear matlabbatch

matlabbatch{1}.spm.stats.con.spmmat = file;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'RDM_corr';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = 1;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.delete = 1;

spm_jobman('run', matlabbatch)
