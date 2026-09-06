clear; clc;

this_dir = fileparts(mfilename('fullpath'));
project_root = this_dir;
data_root = fullfile(this_dir, 'results', 'position_validation', 'generated');
out_root = fullfile(this_dir, 'results', 'position_validation', 'sisses_runs');

patched_dir = getenv('SISSES_PATCH_ROOT');
original_sisses_dir = getenv('SISSES_ROOT');
if isempty(patched_dir) || isempty(original_sisses_dir)
    error('Set SISSES_PATCH_ROOT and SISSES_ROOT before running this adapter.');
end
addpath(patched_dir);
addpath(original_sisses_dir);

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

jobs = dir(data_root);
StimTime = 200;
param_grid = [
    0.05  0.010 0.0
    0.10  0.050 0.2
];
summary_all = table();

for j = 1:numel(jobs)
    if ~jobs(j).isdir || startsWith(jobs(j).name, '.')
        continue;
    end
    job = jobs(j).name;
    out_dir = fullfile(out_root, job);
    out_file = fullfile(out_dir, 's_wen.mat');
    if exist(out_file, 'file')
        fprintf('Skip existing %s\n', job);
        continue;
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    data_dir = fullfile(data_root, job);
    S_eeg = load(fullfile(data_dir, 'sub_EEG.mat'));
    S_meg = load(fullfile(data_dir, 'sub_MEG.mat'));
    VertConn = S_eeg.VertConn;
    [B_eeg_white, L_eeg_white] = white_data_safe(double(S_eeg.F), double(S_eeg.Gain), StimTime);
    [B_meg_white, L_meg_white] = white_data_safe(double(S_meg.F), double(S_meg.Gain), StimTime);
    B = [B_eeg_white; B_meg_white];
    L = [L_eeg_white; L_meg_white];
    col_norm = sqrt(sum(L.^2, 1));
    col_norm = max(col_norm, median(col_norm) * 1e-6);
    L_work = L ./ col_norm;
    Dic1 = TBFSelection(B, 0, 'threshold', 'Kaiser');

    best_score = inf;
    best_s_wen = [];
    scan_rows = table();
    fprintf('\n================ validation %s ================\n', job);
    for r = 1:size(param_grid, 1)
        sigma = param_grid(r, 1);
        alpha = param_grid(r, 2);
        tau = param_grid(r, 3);
        tic;
        run_text = evalc('q_wen = wen_sisses_safe(B, L_work, VertConn, Dic1'', sigma, alpha, tau);');
        elapsed_sec = toc;
        s_wen = q_wen ./ col_norm(:);
        amp = sqrt(sum(s_wen.^2, 2));
        residual = B - L * s_wen;
        rss = norm(residual, 'fro') ^ 2;
        rel_residual = norm(residual, 'fro') / max(norm(B, 'fro'), eps);
        active_count = sum(amp >= 0.05 * max(amp));
        temporal_rank = max(1, rank(Dic1));
        n_obs = numel(B);
        n_params = active_count * temporal_rank;
        selection_bic = n_obs * log(rss / n_obs + eps) + n_params * log(n_obs);
        fprintf('%s sigma=%g alpha=%g tau=%g relRes=%.4f active=%d BIC=%.3f time=%.1fs\n', ...
            job, sigma, alpha, tau, rel_residual, active_count, selection_bic, elapsed_sec);
        new_row = table({job}, sigma, alpha, tau, elapsed_sec, rel_residual, active_count, temporal_rank, selection_bic, ...
            'VariableNames', {'job','sigma','alpha','tau','elapsed_sec','rel_residual','active_count','temporal_rank','selection_bic'});
        scan_rows = [scan_rows; new_row];
        if selection_bic < best_score
            best_score = selection_bic;
            best_s_wen = s_wen;
            best_meta = struct('job', job, 'sigma', sigma, 'alpha', alpha, 'tau', tau, ...
                'rel_residual', rel_residual, 'active_count', active_count, ...
                'temporal_rank', temporal_rank, 'selection_bic', selection_bic, 'run_text', run_text);
        end
    end
    scan_rows = sortrows(scan_rows, 'selection_bic', 'ascend');
    writetable(scan_rows, fullfile(out_dir, 'scan.csv'));
    s_wen = best_s_wen;
    save(out_file, 's_wen', 'best_meta', '-v7.3');
    summary_all = [summary_all; scan_rows(1, :)];
end

writetable(summary_all, fullfile(out_root, 'summary.csv'));
save(fullfile(out_root, 'summary.mat'), 'summary_all');
disp(summary_all);
