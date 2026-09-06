clear; clc;

% ============================================================
% ROI 深部源实验：SISSES 快速候选参数扫描
% 只跑少量参数，避免大网格中某些参数导致长时间迭代。
% ============================================================

root_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(root_dir);
data_root = fullfile(project_root, 'generated');
out_root = fullfile(project_root, 'results', 'latest', 'sisses_warm_start');

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

patched_dir = getenv('SISSES_PATCH_ROOT');
original_sisses_dir = getenv('SISSES_ROOT');
if isempty(patched_dir) || isempty(original_sisses_dir)
    error('Set SISSES_PATCH_ROOT and SISSES_ROOT before running this adapter.');
end
addpath(patched_dir);
addpath(original_sisses_dir);

scenarios = {'deep_only', 'surface_only', 'deep_plus_surface', 'deep_plus_two_surface'};
StimTime = 200;

param_grid = [
    0.01  0.005 0.0
    0.05  0.010 0.0
    0.10  0.010 0.0
    0.10  0.050 0.2
];

summary_all = table();

for s = 1:numel(scenarios)
    scenario = scenarios{s};
    fprintf('\n================ quick scan %s ================\n', scenario);

    data_dir = fullfile(data_root, scenario);
    out_dir = fullfile(out_root, scenario);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    S_eeg = load(fullfile(data_dir, 'sub_EEG.mat'));
    S_meg = load(fullfile(data_dir, 'sub_MEG.mat'));
    truth = load(fullfile(data_dir, 's_true.mat'));

    Y_eeg = double(S_eeg.F);
    G_eeg = double(S_eeg.Gain);
    Y_meg = double(S_meg.F);
    G_meg = double(S_meg.Gain);
    VertConn = S_eeg.VertConn;

    src_vertices = double(truth.src_vertices);
    n_surf = double(truth.n_surf);
    true_deep_idx = double(truth.true_deep_idx1);
    true_deep_pos = src_vertices(true_deep_idx, :);
    if isfield(truth, 'has_deep_source')
        has_deep_source = logical(truth.has_deep_source);
    else
        has_deep_source = true;
    end
    true_surface_indices = double(truth.true_surface_indices1(:));

    [B_eeg_white, L_eeg_white] = white_data_safe(Y_eeg, G_eeg, StimTime);
    [B_meg_white, L_meg_white] = white_data_safe(Y_meg, G_meg, StimTime);
    B = [B_eeg_white; B_meg_white];
    L = [L_eeg_white; L_meg_white];

    col_norm = sqrt(sum(L.^2, 1));
    col_norm = max(col_norm, median(col_norm) * 1e-6);
    L_work = L ./ col_norm;

    Dic1 = TBFSelection(B, 0, 'threshold', 'Kaiser');

    best_score = inf;
    best_s_wen = [];
    best_meta = struct();
    scan_rows = table();

    for r = 1:size(param_grid, 1)
        sigma = param_grid(r, 1);
        alpha = param_grid(r, 2);
        tau = param_grid(r, 3);

        tic;
        run_text = evalc('q_wen = wen_sisses_safe(B, L_work, VertConn, Dic1'', sigma, alpha, tau);');
        elapsed_sec = toc;

        s_wen = q_wen ./ col_norm(:);
        amp = sqrt(sum(s_wen.^2, 2));
        [~, global_peak] = max(amp);

        [~, deep_local] = max(amp(n_surf+1:end));
        deep_peak = n_surf + deep_local;
        if has_deep_source
            deep_dle_mm = norm(src_vertices(deep_peak, :) - true_deep_pos) * 1000;
            global_dle_deep_mm = norm(src_vertices(global_peak, :) - true_deep_pos) * 1000;
        else
            deep_dle_mm = NaN;
            global_dle_deep_mm = NaN;
        end
        deep_energy_ratio = sum(amp(n_surf+1:end)) / sum(amp);

        if isempty(true_surface_indices)
            surface_dle_mm = NaN;
        else
            [~, surface_peak] = max(amp(1:n_surf));
            surf_dist = sqrt(sum((src_vertices(surface_peak, :) - src_vertices(true_surface_indices, :)).^2, 2));
            surface_dle_mm = min(surf_dist) * 1000;
        end

        residual = B - L * s_wen;
        rss = norm(residual, 'fro') ^ 2;
        rel_residual = norm(residual, 'fro') / max(norm(B, 'fro'), eps);
        active_count = sum(amp >= 0.05 * max(amp));
        temporal_rank = max(1, rank(Dic1));
        n_obs = numel(B);
        n_params = active_count * temporal_rank;
        selection_bic = n_obs * log(rss / n_obs + eps) + n_params * log(n_obs);
        score = selection_bic;

        fprintf('%s sigma=%g alpha=%g tau=%g global=%d deepDLE=%.3f surfDLE=%.3f deepRatio=%.4f relRes=%.4f active=%d BIC=%.3f time=%.1fs\n', ...
            scenario, sigma, alpha, tau, global_peak, deep_dle_mm, surface_dle_mm, deep_energy_ratio, ...
            rel_residual, active_count, selection_bic, elapsed_sec);

        new_row = table({scenario}, sigma, alpha, tau, elapsed_sec, global_peak, global_dle_deep_mm, ...
            deep_peak, deep_dle_mm, surface_dle_mm, deep_energy_ratio, rel_residual, ...
            active_count, temporal_rank, selection_bic, score, ...
            'VariableNames', {'scenario','sigma','alpha','tau','elapsed_sec','global_peak','global_dle_to_deep_mm', ...
            'deep_peak','deep_dle_mm','surface_dle_mm','deep_energy_ratio','rel_residual', ...
            'active_count','temporal_rank','selection_bic','score'});
        scan_rows = [scan_rows; new_row];

        if score < best_score
            best_score = score;
            best_s_wen = s_wen;
            best_meta.scenario = scenario;
            best_meta.sigma = sigma;
            best_meta.alpha = alpha;
            best_meta.tau = tau;
            best_meta.score = score;
            best_meta.global_peak = global_peak;
            best_meta.deep_peak = deep_peak;
            best_meta.deep_dle_mm = deep_dle_mm;
            best_meta.surface_dle_mm = surface_dle_mm;
            best_meta.deep_energy_ratio = deep_energy_ratio;
            best_meta.rel_residual = rel_residual;
            best_meta.active_count = active_count;
            best_meta.temporal_rank = temporal_rank;
            best_meta.selection_bic = selection_bic;
            best_meta.run_text = run_text;
        end
    end

    scan_rows = sortrows(scan_rows, 'score', 'ascend');
    writetable(scan_rows, fullfile(out_dir, 'sisses_quick_scan.csv'));
    save(fullfile(out_dir, 'sisses_quick_scan.mat'), 'scan_rows', 'best_meta');

    s_wen = best_s_wen;
    save(fullfile(out_dir, 's_wen_sisses_fusion_quick_best.mat'), 's_wen', 'best_meta', '-v7.3');
    summary_all = [summary_all; scan_rows(1, :)];
end

writetable(summary_all, fullfile(out_root, 'sisses_fusion_quick_scan_summary.csv'));
save(fullfile(out_root, 'sisses_fusion_quick_scan_summary.mat'), 'summary_all');
disp(summary_all);
