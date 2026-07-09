clear; clc;

this_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);
data_root = fullfile(project_root, 'generated');
run_root = fullfile(this_dir, 'results', 'sisses_runs');
out_root = fullfile(this_dir, 'results', 'weighted_sisses');

patched_dir = fullfile(fileparts(project_root), 'sisses_diagnosis', 'patched');
original_sisses_dir = 'F:\博士\工作＆汇报\源定位\李文\SISSES-code-for-MEG-main\SISSES-code-for-MEG-main\SISSES\SISSES';
addpath(patched_dir);
addpath(original_sisses_dir);

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

scenarios = {'deep_only', 'surface_only', 'deep_plus_surface', 'deep_plus_two_surface'};
StimTime = 200;
param_grid = [
    0.05  0.010 0.0
    0.10  0.010 0.0
    0.10  0.050 0.2
];
weight_profiles = [
    0.70 1.35 1.65
    0.60 1.50 1.90
    0.50 1.70 2.20
];

summary_all = table();

for s = 1:numel(scenarios)
    scenario = scenarios{s};
    fprintf('\n================ weighted SISSES %s ================\n', scenario);

    data_dir = fullfile(data_root, scenario);
    out_dir = fullfile(out_root, scenario);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    S_eeg = load(fullfile(data_dir, 'sub_EEG.mat'));
    S_meg = load(fullfile(data_dir, 'sub_MEG.mat'));
    truth = load(fullfile(data_dir, 's_true.mat'));
    n_surf = double(truth.n_surf);
    VertConn = S_eeg.VertConn;

    direct = load(fullfile(run_root, scenario, 'both', 's_wen.mat'));
    meg_only = load(fullfile(run_root, scenario, 'meg_only', 's_wen.mat'));
    amp_direct = sqrt(sum(double(direct.s_wen).^2, 2));
    amp_meg = sqrt(sum(double(meg_only.s_wen).^2, 2));
    global_peak = max(amp_direct);

    surface_score = max(norm01(amp_direct(1:n_surf)), norm01(amp_meg(1:n_surf)));
    has_surface = max(amp_direct(1:n_surf)) >= 0.10 * global_peak;
    surface_candidate = false(size(amp_direct));
    if has_surface
        surface_candidate(1:n_surf) = surface_score >= 0.12;
    end

    deep_candidate = false(size(amp_direct));
    if n_surf < numel(amp_direct)
        deep_amp = amp_direct(n_surf + 1:end);
        has_deep = max(deep_amp) >= 0.10 * global_peak;
        if has_deep
            deep_candidate(n_surf + find(deep_amp >= 0.22 * max(deep_amp))) = true;
        end
    end

    [B_eeg_white, L_eeg_white] = white_data_safe(double(S_eeg.F), double(S_eeg.Gain), StimTime);
    [B_meg_white, L_meg_white] = white_data_safe(double(S_meg.F), double(S_meg.Gain), StimTime);
    B = [B_eeg_white; B_meg_white];
    L = [L_eeg_white; L_meg_white];
    col_norm = sqrt(sum(L.^2, 1));
    col_norm = max(col_norm, median(col_norm) * 1e-6);
    L_work_base = L ./ col_norm;
    Dic1 = TBFSelection(B, 0, 'threshold', 'Kaiser');

    best_score = inf;
    best_s_wen = [];
    best_meta = struct();
    scan_rows = table();

    for p = 1:size(weight_profiles, 1)
        profile = weight_profiles(p, :);
        source_weight = profile(1) * ones(size(amp_direct));
        source_weight(surface_candidate) = profile(2);
        source_weight(deep_candidate) = profile(3);
        L_weighted = L_work_base .* source_weight(:)';

        for r = 1:size(param_grid, 1)
            sigma = param_grid(r, 1);
            alpha = param_grid(r, 2);
            tau = param_grid(r, 3);

            tic;
            run_text = evalc('q_wen = wen_sisses_safe(B, L_weighted, VertConn, Dic1'', sigma, alpha, tau);');
            elapsed_sec = toc;

            s_wen = (q_wen .* source_weight(:)) ./ col_norm(:);
            amp = sqrt(sum(s_wen.^2, 2));
            residual = B - L * s_wen;
            rss = norm(residual, 'fro') ^ 2;
            rel_residual = norm(residual, 'fro') / max(norm(B, 'fro'), eps);
            active_count = sum(amp >= 0.10 * max(amp));
            temporal_rank = max(1, rank(Dic1));
            n_obs = numel(B);
            n_params = active_count * temporal_rank;
            selection_bic = n_obs * log(rss / n_obs + eps) + n_params * log(n_obs);

            fprintf('%s profile=%d sigma=%g alpha=%g tau=%g relRes=%.4f active=%d BIC=%.3f time=%.1fs\n', ...
                scenario, p, sigma, alpha, tau, rel_residual, active_count, selection_bic, elapsed_sec);

            new_row = table({scenario}, p, profile(1), profile(2), profile(3), sigma, alpha, tau, ...
                elapsed_sec, rel_residual, active_count, temporal_rank, selection_bic, ...
                sum(surface_candidate), sum(deep_candidate), ...
                'VariableNames', {'scenario','profile','base_weight','surface_weight','deep_weight', ...
                'sigma','alpha','tau','elapsed_sec','rel_residual','active_count','temporal_rank', ...
                'selection_bic','surface_candidate_count','deep_candidate_count'});
            scan_rows = [scan_rows; new_row];

            if selection_bic < best_score
                best_score = selection_bic;
                best_s_wen = s_wen;
                best_meta.scenario = scenario;
                best_meta.profile = p;
                best_meta.base_weight = profile(1);
                best_meta.surface_weight = profile(2);
                best_meta.deep_weight = profile(3);
                best_meta.sigma = sigma;
                best_meta.alpha = alpha;
                best_meta.tau = tau;
                best_meta.rel_residual = rel_residual;
                best_meta.active_count = active_count;
                best_meta.temporal_rank = temporal_rank;
                best_meta.selection_bic = selection_bic;
                best_meta.surface_candidate_count = sum(surface_candidate);
                best_meta.deep_candidate_count = sum(deep_candidate);
                best_meta.run_text = run_text;
            end
        end
    end

    scan_rows = sortrows(scan_rows, 'selection_bic', 'ascend');
    writetable(scan_rows, fullfile(out_dir, 'scan.csv'));
    s_wen = best_s_wen;
    save(fullfile(out_dir, 's_wen.mat'), 's_wen', 'best_meta', '-v7.3');
    summary_all = [summary_all; scan_rows(1, :)];
end

writetable(summary_all, fullfile(out_root, 'summary.csv'));
save(fullfile(out_root, 'summary.mat'), 'summary_all');
disp(summary_all);

function y = norm01(x)
    x = double(x);
    m = max(x);
    if m > 0
        y = x ./ m;
    else
        y = x;
    end
end
