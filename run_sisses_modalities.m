clear; clc;

this_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);
data_root = fullfile(project_root, 'generated');
out_root = fullfile(this_dir, 'results', 'sisses_runs');

patched_dir = fullfile(fileparts(project_root), 'sisses_diagnosis', 'patched');
original_sisses_dir = 'F:\博士\工作＆汇报\源定位\李文\SISSES-code-for-MEG-main\SISSES-code-for-MEG-main\SISSES\SISSES';
addpath(patched_dir);
addpath(original_sisses_dir);

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

scenarios = {'deep_only', 'surface_only', 'deep_plus_surface', 'deep_plus_two_surface'};
modes = {'both', 'meg_only', 'eeg_only'};
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
    data_dir = fullfile(data_root, scenario);
    S_eeg = load(fullfile(data_dir, 'sub_EEG.mat'));
    S_meg = load(fullfile(data_dir, 'sub_MEG.mat'));

    VertConn = S_eeg.VertConn;
    [B_eeg_white, L_eeg_white] = white_data_safe(double(S_eeg.F), double(S_eeg.Gain), StimTime);
    [B_meg_white, L_meg_white] = white_data_safe(double(S_meg.F), double(S_meg.Gain), StimTime);

    for m = 1:numel(modes)
        mode = modes{m};
        out_dir = fullfile(out_root, scenario, mode);
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end
        out_file = fullfile(out_dir, 's_wen.mat');
        if exist(out_file, 'file')
            fprintf('Skip existing %s %s\n', scenario, mode);
            continue;
        end

        if strcmp(mode, 'both')
            B = [B_eeg_white; B_meg_white];
            L = [L_eeg_white; L_meg_white];
        elseif strcmp(mode, 'meg_only')
            B = B_meg_white;
            L = L_meg_white;
        else
            B = B_eeg_white;
            L = L_eeg_white;
        end

        col_norm = sqrt(sum(L.^2, 1));
        col_norm = max(col_norm, median(col_norm) * 1e-6);
        L_work = L ./ col_norm;
        Dic1 = TBFSelection(B, 0, 'threshold', 'Kaiser');

        best_score = inf;
        best_s_wen = [];
        best_meta = struct();
        scan_rows = table();

        fprintf('\n================ %s %s ================\n', scenario, mode);
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

            fprintf('%s %s sigma=%g alpha=%g tau=%g relRes=%.4f active=%d BIC=%.3f time=%.1fs\n', ...
                scenario, mode, sigma, alpha, tau, rel_residual, active_count, selection_bic, elapsed_sec);

            new_row = table({scenario}, {mode}, sigma, alpha, tau, elapsed_sec, ...
                rel_residual, active_count, temporal_rank, selection_bic, ...
                'VariableNames', {'scenario','mode','sigma','alpha','tau','elapsed_sec', ...
                'rel_residual','active_count','temporal_rank','selection_bic'});
            scan_rows = [scan_rows; new_row];

            if selection_bic < best_score
                best_score = selection_bic;
                best_s_wen = s_wen;
                best_meta.scenario = scenario;
                best_meta.mode = mode;
                best_meta.sigma = sigma;
                best_meta.alpha = alpha;
                best_meta.tau = tau;
                best_meta.rel_residual = rel_residual;
                best_meta.active_count = active_count;
                best_meta.temporal_rank = temporal_rank;
                best_meta.selection_bic = selection_bic;
                best_meta.run_text = run_text;
            end
        end

        scan_rows = sortrows(scan_rows, 'selection_bic', 'ascend');
        writetable(scan_rows, fullfile(out_dir, 'scan.csv'));
        s_wen = best_s_wen;
        save(out_file, 's_wen', 'best_meta', '-v7.3');
        summary_all = [summary_all; scan_rows(1, :)];
    end
end

writetable(summary_all, fullfile(out_root, 'summary.csv'));
save(fullfile(out_root, 'summary.mat'), 'summary_all');
disp(summary_all);
