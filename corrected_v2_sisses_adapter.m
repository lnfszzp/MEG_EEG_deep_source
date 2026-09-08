function output_files = corrected_v2_sisses_adapter(input_file, output_dir, expected_manifest, case_limit, solver_fingerprint)
%CORRECTED_V2_SISSES_ADAPTER Run external SISSES on one frozen MAT chunk.
% Third-party code is loaded from SISSES_ROOT.  SISSES_FAST_ROOT is optional.

if nargin == 1 && strcmpi(char(string(input_file)), 'selftest')
    [~, use_fast] = external_solver();
    assert(requested_case_count(20, 0) == 20 && ...
        requested_case_count(20, -1) == 20 && requested_case_count(20, 1) == 1);
    fprintf('corrected_v2_sisses_adapter ok; fast=%d\n', use_fast);
    output_files = {};
    return;
end
if nargin < 3
    error('Usage: corrected_v2_sisses_adapter(input, output, manifest_sha, [case_limit], [solver_fingerprint])');
end
if nargin < 4 || isempty(case_limit)
    case_limit = inf;
end
if nargin < 5
    solver_fingerprint = '';
end

[~, use_fast] = external_solver();
input = load(input_file);
required = {'F_EEG','F_MEG','Gain_EEG','Gain_MEG','VertConn','case_ids', ...
    'manifest_sha256','format_version'};
for index = 1:numel(required)
    assert(isfield(input, required{index}), 'Input MAT lacks %s.', required{index});
end
assert(strcmp(text_scalar(input.format_version), 'strict_blind_sisses_input_v2'), ...
    'Unsupported input format.');
manifest_sha256 = text_scalar(input.manifest_sha256);
assert(strcmp(manifest_sha256, char(string(expected_manifest))), ...
    'Input manifest SHA-256 mismatch.');

F_EEG = double(input.F_EEG);
F_MEG = double(input.F_MEG);
Gain_EEG = double(input.Gain_EEG);
Gain_MEG = double(input.Gain_MEG);
assert(ndims(F_EEG) <= 3 && ndims(F_MEG) <= 3, 'Observations must be channel-by-time-by-case.');
assert(ismatrix(Gain_EEG) && ismatrix(Gain_MEG), 'Gain matrices must be two-dimensional.');
n_cases = size(F_EEG, 3);
n_times = size(F_EEG, 2);
n_sources = size(Gain_EEG, 2);
assert(size(F_MEG,2) == n_times && size(F_MEG,3) == n_cases, 'EEG/MEG time or case dimensions differ.');
assert(size(F_EEG,1) == size(Gain_EEG,1) && size(F_MEG,1) == size(Gain_MEG,1), 'Observation/gain channel mismatch.');
assert(size(Gain_MEG,2) == n_sources && n_times >= 200, 'Gain source count differs or baseline is too short.');
assert(all(isfinite(F_EEG(:))) && all(isfinite(F_MEG(:))) && ...
    all(isfinite(Gain_EEG(:))) && all(isfinite(Gain_MEG(:))), 'Input contains NaN or Inf.');

VertConn = spones(sparse(double(input.VertConn)));
assert(isequal(size(VertConn), [n_sources n_sources]), 'VertConn shape mismatch.');
assert(nnz(VertConn - VertConn') == 0 && all(sum(VertConn,2) > 0), ...
    'VertConn must be symmetric and contain no isolated source.');
case_ids = normalize_case_ids(input.case_ids, n_cases);
case_count = requested_case_count(n_cases, case_limit);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_files = cell(case_count, 1);
failed = 0;
for case_index = 1:case_count
    case_id = case_ids{case_index};
    output_file = fullfile(output_dir, sprintf('case_%04d.mat', case_index));
    if valid_output(output_file, case_id, manifest_sha256, solver_fingerprint, n_sources, n_times)
        output_files{case_index} = output_file;
        fprintf('skip %s\n', case_id);
        continue;
    end

    source_estimates = nan(n_sources, n_times);
    method_names = {'SISSES'};
    success = false;
    errors = {''};
    source_shape = [n_sources n_times 1];
    format_version = 'corrected_v2_sisses_output_v1';
    preprocessing_elapsed_sec = nan;
    metadata = struct('method','SISSES','variant','', 'preprocessing','', ...
        'temporal_rank',nan,'parameters',struct(), 'selection_bic',nan, ...
        'rel_residual',nan,'active_count',nan,'scan',struct([]), ...
        'elapsed_sec',nan,'success',false,'error','');
    try
        timer = tic;
        [B_EEG, L_EEG] = whiten_modality(F_EEG(:,:,case_index), Gain_EEG, 200);
        [B_MEG, L_MEG] = whiten_modality(F_MEG(:,:,case_index), Gain_MEG, 200);
        B = [B_EEG; B_MEG];
        L = [L_EEG; L_MEG];
        col_norm = sqrt(sum(L.^2, 1));
        floor_value = max(median(col_norm) * 1e-6, eps);
        col_norm = max(col_norm, floor_value);
        L_work = L ./ col_norm;
        GB = TBFSelection(B, 0, 'threshold', 'Kaiser')';
        assert(~isempty(GB) && all(isfinite(GB(:))), 'Kaiser TBF selection returned no finite basis.');
        preprocessing_elapsed_sec = toc(timer);

        grid = [0.05 0.010 0.0; 0.10 0.050 0.2];
        scan = repmat(struct('sigma',nan,'alpha',nan,'tau',nan,'selection_bic',inf, ...
            'rel_residual',nan,'active_count',nan,'elapsed_sec',nan,'success',false,'error',''), size(grid,1), 1);
        best_bic = inf;
        best_source = [];
        best_index = 0;
        for row = 1:size(grid,1)
            sigma = grid(row,1); alpha = grid(row,2); tau = grid(row,3);
            scan(row).sigma = sigma; scan(row).alpha = alpha; scan(row).tau = tau;
            solver_timer = tic;
            try
                if use_fast
                    [solutions, ~] = wen_sisses_fast(B, L_work, VertConn, GB, sigma, alpha, tau);
                    candidate = solutions{1};
                else
                    candidate = wen_sisses(B, L_work, VertConn, GB, sigma, alpha, tau);
                end
                candidate = double(candidate) ./ col_norm(:);
                assert(isequal(size(candidate), [n_sources n_times]) && all(isfinite(candidate(:))), ...
                    'SISSES returned an invalid source matrix.');
                residual = B - L * candidate;
                rss = norm(residual, 'fro') ^ 2;
                rel_residual = norm(residual, 'fro') / max(norm(B, 'fro'), eps);
                amplitude = sqrt(sum(candidate.^2, 2));
                peak = max(amplitude);
                active_count = 0;
                if peak > 0
                    active_count = sum(amplitude >= 0.05 * peak);
                end
                n_observations = numel(B);
                n_parameters = active_count * size(GB,2);
                bic = n_observations * log(rss / n_observations + eps) + n_parameters * log(n_observations);
                scan(row).selection_bic = bic;
                scan(row).rel_residual = rel_residual;
                scan(row).active_count = active_count;
                scan(row).success = true;
                if bic < best_bic
                    best_bic = bic; best_source = candidate; best_index = row;
                end
            catch exception
                scan(row).error = exception_text(exception);
            end
            scan(row).elapsed_sec = toc(solver_timer);
        end
        assert(best_index > 0, 'All SISSES parameter candidates failed.');
        source_estimates = best_source;
        success = true;
        metadata.method = 'SISSES';
        metadata.variant = 'joint EEG+MEG SISSES; two-candidate observation-only BIC selection';
        if use_fast
            metadata.variant = [metadata.variant '; external wen_sisses_fast'];
        else
            metadata.variant = [metadata.variant '; upstream wen_sisses'];
        end
        metadata.preprocessing = 'per-modality first-200 whitening; joint stack; joint leadfield column normalization; Kaiser TBF';
        metadata.temporal_rank = size(GB,2);
        metadata.parameters = struct('sigma',grid(best_index,1),'alpha',grid(best_index,2),'tau',grid(best_index,3));
        metadata.selection_bic = best_bic;
        metadata.rel_residual = scan(best_index).rel_residual;
        metadata.active_count = scan(best_index).active_count;
        metadata.scan = scan;
        metadata.elapsed_sec = sum([scan.elapsed_sec]);
        metadata.success = true;
    catch exception
        failed = failed + 1;
        errors = {exception_text(exception)};
        metadata.error = errors{1};
    end

    temporary = [output_file '.tmp.mat'];
    save(temporary, 'case_id','method_names','source_estimates','source_shape', ...
        'metadata','success','errors','preprocessing_elapsed_sec','format_version', ...
        'manifest_sha256','solver_fingerprint','-v7');
    movefile(temporary, output_file, 'f');
    output_files{case_index} = output_file;
    fprintf('%s %s\n', case_id, char(string(success)));
end
if failed
    error('%d SISSES case(s) failed; outputs retain their error metadata.', failed);
end
end


function [root, use_fast] = external_solver()
root = getenv('SISSES_ROOT');
assert(~isempty(root) && exist(root,'dir') == 7, 'Set SISSES_ROOT to the external SISSES directory.');
required = {'wen_sisses.m','wen_mrf_admm.m','VariationEdge.m','TBFSelection.m'};
for index = 1:numel(required)
    assert(exist(fullfile(root, required{index}), 'file') == 2, 'SISSES_ROOT lacks %s.', required{index});
end
addpath(root);
fast_root = getenv('SISSES_FAST_ROOT');
use_fast = ~isempty(fast_root);
if use_fast
    assert(exist(fullfile(fast_root, 'wen_sisses_fast.m'), 'file') == 2, ...
        'SISSES_FAST_ROOT lacks wen_sisses_fast.m.');
    addpath(fast_root);
end
end


function count = requested_case_count(total, requested)
if isinf(requested) || requested <= 0
    count = total;
else
    count = min(total, floor(double(requested)));
end
end


function [B_white, L_white] = whiten_modality(B, Gain, baseline_samples)
baseline = B(:,1:baseline_samples) - mean(B(:,1:baseline_samples),2);
covariance = (baseline * baseline') / (baseline_samples - 1);
covariance = (covariance + covariance') / 2;
[vectors, values] = eig(covariance);
[values, order] = sort(real(diag(values)), 'descend');
vectors = real(vectors(:,order));
keep = values > max(values) * 1e-8;
assert(any(keep), 'Noise covariance has no positive eigenvalues.');
W = diag(1 ./ sqrt(values(keep))) * vectors(:,keep)';
B_white = W * B;
L_white = W * Gain;
end


function ids = normalize_case_ids(value, expected)
if iscell(value)
    ids = cellfun(@(item) char(string(item)), value(:), 'UniformOutput', false);
elseif isstring(value)
    ids = cellstr(value(:));
elseif ischar(value) && expected == 1
    ids = {value};
else
    error('case_ids must be a cell/string vector.');
end
assert(numel(ids) == expected && numel(unique(ids)) == expected, 'case_ids count or uniqueness is invalid.');
end


function value = text_scalar(raw)
value = char(string(raw));
value = strtrim(value(:)');
end


function valid = valid_output(path, case_id, manifest_sha, fingerprint, n_sources, n_times)
valid = false;
if exist(path, 'file') ~= 2
    return;
end
try
    fields = load(path, 'case_id','manifest_sha256','solver_fingerprint','format_version','success','source_shape');
    info = whos('-file', path);
    source = info(strcmp({info.name}, 'source_estimates'));
    valid = numel(source) == 1 && source.size(1) == n_sources && source.size(2) == n_times && ...
        logical(fields.success) && strcmp(text_scalar(fields.case_id), case_id) && ...
        strcmp(text_scalar(fields.manifest_sha256), manifest_sha) && ...
        strcmp(text_scalar(fields.solver_fingerprint), char(string(fingerprint))) && ...
        strcmp(text_scalar(fields.format_version), 'corrected_v2_sisses_output_v1') && ...
        isequal(double(fields.source_shape(:)'), [n_sources n_times 1]);
catch
    valid = false;
end
end


function value = exception_text(exception)
value = getReport(exception, 'extended', 'hyperlinks', 'off');
end
