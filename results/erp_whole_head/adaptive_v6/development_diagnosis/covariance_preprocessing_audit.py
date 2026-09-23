"""Read-only development audit of mean-baseline versus trial-contrast whitening."""
# %% 重建相同五例观测，只检查白化，不调用任何源反演。
from pathlib import Path
import hashlib
import json
import os
import sys
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
output = Path(__file__).resolve().parent
root = output.parents[3]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import protocol, erp_replicates, erp_trial_covariance
import run_erp_whole_head_matrix as original

base = root / "results/erp_whole_head/adaptive_v6"
mean_dir, trial_dir = base / "dev_predictive_wide_300", base / "dev_trial_covariance_irls_5_5"
trial_meta = json.loads((trial_dir / "metadata.json").read_text(encoding="utf-8"))
mean_meta = json.loads((mean_dir / "metadata.json").read_text(encoding="utf-8"))
assert trial_meta["complete"] and trial_meta["phase"] == mean_meta["phase"] == "development"
for name in ("benchmark/erp_replicates.py", "benchmark/erp_trial_covariance.py", "benchmark/methods.py"):
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == trial_meta["code_sha256"][name.replace("/", "\\")]
manifest_path = base / "protocol/development_manifest.json"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == trial_meta["manifest_sha256"] == mean_meta["manifest_sha256"]
cases = {case["case_id"]: case for case in json.loads(manifest_path.read_text(encoding="utf-8"))}
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
assert original._shared_fingerprint(shared) == trial_meta["shared_fingerprint"] == mean_meta["shared_fingerprint"]
rows = []
for case_id in trial_meta["case_ids"]:
    case = cases[case_id]
    mean = erp_replicates.prepare_replicated_case(shared, case, trial_meta["seed_root"])
    trial = erp_trial_covariance.prepare_trial_covariance_case(shared, case, trial_meta["seed_root"])
    saved = json.loads((trial_dir / (case_id + ".json")).read_text(encoding="utf-8"))
    old = json.loads((mean_dir / (case_id + ".json")).read_text(encoding="utf-8"))
    assert trial["metadata"] == saved["simulation"] and mean["metadata"] == old["simulation"]
    assert not np.any(mean["truth"][:, :200]), "old whitening window must contain no simulated signal"
    for key in ("eeg_train", "meg_train", "eeg_confirmation", "meg_confirmation"):
        assert np.array_equal(mean[key], trial[key])
        assert hashlib.sha256(np.ascontiguousarray(trial[key]).tobytes()).hexdigest() == saved["raw_mean_sha256"][key]
    baseline = mean["baseline"]
    counts = trial["metadata"]["trial_baseline_covariance"]
    n_trials, n_baseline = counts["n_training_trials"], counts["n_baseline"]
    split_mean = split_trial = 0
    for index, modality in enumerate(("eeg", "meg")):
        raw_train, raw_check = mean[modality + "_train"], mean[modality + "_confirmation"]
        centered = raw_train[:, :200] - raw_train[:, :200].mean(axis=1, keepdims=True)
        mean_covariance = centered @ centered.T / 199
        info = counts["modalities"][modality]
        seed = np.random.SeedSequence(info["seed_entropy"], spawn_key=info["seed_spawn_key"])
        factor = np.asarray(shared["noise_factor_" + modality], dtype=float)
        scale = mean["metadata"]["noise_scale"][modality]
        contrast = factor @ np.random.default_rng(seed).standard_normal((len(factor), n_trials * n_baseline))
        contrast = (contrast * scale).reshape(len(factor), n_trials, n_baseline)
        contrast -= contrast.mean(axis=1, keepdims=True)
        contrast -= contrast.mean(axis=2, keepdims=True)
        samples = contrast.reshape(len(factor), -1)
        trial_covariance = samples @ samples.T / ((n_trials - 1) * (n_baseline - 1) * n_trials)
        # 下列生成协方差只用于白化质量审计，从未传入已有或新反演。
        generating_mean_covariance = factor @ factor.T * scale ** 2 / n_trials
        record = dict(case_id=case_id, configuration_kind=case["configuration_kind"], modality=modality,
                      raw_channels=len(factor), mean_covariance_dof=199,
                      trial_covariance_dof=(n_trials - 1) * (n_baseline - 1),
                      old_modality_weight=mean["metadata"]["modality_weights"][index],
                      trial_modality_weight=trial["metadata"]["modality_weights"][index])
        retained_vectors = []
        for label, covariance, prepared, split in (("mean", mean_covariance, mean, split_mean),
                                                    ("trial", trial_covariance, trial, split_trial)):
            values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
            keep = values > values[-1] * 1e-8
            rank = int(keep.sum())
            whitener = (vectors[:, keep] / np.sqrt(values[keep])).T
            retained_vectors.append(vectors[:, keep])
            assert rank == prepared["metadata"]["retained_channels"][index]
            assert np.allclose(whitener @ shared["gain_" + modality], prepared["gain"][split:split + rank], rtol=1e-11, atol=1e-10)
            expected_white_covariance = whitener @ generating_mean_covariance @ whitener.T
            white_eigenvalues = np.linalg.eigvalsh((expected_white_covariance + expected_white_covariance.T) / 2)
            check_centered = raw_check[:, :200] - raw_check[:, :200].mean(axis=1, keepdims=True)
            check_white = whitener @ check_centered
            record[label] = dict(retained_rank=rank,
                condition_full=float(values[-1] / values[0]) if values[0] > 0 else None,
                condition_retained=float(values[-1] / values[keep][0]),
                trace_relative_to_generating_covariance=float(np.trace(covariance) / np.trace(generating_mean_covariance)),
                expected_white_noise_variance_mean=float(white_eigenvalues.mean()),
                expected_white_noise_variance_min=float(white_eigenvalues[0]),
                expected_white_noise_variance_max=float(white_eigenvalues[-1]),
                confirmation_first200_white_variance_per_retained_channel=float(np.sum(check_white ** 2) / (199 * rank)))
        cosines = np.linalg.svd(retained_vectors[0].T @ retained_vectors[1], compute_uv=False)
        record["largest_kept_subspace_angle_degrees"] = float(np.degrees(np.arccos(np.clip(cosines.min(), -1, 1))))
        rows.append(record)
        split_mean += mean["metadata"]["retained_channels"][index]
        split_trial += trial["metadata"]["retained_channels"][index]
        print(case["configuration_kind"], modality,
              "rank", record["mean"]["retained_rank"], record["trial"]["retained_rank"],
              "condition", round(record["mean"]["condition_retained"], 1), round(record["trial"]["condition_retained"], 1),
              "white_noise", round(record["mean"]["expected_white_noise_variance_mean"], 3), round(record["trial"]["expected_white_noise_variance_mean"], 3),
              "weight", round(record["old_modality_weight"], 3), round(record["trial_modality_weight"], 3), flush=True)
assert len(rows) == 10
(output / "covariance_preprocessing_audit.json").write_text(json.dumps(dict(
    scope="Five completed 5/5 dB development cases only; preprocessing audit, no inverses and no calibration/validation reads",
    raw_means_unchanged=True, metadata_and_code_fingerprints_checked=True,
    oracle_covariance_used_only_for_diagnostic_whitening_quality=True,
    source_means="exact original train/check means regenerated and matched to saved hashes",
    rows=rows), indent=2) + "\n", encoding="utf-8")
