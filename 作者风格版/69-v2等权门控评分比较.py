"""# %% 只比较已冻结的15例开发拟合；不读取validation，也不重新运行逆解。"""

# %% 1. 导入与固定路径。输出目录在所有输入审计和计算通过前不会创建。
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import _smooth_temporal_basis
from metrics.user_metrics.An_auc import An_auc
import run_erp_whole_head_matrix as original

source = root / "results/erp_whole_head/adaptive_v6/dev_hierarchical_v2_equal_weight_outer80_source"
expected_manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_localization_v5_new_positions.json"
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/hierarchical_v2_equal_weight_score_comparison"
if output.exists():
    raise FileExistsError(f"不覆盖旧开发结果：{output}")


# %% 2. 先做轻量预检。未满15例或任何求解未收敛时立即停止，不创建输出。
metadata_path = source / "metadata.json"
evidence_path = source / "evidence.csv"
if not metadata_path.is_file() or not evidence_path.is_file():
    raise FileNotFoundError("等权开发运行还没有 metadata.json 和 evidence.csv")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
manifest = Path(metadata["manifest"])
if (metadata.get("phase") != "development"
        or metadata.get("covariance") != "trial"
        or metadata.get("primary_channel_weights") != "whitened_channel_equal"
        or metadata.get("calibration_path") is not None
        or manifest.resolve() != expected_manifest.resolve()):
    raise RuntimeError("只接受指定的等权、trial-covariance、未校准 development 运行")
if "validation" in str(manifest).lower():
    raise RuntimeError("本脚本禁止读取 validation manifest")
if hashlib.sha256(manifest.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
    raise RuntimeError("development manifest 哈希不匹配")

cases_in_order = json.loads(manifest.read_text(encoding="utf-8"))
case_ids = list(metadata["case_ids"])
if (len(cases_in_order) != 15 or len(case_ids) != 15 or len(set(case_ids)) != 15
        or case_ids != [case["case_id"] for case in cases_in_order]):
    raise RuntimeError("等权开发输入必须是 manifest 顺序一致的完整15例")
cases = {case["case_id"]: case for case in cases_in_order}
missing = [str(source / (case_id + suffix)) for case_id in case_ids
           for suffix in (".json", ".npz") if not (source / (case_id + suffix)).is_file()]
if missing:
    raise RuntimeError(f"等权开发运行尚未完成15例，缺少 {len(missing)} 个文件：{missing[0]}")

with evidence_path.open(encoding="utf-8-sig", newline="") as stream:
    evidence_rows = list(csv.DictReader(stream))
if (len(evidence_rows) != 15
        or [row["case_id"] for row in evidence_rows] != case_ids
        or any(row["null_converged"] != "1" or row["full_converged"] != "1"
               or row["all_converged"] != "1" for row in evidence_rows)):
    raise RuntimeError("15例 evidence 必须齐全、顺序一致且 H0/H1 求解全部收敛")

direct_dependencies = [
    "benchmark/erp_trial_covariance.py",
    "benchmark/erp_replicates.py",
    "benchmark/erp_protocol.py",
    "benchmark/protocol.py",
    "candidates/oaster_balanced.py",
    "metrics/user_metrics/An_auc.py",
    "metrics/user_metrics/An_roc.py",
    "run_erp_whole_head_matrix.py",
]
for name in direct_dependencies:
    metadata_key = name.replace("/", "\\")
    current_hash = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if metadata["code_sha256"].get(metadata_key) != current_hash:
        raise RuntimeError(f"重建直接依赖已变化，不能解释冻结结果：{name}")
script_path = Path(__file__).resolve()
frozen_input_paths = [metadata_path, evidence_path, manifest]
frozen_input_paths += [source / (case_id + suffix) for case_id in case_ids
                       for suffix in (".json", ".npz")]
frozen_input_sha256 = {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in frozen_input_paths
}


# %% 3. 精确重建独立确认观测，并审计每例冻结的 full/null 源估计。
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
if original._shared_fingerprint(shared) != metadata["shared_fingerprint"]:
    raise RuntimeError("共享增益矩阵、源空间或噪声模型与冻结运行不一致")

rows = []
for case_id in case_ids:
    case = cases[case_id]
    record = json.loads((source / (case_id + ".json")).read_text(encoding="utf-8"))
    observation = prepare_trial_covariance_case(
        shared, case, seed_root=metadata["seed_root"])
    if record.get("case_id") != case_id or observation["metadata"] != record["simulation"]:
        raise RuntimeError(f"{case_id} 的重建观测与JSON审计信息不一致")
    if not (record["fitting"]["null_model"]["windows"][0]["solver"]["converged"]
            and record["fitting"]["full_model"]["windows"][0]["solver"]["converged"]):
        raise RuntimeError(f"{case_id} 的JSON求解状态不是全部收敛")

    with np.load(source / (case_id + ".npz")) as saved:
        required_arrays = {"truth", "null", "full", "vertices", "times", "active", "baseline"}
        if set(saved.files) != required_arrays:
            raise RuntimeError(f"{case_id} 的NPZ字段不完整或出现未知字段")
        truth = saved["truth"]
        null = saved["null"].astype(float)
        full = saved["full"].astype(float)
        if (not np.array_equal(saved["baseline"], observation["baseline"])
                or not np.array_equal(saved["active"], observation["active"])
                or not np.array_equal(truth, observation["truth"].astype(np.float32))
                or not np.array_equal(saved["vertices"], shared["vertices"])
                or not np.array_equal(saved["times"], shared["times"])):
            raise RuntimeError(f"{case_id} 的NPZ与重建观测或共享源空间不一致")
    if (null.shape != observation["truth"].shape or full.shape != null.shape
            or not np.isfinite(null).all() or not np.isfinite(full).all()):
        raise RuntimeError(f"{case_id} 的冻结full/null形状错误或含非有限值")

    baseline = np.asarray(observation["baseline"], bool)
    active = np.asarray(observation["active_windows"][0], bool)
    confirmation = np.asarray(observation["confirmation"], float)
    gain = np.asarray(observation["gain"], float)
    basis, _ = _smooth_temporal_basis(confirmation, baseline, active)
    centered = confirmation - confirmation[:, baseline].mean(axis=1, keepdims=True)
    source_modes = null @ basis.T, full @ basis.T
    basis_sum = basis[:, active].sum(axis=1)
    baseline_samples = int(baseline.sum())
    mean_correction = float(np.sum(basis_sum ** 2) / baseline_samples)
    modality_sizes = list(map(int, observation["metadata"]["retained_channels"]))
    archived = record["evidence"]
    archived_gains = np.asarray(archived["modality_noise_scores"], float)
    archived_excess = np.asarray(archived["modality_response_excess_ratios"], float)
    if (len(modality_sizes) != 2 or sum(modality_sizes) != len(confirmation)
            or archived_gains.shape != (2,) or archived_excess.shape != (2,)):
        raise RuntimeError(f"{case_id} 必须包含EEG、MEG两个完整模态")

    gains, excess_ratios, loss_improvements = [], [], []
    diagonal_variances, full_covariance_variances = [], []
    diagonal_z, full_covariance_z = [], []
    start = 0
    for modality_number, size in enumerate(modality_sizes):
        block = slice(start, start + size)
        response = centered[block] @ basis.T
        block_gain = gain[block]
        null_prediction = block_gain @ source_modes[0]
        full_prediction = block_gain @ source_modes[1]
        delta = full_prediction - null_prediction
        null_loss = float(np.sum((response - null_prediction) ** 2))
        full_loss = float(np.sum((response - full_prediction) ** 2))
        loss_improvement = null_loss - full_loss
        baseline_block = centered[block, baseline]
        channel_variance = np.sum(baseline_block ** 2, axis=1) / (baseline_samples - 1)
        noise_energy = float(np.sum(channel_variance) * (len(basis) + mean_correction))
        response_energy = float(np.sum(response ** 2))
        gain_score = loss_improvement / noise_energy
        response_excess = max(response_energy / noise_energy - 1., 0.)

        archived_values = np.array([
            archived["modality_null_losses"][modality_number],
            archived["modality_full_losses"][modality_number],
            archived["modality_expected_response_noise_energy"][modality_number],
            archived["modality_response_energy"][modality_number],
            archived_gains[modality_number],
            archived_excess[modality_number],
        ], float)
        reproduced_values = np.array([
            null_loss, full_loss, noise_energy, response_energy,
            gain_score, response_excess,
        ])
        if not np.allclose(reproduced_values, archived_values, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"{case_id} 的冻结full/null不能复现JSON中的第{modality_number + 1}模态证据")

        mean_direction = delta @ basis_sum
        diagonal_variance = 4. * float(
            np.sum(channel_variance[:, None] * delta ** 2)
            + np.sum(channel_variance * mean_direction ** 2) / baseline_samples)
        covariance = baseline_block @ baseline_block.T / (baseline_samples - 1)
        full_covariance_variance = 4. * float(
            np.sum(delta * (covariance @ delta))
            + mean_direction @ (covariance @ mean_direction) / baseline_samples)
        if (not np.isfinite(diagonal_variance) or diagonal_variance <= 0
                or not np.isfinite(full_covariance_variance) or full_covariance_variance <= 0):
            raise RuntimeError(f"{case_id} 的studentized方差不是正有限数")

        gains.append(gain_score)
        excess_ratios.append(response_excess)
        loss_improvements.append(loss_improvement)
        diagonal_variances.append(diagonal_variance)
        full_covariance_variances.append(full_covariance_variance)
        diagonal_z.append(loss_improvement / np.sqrt(diagonal_variance))
        full_covariance_z.append(loss_improvement / np.sqrt(full_covariance_variance))
        start += size

    gains = np.asarray(gains)
    excess_ratios = np.asarray(excess_ratios)
    positive_gains = np.maximum(gains, 0.)
    consensus = float(gains.min() + np.sqrt(positive_gains[0] * positive_gains[1]))
    blind_consensus = consensus / (1. + float(excess_ratios.max())) ** .25
    if not np.isclose(blind_consensus, archived["snr_blind_consensus_score"],
                      rtol=1e-5, atol=1e-7):
        raise RuntimeError(f"{case_id} 的SNR盲一致性分数不能复现")

    raw_fixed = .75 * gains[0] + .25 * gains[1]
    global_energy_fixed = raw_fixed / np.sqrt(1. + excess_ratios.max())
    per_modality_energy_fixed = (
        .75 * gains[0] / np.sqrt(1. + excess_ratios[0])
        + .25 * gains[1] / np.sqrt(1. + excess_ratios[1]))
    diagonal_studentized = .75 * diagonal_z[0] + .25 * diagonal_z[1]
    full_covariance_studentized = .75 * full_covariance_z[0] + .25 * full_covariance_z[1]
    rows.append({
        "case_id": case_id,
        "configuration_id": case["configuration_id"],
        "scenario": case["scenario"],
        "has_deep_true_posthoc": int(case.get("deep_index") is not None),
        "eeg_snr_db_posthoc_not_used": case["eeg_snr_db"],
        "meg_snr_db_posthoc_not_used": case["meg_snr_db"],
        "baseline_samples": baseline_samples,
        "eeg_channels": modality_sizes[0],
        "meg_channels": modality_sizes[1],
        "eeg_gain": gains[0], "meg_gain": gains[1],
        "eeg_response_excess_ratio": excess_ratios[0],
        "meg_response_excess_ratio": excess_ratios[1],
        "eeg_loss_improvement": loss_improvements[0],
        "meg_loss_improvement": loss_improvements[1],
        "eeg_diagonal_variance": diagonal_variances[0],
        "meg_diagonal_variance": diagonal_variances[1],
        "eeg_full_covariance_variance": full_covariance_variances[0],
        "meg_full_covariance_variance": full_covariance_variances[1],
        "eeg_diagonal_studentized": diagonal_z[0],
        "meg_diagonal_studentized": diagonal_z[1],
        "eeg_full_covariance_studentized": full_covariance_z[0],
        "meg_full_covariance_studentized": full_covariance_z[1],
        "raw_fixed_3_to_1": raw_fixed,
        "global_energy_normalized_fixed_3_to_1": global_energy_fixed,
        "per_modality_energy_normalized_fixed_3_to_1": per_modality_energy_fixed,
        "snr_blind_consensus": blind_consensus,
        "diagonal_studentized_fixed_3_to_1": diagonal_studentized,
        "full_covariance_studentized_fixed_3_to_1": full_covariance_studentized,
    })


# %% 4. 用An_auc汇总；标签和SNR只在所有盲分数算完后用于开发评估。
method_specs = [
    ("raw_fixed_3_to_1", "Raw fixed 3:1",
     "0.75*g_EEG + 0.25*g_MEG"),
    ("global_energy_normalized_fixed_3_to_1", "Global-energy fixed 3:1",
     "(0.75*g_EEG + 0.25*g_MEG) / sqrt(1 + max(r_EEG,r_MEG))"),
    ("per_modality_energy_normalized_fixed_3_to_1", "Per-modality-energy fixed 3:1",
     "0.75*g_EEG/sqrt(1+r_EEG) + 0.25*g_MEG/sqrt(1+r_MEG)"),
    ("snr_blind_consensus", "SNR-blind consensus",
     "[min(g_EEG,g_MEG)+sqrt(max(g_EEG,0)*max(g_MEG,0))] / [1+max(r_EEG,r_MEG)]^0.25"),
    ("diagonal_studentized_fixed_3_to_1", "Diagonal studentized 3:1",
     "0.75*D_EEG/sqrt(Vdiag_EEG) + 0.25*D_MEG/sqrt(Vdiag_MEG)"),
    ("full_covariance_studentized_fixed_3_to_1", "Full-covariance studentized 3:1 (ablation)",
     "0.75*D_EEG/sqrt(Vfull_EEG) + 0.25*D_MEG/sqrt(Vfull_MEG)"),
]
labels = np.asarray([row["has_deep_true_posthoc"] for row in rows], int)
if labels.shape != (15,) or np.sum(labels == 0) != 6 or np.sum(labels == 1) != 9:
    raise RuntimeError("开发面板应包含6例纯表层H0和9例含深源H1")
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
method_summary, snr_summary = [], []
for method, display_name, formula in method_specs:
    scores = np.asarray([row[method] for row in rows], float)
    h0, h1 = scores[labels == 0], scores[labels == 1]
    method_summary.append({
        "method": method, "display_name": display_name, "formula": formula,
        "An_auc": float(An_auc(np.c_[labels, scores])),
        "H0_count": int(len(h0)), "H1_count": int(len(h1)),
        "H0_median": float(np.median(h0)), "H1_median": float(np.median(h1)),
        "H0_max": float(h0.max()), "H1_min": float(h1.min()),
        "separation_margin_H1_min_minus_H0_max": float(h1.min() - h0.max()),
        "H0_positive": int(np.sum(h0 > 0)), "H1_positive": int(np.sum(h1 > 0)),
    })
    for eeg_snr, meg_snr in snr_pairs:
        selected = np.asarray([
            row["eeg_snr_db_posthoc_not_used"] == eeg_snr
            and row["meg_snr_db_posthoc_not_used"] == meg_snr
            for row in rows
        ])
        pair_labels, pair_scores = labels[selected], scores[selected]
        pair_h0, pair_h1 = pair_scores[pair_labels == 0], pair_scores[pair_labels == 1]
        if len(pair_h0) != 2 or len(pair_h1) != 3:
            raise RuntimeError("每个SNR组合应有2例H0和3例H1")
        snr_summary.append({
            "method": method, "eeg_snr_db_posthoc": eeg_snr,
            "meg_snr_db_posthoc": meg_snr,
            "An_auc": float(An_auc(np.c_[pair_labels, pair_scores])),
            "H0_count": int(len(pair_h0)), "H1_count": int(len(pair_h1)),
            "H0_max": float(pair_h0.max()), "H1_min": float(pair_h1.min()),
            "separation_margin_H1_min_minus_H0_max": float(pair_h1.min() - pair_h0.max()),
        })

singular_modality_cases = int(sum(
    row[modality + "_channels"] > row["baseline_samples"] - 1
    for row in rows for modality in ("eeg", "meg")))
summary = {
    "complete": True,
    "phase": "development",
    "formal_claim_allowed": False,
    "validation_results_read": False,
    "inverse_refitted": False,
    "truth_or_snr_used_by_score": False,
    "case_count": len(rows),
    "H0_count": int(np.sum(labels == 0)),
    "H1_count": int(np.sum(labels == 1)),
    "selected_method": None,
    "selection_status": "comparison_only; no score selected before fresh calibration",
    "methods": method_summary,
    "snr_summary_posthoc": snr_summary,
    "full_covariance_status": (
        "ablation_only: the baseline-centered sample covariance has rank at most B-1; "
        "when channels exceed B-1 it is singular and must not be treated as a calibrated Gaussian Z test"),
    "singular_full_covariance_modality_cases": singular_modality_cases,
    "input_sha256": frozen_input_sha256,
    "verified_code_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in direct_dependencies
    },
    "analysis_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
}


# %% 5. 先在内存生成2x3点图，再一次性创建开发输出目录。
palette = {0: "#0072B2", 1: "#D55E00"}
figure, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
for axis, (method, display_name, _formula), statistics in zip(
        axes.ravel(), method_specs, method_summary):
    for pair_number, pair in enumerate(snr_pairs):
        for label, offset in ((0, -.12), (1, .12)):
            chosen = [row for row in rows
                      if row["has_deep_true_posthoc"] == label
                      and (row["eeg_snr_db_posthoc_not_used"],
                           row["meg_snr_db_posthoc_not_used"]) == pair]
            x = pair_number + offset + np.linspace(-.045, .045, len(chosen))
            axis.scatter(x, [row[method] for row in chosen], s=34, alpha=.88,
                         color=palette[label], edgecolor="white", linewidth=.45,
                         label=("surface-only H0" if label == 0 else "deep-present H1")
                         if pair_number == 0 else None)
    axis.axhline(0., color="#777777", linewidth=.8, linestyle="--")
    axis.set(xticks=range(3), xticklabels=["-10/-10", "-10/+20", "+20/-10"],
             xlabel="EEG/MEG SNR (dB; post-hoc)", ylabel="blind score",
             title=f"{display_name}\nAn_auc={statistics['An_auc']:.3f}")
axes[0, 0].legend(frameon=False, fontsize=8)
figure.suptitle("Equal-weight frozen fits: six development-only gate scores", fontsize=14)

lines = [
    "# hierarchical-v2 等权门控六种评分开发比较", "",
    "只读取15例 development 冻结拟合；没有读取 validation，也没有重新运行逆解。标签和SNR只在盲分数全部算完后用于开发评估。", "",
    "## 六种评分", "",
]
for number, (_method, display_name, formula) in enumerate(method_specs, 1):
    lines.append(f"{number}. **{display_name}**：`{formula}`")
lines += ["", "两种能量归一化分别比较，不预先选择。`r` 是确认观测活动响应相对基线噪声期望的非负超额比。", "",
          "## 总体结果", "",
          "| 方法 | An_auc | H0最大值 | H1最小值 | 分离间隔 | H0正分 | H1正分 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
for item in method_summary:
    lines.append(
        f"| {item['display_name']} | {item['An_auc']:.4f} | {item['H0_max']:.6g} | "
        f"{item['H1_min']:.6g} | {item['separation_margin_H1_min_minus_H0_max']:.6g} | "
        f"{item['H0_positive']}/{item['H0_count']} | {item['H1_positive']}/{item['H1_count']} |")
lines += ["", "## 分SNR结果", "",
          "| 方法 | EEG/MEG SNR | An_auc | H0最大值 | H1最小值 | 分离间隔 |",
          "|---|---|---:|---:|---:|---:|"]
display_lookup = {item[0]: item[1] for item in method_specs}
for item in snr_summary:
    lines.append(
        f"| {display_lookup[item['method']]} | {item['eeg_snr_db_posthoc']:+d}/"
        f"{item['meg_snr_db_posthoc']:+d} | {item['An_auc']:.4f} | {item['H0_max']:.6g} | "
        f"{item['H1_min']:.6g} | {item['separation_margin_H1_min_minus_H0_max']:.6g} |")
lines += [
    "", "## 限制", "",
    "- `selected_method = null`：这只是15例开发比较，不能代替新校准。",
    "- full-covariance studentized 只作消融。基线中心化后样本协方差秩最多为 `B-1`；通道数大于 `B-1` 时必然奇异。",
    "- studentized 值是经验排序分数，不是Gaussian Z检验，也不能直接解释成p值。", "",
]

if frozen_input_sha256 != {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in frozen_input_paths}:
    raise RuntimeError("评分期间冻结输入发生变化，拒绝写出结果")
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix=output.name + ".tmp-", dir=output.parent) as temporary_name:
    temporary = Path(temporary_name)
    with (temporary / "scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (temporary / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    figure.savefig(temporary / "score_comparison.png", dpi=220, facecolor="white")
    plt.close(figure)
    (temporary / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    temporary.rename(output)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
