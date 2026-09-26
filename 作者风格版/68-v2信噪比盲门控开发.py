"""# %% 用已经退役的 hierarchical-v1 校准结果开发 v2 门控；绝不读取 validation 结果。"""

# %% 1. 固定只读输入。v1 的二级校准无效，因此两批已消费病例只能降级为开发资料。
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import _smooth_temporal_basis
from candidates.oaster_predictive import snr_blind_consensus_score
from metrics.user_metrics.An_auc import An_auc
import run_erp_whole_head_matrix as original

protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
gate_dir = root / "results/erp_whole_head/adaptive_v6/formal_hierarchical_v1_gate_calibration"
deep_dir = root / "results/erp_whole_head/adaptive_v6/formal_hierarchical_v1_surface_calibration"
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/hierarchical_v2_snr_blind_gate"
if output.exists():
    raise FileExistsError(f"不覆盖旧开发结果：{output}")
gate_completion = json.loads((gate_dir / "completion.json").read_text(encoding="utf-8"))
deep_completion = json.loads((deep_dir / "completion.json").read_text(encoding="utf-8"))
if not gate_completion.get("valid") or deep_completion.get("valid") is not False:
    raise ValueError("这里只接受一级完成、二级已经判无效的 hierarchical-v1 退役链")
if (root / "results/erp_whole_head/adaptive_v6/formal_hierarchical_v1_validation").exists() or \
        (protocol_dir / "hierarchical_v1_validation_manifest_consumed.json").exists():
    raise RuntimeError("hierarchical-v1 validation 已经被打开，不能再做本开发诊断")

sources = [
    dict(label=0, role="pure_surface_H0", directory=gate_dir,
         manifest=protocol_dir / "hierarchical_v1_gate_calibration_manifest.json",
         seed_root=2026093001),
    dict(label=1, role="pure_deep_H1", directory=deep_dir,
         manifest=protocol_dir / "hierarchical_v1_surface_calibration_manifest.json",
         seed_root=2026093003),
]
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
output.mkdir(parents=True)
started = time.perf_counter()


# %% 2. 只从 confirmation 数据估计活动/基线能量；真实 SNR 和真源位置不进入分数。
rows = []
failed_input_cases = []
for source in sources:
    cases = {
        case["case_id"]: case
        for case in json.loads(source["manifest"].read_text(encoding="utf-8"))
    }
    evidence_rows = list(csv.DictReader(
        (source["directory"] / "evidence.csv").open(encoding="utf-8-sig")))
    for evidence in evidence_rows:
        if evidence["status"] != "ok":
            failed_input_cases.append(evidence["case_id"])
            continue
        case = cases[evidence["case_id"]]
        observation = prepare_trial_covariance_case(
            shared, case, seed_root=source["seed_root"])
        baseline = np.asarray(observation["baseline"], bool)
        active = np.asarray(observation["active_windows"][0], bool)
        confirmation = np.asarray(observation["confirmation"], float)
        basis, _ = _smooth_temporal_basis(confirmation, baseline, active)
        centered = confirmation - confirmation[:, baseline].mean(axis=1, keepdims=True)
        mean_correction = float(
            np.sum(basis[:, active].sum(axis=1) ** 2) / baseline.sum())
        response_excess = []
        start = 0
        for size in map(int, observation["metadata"]["retained_channels"]):
            block = slice(start, start + size)
            response_energy = float(np.sum((centered[block] @ basis.T) ** 2))
            noise = (float(np.sum(centered[block, baseline] ** 2)
                           / (baseline.sum() - 1))
                     * (len(basis) + mean_correction))
            response_excess.append(max(response_energy / noise - 1., 0.))
            start += size
        modality_scores = np.array([
            float(evidence["gate_eeg_score"]),
            float(evidence["gate_meg_score"]),
        ])
        positive = np.maximum(modality_scores, 0.)
        consensus = float(
            modality_scores.min() + np.sqrt(positive[0]) * np.sqrt(positive[1]))
        blind_score = snr_blind_consensus_score(modality_scores, response_excess)
        rows.append({
            "case_id": case["case_id"], "development_role": source["role"],
            "label_posthoc": source["label"],
            "eeg_snr_db_posthoc_not_used": case["eeg_snr_db"],
            "meg_snr_db_posthoc_not_used": case["meg_snr_db"],
            "old_fixed_3_to_1_score": float(evidence["gate_score"]),
            "eeg_gain": modality_scores[0], "meg_gain": modality_scores[1],
            "eeg_response_excess": response_excess[0],
            "meg_response_excess": response_excess[1],
            "cross_modal_consensus": consensus,
            "snr_blind_consensus": blind_score,
        })
        if len(rows) % 10 == 0:
            print(f"已复算 {len(rows)}/113", flush=True)


# %% 3. An_auc、尾部阈值和归一化指数消融；这些仍是开发结果，不冒充新校准。
labels = np.asarray([row["label_posthoc"] for row in rows], int)
old_scores = np.asarray([row["old_fixed_3_to_1_score"] for row in rows], float)
consensus_scores = np.asarray([row["cross_modal_consensus"] for row in rows], float)
blind_scores = np.asarray([row["snr_blind_consensus"] for row in rows], float)
if labels.shape != (113,) or np.sum(labels == 0) != 57 or np.sum(labels == 1) != 56:
    raise RuntimeError("退役输入应有57个H0和56个严格完成的H1")
threshold = max(0., float(np.sort(blind_scores[labels == 0])[-2]))
decisions = blind_scores > threshold
snr_summary = []
for eeg_snr, meg_snr in [(-10, -10), (-10, 20), (20, -10)]:
    selected = np.array([
        row["eeg_snr_db_posthoc_not_used"] == eeg_snr
        and row["meg_snr_db_posthoc_not_used"] == meg_snr
        for row in rows
    ])
    snr_summary.append({
        "eeg_snr_db_posthoc": eeg_snr, "meg_snr_db_posthoc": meg_snr,
        "h0_above_threshold": int(np.sum(decisions[selected & (labels == 0)])),
        "h0_count": int(np.sum(selected & (labels == 0))),
        "h1_detected": int(np.sum(decisions[selected & (labels == 1)])),
        "h1_count": int(np.sum(selected & (labels == 1))),
    })

ablation = []
dominant_response_excess = np.asarray([
    max(row["eeg_response_excess"], row["meg_response_excess"])
    for row in rows
])
for exponent in [0., .125, .25, .375, .5, 1.]:
    scores = consensus_scores / (1. + dominant_response_excess) ** exponent
    null_scores = np.sort(scores[labels == 0])
    candidate_threshold = max(0., float(null_scores[-2]))
    ablation.append({
        "normalization_exponent": exponent,
        "An_auc": An_auc(np.c_[labels, scores]),
        "second_largest_H0_threshold": candidate_threshold,
        "H0_above_threshold": int(np.sum(scores[labels == 0] > candidate_threshold)),
        "H1_above_threshold": int(np.sum(scores[labels == 1] > candidate_threshold)),
    })

summary = {
    "complete": True, "phase": "development", "formal_claim_allowed": False,
    "retired_v1_gate_case_count": 57, "retired_v1_deep_case_count_valid": 56,
    "failed_input_cases": failed_input_cases,
    "validation_results_read": False, "true_snr_used_by_score": False,
    "old_fixed_3_to_1_An_auc": An_auc(np.c_[labels, old_scores]),
    "cross_modal_consensus_An_auc": An_auc(np.c_[labels, consensus_scores]),
    "snr_blind_consensus_An_auc": An_auc(np.c_[labels, blind_scores]),
    "development_threshold_second_largest_H0": threshold,
    "H0_above_development_threshold": int(np.sum(decisions[labels == 0])),
    "H1_above_development_threshold": int(np.sum(decisions[labels == 1])),
    "H1_valid_count": int(np.sum(labels == 1)),
    "snr_summary_posthoc": snr_summary,
    "selected_formula": (
        "[min(g_EEG,g_MEG)+sqrt(max(g_EEG,0)*max(g_MEG,0))] / "
        "[1+max(active_response_energy/noise_expectation-1,0)]^0.25"),
    "selection_status": "chosen on retired v1 calibration data; fresh calibration required",
    "solver_fix_for_v2": {
        "outer_iterations": 80, "old_outer_iterations": 40,
        "tolerance_unchanged": .001, "early_stopping_preserved": True,
        "case055_replayed_converged_outer_iteration": 42,
    },
    "wall_seconds": time.perf_counter() - started,
    "input_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for source in sources
        for path in (source["manifest"], source["directory"] / "evidence.csv")
    },
}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
for name, values in (("scores.csv", rows), ("exponent_ablation.csv", ablation)):
    with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=values[0].keys())
        writer.writeheader()
        writer.writerows(values)


# %% 4. 图片和短报告。
palette = {0: "#0072B2", 1: "#D55E00"}
figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
pair_order = [(-10, -10), (-10, 20), (20, -10)]
for pair_number, pair in enumerate(pair_order):
    for label, offset in ((0, -.10), (1, .10)):
        chosen = [
            row for row in rows
            if row["label_posthoc"] == label
            and (row["eeg_snr_db_posthoc_not_used"], row["meg_snr_db_posthoc_not_used"])
            == pair
        ]
        x = pair_number + offset + np.linspace(-.055, .055, len(chosen))
        axes[0].scatter(x, [row["snr_blind_consensus"] for row in chosen],
                        s=30, alpha=.82, color=palette[label], edgecolor="white",
                        linewidth=.4, label=("pure surface" if label == 0 else "pure deep")
                        if pair_number == 0 else None)
axes[0].axhline(threshold, color="#333333", linestyle="--", linewidth=1.3,
                label="development threshold")
axes[0].set(xticks=range(3), xticklabels=["−10/−10", "−10/+20", "+20/−10"],
            xlabel="EEG/MEG SNR (dB; post-hoc only)", ylabel="SNR-blind consensus",
            title="Retired cases used only for v2 development")
axes[0].set_yscale("symlog", linthresh=.002)
axes[0].legend(frameon=False, fontsize=8)
names = ["old 3:1", "consensus", "consensus +\nblind energy"]
aucs = [summary["old_fixed_3_to_1_An_auc"],
        summary["cross_modal_consensus_An_auc"],
        summary["snr_blind_consensus_An_auc"]]
axes[1].bar(names, aucs, color=["#999999", "#56B4E9", "#009E73"], width=.68)
axes[1].set(ylim=(.94, 1.), ylabel="An_auc", title="Gate discrimination")
for index, value in enumerate(aucs):
    axes[1].text(index, value + .001, f"{value:.3f}", ha="center", fontsize=9)
figure.savefig(output / "gate_development_comparison.png", dpi=220, facecolor="white")
plt.close(figure)

lines = [
    "# hierarchical-v2 信噪比盲门控开发", "",
    "v1 二级校准已判无效，所以已消费的一级/二级病例只作为开发资料；未读取也未运行 validation。", "",
    f"- 原固定3:1门控 An_auc：`{summary['old_fixed_3_to_1_An_auc']:.3f}`",
    f"- 跨模态一致性 An_auc：`{summary['cross_modal_consensus_An_auc']:.3f}`",
    f"- 加盲能量归一化 An_auc：`{summary['snr_blind_consensus_An_auc']:.3f}`",
    f"- 开发阈值：`{threshold:.8f}`；H0超阈 `1/57`；H1检出 `{summary['H1_above_development_threshold']}/56`。", "",
    "| EEG/MEG SNR | H0超阈 | H1检出 |", "|---|---:|---:|",
]
for item in snr_summary:
    lines.append(
        f"| {item['eeg_snr_db_posthoc']:+d}/{item['meg_snr_db_posthoc']:+d} | "
        f"{item['h0_above_threshold']}/{item['h0_count']} | "
        f"{item['h1_detected']}/{item['h1_count']} |")
lines += ["", "这只是开发筛选；必须用新位置/新噪声重新校准后，才能打开 untouched validation。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
