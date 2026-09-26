"""# %% 只比较三组既有开发分数；不读取validation，也不运行逆解。"""

# %% 1. 固定输入、输出和依赖。
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
from metrics.user_metrics.An_auc import An_auc

development = root / "results/erp_whole_head/adaptive_v6/development_diagnosis"
current_dir = development / "hierarchical_v2_equal_weight_score_comparison"
retired_dir = development / "hierarchical_v2_snr_blind_gate"
hard_dir = development / "hierarchical_v2_equal_weight_hard_cases"
output = development / "hierarchical_v2_blind_reliability_fusion"
if output.exists():
    raise FileExistsError(f"不覆盖旧开发结果：{output}")

input_paths = [
    current_dir / "scores.csv", current_dir / "summary.json",
    retired_dir / "scores.csv", retired_dir / "summary.json",
    hard_dir / "rows.csv", hard_dir / "summary.json",
]
if any(not path.is_file() for path in input_paths):
    raise FileNotFoundError("69号、retired113或hard6开发输入不完整")
if any("validation" in str(path).lower() for path in input_paths):
    raise RuntimeError("本脚本禁止读取validation路径")

script_path = Path(__file__).resolve()
code_paths = [script_path, root / "metrics/user_metrics/An_auc.py",
              root / "metrics/user_metrics/An_roc.py"]
input_sha256 = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in input_paths}
code_sha256 = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in code_paths}


# %% 2. 读取并严格核对三个已查看的开发集合。
with input_paths[0].open(encoding="utf-8-sig", newline="") as stream:
    current_rows = list(csv.DictReader(stream))
current_summary = json.loads(input_paths[1].read_text(encoding="utf-8"))
with input_paths[2].open(encoding="utf-8-sig", newline="") as stream:
    retired_rows = list(csv.DictReader(stream))
retired_summary = json.loads(input_paths[3].read_text(encoding="utf-8"))
with input_paths[4].open(encoding="utf-8-sig", newline="") as stream:
    hard_rows = list(csv.DictReader(stream))
hard_summary = json.loads(input_paths[5].read_text(encoding="utf-8"))

if (not current_summary.get("complete") or current_summary.get("phase") != "development"
        or current_summary.get("validation_results_read") is not False
        or len(current_rows) != 15 or current_summary.get("H0_count") != 6
        or current_summary.get("H1_count") != 9):
    raise RuntimeError("69号current15不是完整、未读validation的6 H0/9 H1开发结果")
if (not retired_summary.get("complete") or retired_summary.get("phase") != "development"
        or retired_summary.get("validation_results_read") is not False
        or len(retired_rows) != 113
        or retired_summary.get("retired_v1_gate_case_count") != 57
        or retired_summary.get("retired_v1_deep_case_count_valid") != 56):
    raise RuntimeError("retired113不是完整、未读validation的57 H0/56 H1开发结果")
if (not hard_summary.get("complete") or hard_summary.get("phase") != "development_diagnosis"
        or hard_summary.get("validation_read") is not False or len(hard_rows) != 6
        or hard_summary.get("h0_count") != 3 or hard_summary.get("h1_count") != 3):
    raise RuntimeError("hard6不是完整、未读validation的3 H0/3 H1诊断结果")

required_current = {"case_id", "has_deep_true_posthoc", "eeg_snr_db_posthoc_not_used",
                    "meg_snr_db_posthoc_not_used", "eeg_gain", "meg_gain",
                    "eeg_response_excess_ratio", "meg_response_excess_ratio"}
required_retired = {"case_id", "label_posthoc", "eeg_snr_db_posthoc_not_used",
                    "meg_snr_db_posthoc_not_used", "eeg_gain", "meg_gain",
                    "eeg_response_excess", "meg_response_excess"}
required_hard = {"case_id", "posthoc_role", "eeg_snr_db_posthoc", "meg_snr_db_posthoc",
                 "equal_eeg_gain", "equal_meg_gain", "eeg_response_excess",
                 "meg_response_excess"}
if (not required_current.issubset(current_rows[0])
        or not required_retired.issubset(retired_rows[0])
        or not required_hard.issubset(hard_rows[0])):
    raise RuntimeError("开发输入缺少盲融合所需字段")


# %% 3. 明列全部已查看候选；不根据结果继续扩展网格。
candidate_specs = []
for exponent in (.5, .625, .75, .875, 1.):
    candidate_specs.append({
        "candidate": f"per_modality_fixed_3_to_1_e{str(exponent).replace('.', '')}",
        "family": "per_modality_exponent", "exponent": exponent,
        "eeg_weight": .75, "balance_power": None,
        "formula": f"0.75*gE/(1+qE)^{exponent:g} + 0.25*gM/(1+qM)^{exponent:g}",
    })
for weight in (.5, .6, 2 / 3, .75):
    candidate_specs.append({
        "candidate": f"sqrt_weight_eeg_{weight:.6g}".replace(".", "p"),
        "family": "sqrt_weight", "exponent": .5,
        "eeg_weight": weight, "balance_power": None,
        "formula": f"{weight:.6g}*hE + {1-weight:.6g}*hM; hm=gm/sqrt(1+qm)",
    })
for power in (.5, 1., 2.):
    candidate_specs.append({
        "candidate": f"adaptive_balance_p{power:g}".replace(".", "p"),
        "family": "adaptive_balance", "exponent": .5,
        "eeg_weight": None, "balance_power": power,
        "formula": f"wE*hE+(1-wE)*hM; b=min(1+q)/max(1+q); wE=0.5+0.25*b^{power:g}",
    })
candidate_specs += [
    {"candidate": "reference_global_sqrt_fixed_3_to_1", "family": "reference_global",
     "exponent": .5, "eeg_weight": .75, "balance_power": None,
     "formula": "(0.75*gE+0.25*gM)/sqrt(1+max(qE,qM))"},
    {"candidate": "reference_snr_blind_consensus", "family": "reference_consensus",
     "exponent": .25, "eeg_weight": None, "balance_power": None,
     "formula": "[min(gE,gM)+sqrt(max(gE,0)*max(gM,0))]/[1+max(qE,qM)]^0.25"},
]
if len(candidate_specs) != 14 or len({item["candidate"] for item in candidate_specs}) != 14:
    raise RuntimeError("候选网格必须固定为14个具名条目")

datasets = [
    ("current15", current_rows, "has_deep_true_posthoc",
     "eeg_snr_db_posthoc_not_used", "meg_snr_db_posthoc_not_used",
     "eeg_gain", "meg_gain", "eeg_response_excess_ratio", "meg_response_excess_ratio"),
    ("retired113", retired_rows, "label_posthoc",
     "eeg_snr_db_posthoc_not_used", "meg_snr_db_posthoc_not_used",
     "eeg_gain", "meg_gain", "eeg_response_excess", "meg_response_excess"),
    ("hard6", hard_rows, "posthoc_role",
     "eeg_snr_db_posthoc", "meg_snr_db_posthoc",
     "equal_eeg_gain", "equal_meg_gain", "eeg_response_excess", "meg_response_excess"),
]
case_scores = []
for dataset, source_rows, label_key, eeg_snr_key, meg_snr_key, eeg_gain_key, meg_gain_key, eeg_q_key, meg_q_key in datasets:
    for source_row in source_rows:
        label = (int(source_row[label_key]) if dataset != "hard6"
                 else int(source_row[label_key] == "pure_deep_H1"))
        eeg_snr, meg_snr = int(source_row[eeg_snr_key]), int(source_row[meg_snr_key])
        eeg_gain, meg_gain = float(source_row[eeg_gain_key]), float(source_row[meg_gain_key])
        eeg_q, meg_q = float(source_row[eeg_q_key]), float(source_row[meg_q_key])
        if (label not in (0, 1) or eeg_q < 0 or meg_q < 0
                or not np.isfinite([eeg_gain, meg_gain, eeg_q, meg_q]).all()):
            raise RuntimeError(f"{dataset}/{source_row['case_id']} 含非法盲评分输入")
        eeg_h = eeg_gain / np.sqrt(1. + eeg_q)
        meg_h = meg_gain / np.sqrt(1. + meg_q)
        balance = min(1. + eeg_q, 1. + meg_q) / max(1. + eeg_q, 1. + meg_q)
        for spec in candidate_specs:
            if spec["family"] == "per_modality_exponent":
                score = (.75 * eeg_gain / (1. + eeg_q) ** spec["exponent"]
                         + .25 * meg_gain / (1. + meg_q) ** spec["exponent"])
            elif spec["family"] == "sqrt_weight":
                score = spec["eeg_weight"] * eeg_h + (1. - spec["eeg_weight"]) * meg_h
            elif spec["family"] == "adaptive_balance":
                eeg_weight = .5 + .25 * balance ** spec["balance_power"]
                score = eeg_weight * eeg_h + (1. - eeg_weight) * meg_h
            elif spec["family"] == "reference_global":
                score = (.75 * eeg_gain + .25 * meg_gain) / np.sqrt(1. + max(eeg_q, meg_q))
            else:
                numerator = min(eeg_gain, meg_gain) + np.sqrt(max(eeg_gain, 0.) * max(meg_gain, 0.))
                score = numerator / (1. + max(eeg_q, meg_q)) ** .25
            if not np.isfinite(score):
                raise RuntimeError(f"{dataset}/{source_row['case_id']}/{spec['candidate']} 分数非有限")
            case_scores.append({
                "dataset": dataset, "case_id": source_row["case_id"], "label_posthoc": label,
                "eeg_snr_db_posthoc": eeg_snr, "meg_snr_db_posthoc": meg_snr,
                "candidate": spec["candidate"], "family": spec["family"], "score": float(score),
            })


# %% 4. 固定验收规则并汇总，不用真值或SNR计算任何单例分数。
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
candidate_scores = []
for spec in candidate_specs:
    current = [row for row in case_scores
               if row["dataset"] == "current15" and row["candidate"] == spec["candidate"]]
    retired = [row for row in case_scores
               if row["dataset"] == "retired113" and row["candidate"] == spec["candidate"]]
    hard = [row for row in case_scores
            if row["dataset"] == "hard6" and row["candidate"] == spec["candidate"]]
    current_labels = np.asarray([row["label_posthoc"] for row in current], int)
    current_values = np.asarray([row["score"] for row in current], float)
    retired_labels = np.asarray([row["label_posthoc"] for row in retired], int)
    retired_values = np.asarray([row["score"] for row in retired], float)
    hard_labels = np.asarray([row["label_posthoc"] for row in hard], int)
    hard_values = np.asarray([row["score"] for row in hard], float)
    summary_row = {
        "candidate": spec["candidate"], "family": spec["family"], "formula": spec["formula"],
        "exponent": "" if spec["exponent"] is None else spec["exponent"],
        "eeg_weight": "" if spec["eeg_weight"] is None else spec["eeg_weight"],
        "balance_power": "" if spec["balance_power"] is None else spec["balance_power"],
        "current_all_An_auc": float(An_auc(np.c_[current_labels, current_values])),
        "current_all_H1_positive": int(np.sum(current_values[current_labels == 1] > 0)),
        "current_all_H1_count": int(np.sum(current_labels == 1)),
        "retired113_An_auc": float(An_auc(np.c_[retired_labels, retired_values])),
        "hard6_An_auc": float(An_auc(np.c_[hard_labels, hard_values])),
        "hard6_H0_max": float(hard_values[hard_labels == 0].max()),
        "hard6_H1_min": float(hard_values[hard_labels == 1].min()),
    }
    summary_row["hard6_separation_margin"] = summary_row["hard6_H1_min"] - summary_row["hard6_H0_max"]
    for eeg_snr, meg_snr in snr_pairs:
        selected = np.asarray([(row["eeg_snr_db_posthoc"], row["meg_snr_db_posthoc"])
                               == (eeg_snr, meg_snr) for row in current])
        pair_labels, pair_values = current_labels[selected], current_values[selected]
        if np.sum(pair_labels == 0) != 2 or np.sum(pair_labels == 1) != 3:
            raise RuntimeError("current15每个SNR格必须是2 H0/3 H1")
        key = f"current_{eeg_snr:+d}_{meg_snr:+d}".replace("+", "p").replace("-", "m")
        summary_row[key + "_An_auc"] = float(An_auc(np.c_[pair_labels, pair_values]))
        summary_row[key + "_H1_positive"] = int(np.sum(pair_values[pair_labels == 1] > 0))
        summary_row[key + "_H1_count"] = int(np.sum(pair_labels == 1))
    cell_aucs = [summary_row[key + "_An_auc"] for key in
                 ("current_m10_m10", "current_m10_p20", "current_p20_m10")]
    summary_row["passes_frozen_rule"] = int(
        summary_row["current_all_An_auc"] >= .9 and min(cell_aucs) >= .9
        and summary_row["current_all_H1_positive"] == 9
        and summary_row["current_all_H1_count"] == 9
        and summary_row["retired113_An_auc"] >= .98
        and summary_row["hard6_An_auc"] == 1.
        and summary_row["hard6_separation_margin"] > 0.)
    candidate_scores.append(summary_row)

preferred_candidate = "adaptive_balance_p1"
passing_candidates = [row["candidate"] for row in candidate_scores if row["passes_frozen_rule"]]
selected_candidate = preferred_candidate if preferred_candidate in passing_candidates else None
for row in candidate_scores:
    row["selected"] = int(row["candidate"] == selected_candidate)


# %% 5. 形成图、报告和可追溯汇总。
summary = {
    "complete": True, "phase": "development", "formal_claim_allowed": False,
    "validation_results_read": False, "inverse_refitted": False,
    "truth_or_snr_used_by_score": False,
    "candidate_count": len(candidate_specs), "current_case_count": 15,
    "retired_case_count": 113, "hard_case_count": 6,
    "frozen_acceptance_rule": (
        "current overall and each of three SNR-cell An_auc >= 0.9; current H1 positive 9/9; "
        "retired113 An_auc >= 0.98; hard6 An_auc == 1 and H1_min-H0_max > 0"),
    "passing_candidates": passing_candidates,
    "selected_candidate": selected_candidate,
    "selection_reason": ("adaptive p=1 uses the observed reliability balance b directly, without another power"
                         if selected_candidate else
                         "predeclared adaptive p=1 did not pass every frozen development criterion; no candidate selected"),
    "all_candidates": candidate_specs,
    "candidate_metrics": candidate_scores,
    "input_sha256": input_sha256, "code_sha256": code_sha256,
}

display_candidate = selected_candidate or preferred_candidate
point_rows = [row for row in case_scores
              if row["dataset"] == "current15" and row["candidate"] == display_candidate]
palette = {0: "#0072B2", 1: "#D55E00"}
figure_points, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
for pair_number, pair in enumerate(snr_pairs):
    for label, offset in ((0, -.12), (1, .12)):
        chosen = [row for row in point_rows if row["label_posthoc"] == label
                  and (row["eeg_snr_db_posthoc"], row["meg_snr_db_posthoc"]) == pair]
        x = pair_number + offset + np.linspace(-.04, .04, len(chosen))
        axis.scatter(x, [row["score"] for row in chosen], s=48, color=palette[label],
                     edgecolor="white", linewidth=.5,
                     label=("surface-only H0" if label == 0 else "deep-present H1")
                     if pair_number == 0 else None)
axis.axhline(0., color="#777777", linestyle="--", linewidth=.8)
axis.set(xticks=range(3), xticklabels=["-10/-10", "-10/+20", "+20/-10"],
         xlabel="EEG/MEG SNR (dB; post-hoc)", ylabel="blind reliability score",
         title=f"Current15: {display_candidate}")
axis.legend(frameon=False)

figure_lines, axis = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
x = np.arange(len(candidate_scores))
series = [("current_all_An_auc", "Current15 overall"),
          ("current_m10_m10_An_auc", "Current -10/-10"),
          ("current_m10_p20_An_auc", "Current -10/+20"),
          ("current_p20_m10_An_auc", "Current +20/-10"),
          ("retired113_An_auc", "Retired113"), ("hard6_An_auc", "Hard6")]
for key, label in series:
    axis.plot(x, [row[key] for row in candidate_scores], marker="o", linewidth=1.35,
              markersize=4, label=label)
axis.axhline(.9, color="#888888", linestyle="--", linewidth=.8)
axis.axhline(.98, color="#444444", linestyle=":", linewidth=.8)
axis.set(xticks=x, xticklabels=[row["candidate"] for row in candidate_scores],
         ylabel="An_auc", ylim=(0., 1.03), title="Frozen development reliability criteria")
axis.tick_params(axis="x", rotation=55, labelsize=8)
axis.legend(ncol=3, frameon=False, fontsize=8)

report = [
    "# v2盲可靠性融合开发", "",
    "只读取69号current15、retired113和hard6开发分数；没有读取validation，也没有运行逆解。", "",
    "## 冻结选择规则", "", summary["frozen_acceptance_rule"], "",
    f"通过候选：{', '.join(passing_candidates) if passing_candidates else '无'}。",
    f"选择：`{selected_candidate}`。" if selected_candidate else "预指定的adaptive p=1未全通过，因此不选择任何候选。",
    "选择理由固定为：`b`本身就是可靠性平衡量，`p=1`直接使用它，不再添加额外幂。", "",
    "## 候选结果", "",
    "| 候选 | current | 三格最小 | H1正分 | retired113 | hard6 | hard间隔 | 通过 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in candidate_scores:
    cell_min = min(row[key] for key in
                   ("current_m10_m10_An_auc", "current_m10_p20_An_auc", "current_p20_m10_An_auc"))
    report.append(f"| {row['candidate']} | {row['current_all_An_auc']:.4f} | {cell_min:.4f} | "
                  f"{row['current_all_H1_positive']}/9 | {row['retired113_An_auc']:.4f} | "
                  f"{row['hard6_An_auc']:.4f} | {row['hard6_separation_margin']:.6g} | "
                  f"{row['passes_frozen_rule']} |")
report += ["", "## 限制", "",
           "三组数据均已参与开发判断；这些结果只用于冻结下一轮新校准方案，不能作为独立验证或论文性能声明。", ""]


# %% 6. 输入和代码前后哈希一致后，临时目录完整写出再原子提交。
if input_sha256 != {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in input_paths}:
    raise RuntimeError("分析期间开发输入发生变化，拒绝写出")
if code_sha256 != {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in code_paths}:
    raise RuntimeError("分析期间脚本或AUC实现发生变化，拒绝写出")
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix=output.name + ".tmp-", dir=output.parent) as temporary_name:
    temporary = Path(temporary_name)
    with (temporary / "candidate_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_scores[0].keys())
        writer.writeheader()
        writer.writerows(candidate_scores)
    with (temporary / "case_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_scores[0].keys())
        writer.writeheader()
        writer.writerows(case_scores)
    (temporary / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure_points.savefig(temporary / "current15_candidate_points.png", dpi=220, facecolor="white")
    figure_lines.savefig(temporary / "candidate_auc_lines.png", dpi=220, facecolor="white")
    plt.close(figure_points)
    plt.close(figure_lines)
    (temporary / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    temporary.rename(output)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
