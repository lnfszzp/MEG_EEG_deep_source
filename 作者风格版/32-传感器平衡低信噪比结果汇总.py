"""#%% 合并分片开发结果，只做事实汇总，不读取校准集和验证集。"""

# %% 路径与完整性检查。
import csv
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import run_strict_oaster as archive

part_a = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases00_01"
part_b = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases02_04"
old_dir = root / "results/erp_whole_head/adaptive_v6/dev_trial_admm_mrf080_m10_m10"
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/sensor_balanced_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧汇总：{output}")

expected_ids = [f"erp-v6-development-{number:02d}-eeg-10-meg-10" for number in range(5)]
metadata_a = json.loads((part_a / "metadata.json").read_text(encoding="utf-8"))
metadata_b = json.loads((part_b / "metadata.json").read_text(encoding="utf-8"))
completion_b = json.loads((part_b / "completion.json").read_text(encoding="utf-8"))
assert metadata_a["phase"] == metadata_b["phase"] == "development"
assert metadata_a["case_ids"] == expected_ids
assert metadata_b["case_ids"] == expected_ids[2:]
assert metadata_a["covariance"] == metadata_b["covariance"] == "trial"
assert metadata_a["solver_settings"] == metadata_b["solver_settings"] == {"solver_kind": "admm", "mrf_strength": .8}
assert completion_b["complete"] and completion_b["case_count"] == 3

# %% 合并两个明确分片；第一个分片按计划在 case01 后停止，所以不用伪造 completion.json。
rows, evidence = [], []
for directory in (part_a, part_b):
    with (directory / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows.extend(csv.DictReader(stream))
    with (directory / "evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
        evidence.extend(csv.DictReader(stream))
assert sorted({row["case_id"] for row in rows}) == expected_ids
assert [row["case_id"] for row in evidence] == expected_ids
assert len(evidence) == 5

output.mkdir(parents=True)
archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
archive._atomic_csv(output / "evidence.csv", evidence, evidence[0].keys())

# %% 与未做传感器幅度平衡的同一 -10/-10 dB 开发结果逐例比较。
with (old_dir / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
    old_rows = list(csv.DictReader(stream))
old_h1 = {row["case_id"]: row for row in old_rows if row["method"] == "v6-full-ungated"}
new_h0 = {row["case_id"]: row for row in rows if row["method"] == "v6-surface-only"}
new_h1 = {row["case_id"]: row for row in rows if row["method"] == "v6-full-ungated"}
assert set(old_h1) == set(new_h0) == set(new_h1) == set(expected_ids)

case_summary = []
for case_id in expected_ids:
    number = int(case_id.split("-")[3])
    check = evidence[number]
    surface_auc = float(new_h1[case_id]["surface_auc_tie_corrected"])
    deep_auc = float(new_h1[case_id]["deep_auc_tie_corrected"])
    surface_dle = float(new_h1[case_id]["surface_dle_mm"])
    deep_dle = float(new_h1[case_id]["deep_dle_mm"])
    case_summary.append({
        "case": number,
        "scenario": new_h1[case_id]["scenario"],
        "h0_An_cal_AUC": float(new_h0[case_id]["auc"]),
        "h1_An_cal_AUC": float(new_h1[case_id]["auc"]),
        "old_h1_An_cal_AUC": float(old_h1[case_id]["auc"]),
        "h1_An_auc": float(new_h1[case_id]["auc_tie_corrected"]),
        "surface_An_auc": surface_auc if math.isfinite(surface_auc) else None,
        "deep_An_auc": deep_auc if math.isfinite(deep_auc) else None,
        "surface_DLE_mm": surface_dle if math.isfinite(surface_dle) else None,
        "deep_DLE_mm": deep_dle if math.isfinite(deep_dle) else None,
        "H0_converged": bool(int(check["null_converged"])),
        "H1_converged": bool(int(check["full_converged"])),
    })

summary = {
    "complete": True,
    "phase": "development",
    "snr_eeg_meg_db": [-10, -10],
    "mrf_strength": 0.8,
    "case_count": 5,
    "all_fits_converged": all(item["H0_converged"] and item["H1_converged"] for item in case_summary),
    "An_cal_AUC_gate": 0.9,
    "passed_cases_H1": sum(item["h1_An_cal_AUC"] >= .9 for item in case_summary),
    "accepted": False,
    "reason": "case00, case01, case03 and case04 H1 An_cal_AUC remain below 0.90; case01 H1 did not converge",
    "protocol_note": "Only mixed cases 03/04 rescale the deep component to a fixed 0.5 joint EEG/MEG noise-normalized sensor-amplitude ratio; no truth enters the inverse.",
    "provenance_note": "Combined from an intentionally stopped two-case shard and one completed three-case shard; calibration and validation were not read.",
    "cases": case_summary,
}
(output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# 传感器平衡 -10/-10 dB 开发结果（拒绝）", "",
    "这不是通过结果：局部 `An_cal_AUC` 的 0.90 门槛只通过 1/5，且 case01 的 H1 未收敛。",
    "传感器平衡只改变含深源的 case03/04；纯表层 case00/01 与纯深层 case02 保持原协议。",
    "校准集和验证集均未读取。", "",
    "| case | 情况 | H0 AUC | H1 AUC | 旧 H1 AUC | H1 An_auc | 表层 An_auc | 深层 An_auc | 表层 DLE | 深层 DLE | 收敛 |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for item in case_summary:
    display = lambda value: "—" if value is None else f"{value:.4f}"
    lines.append(f"| {item['case']:02d} | {item['scenario']} | {item['h0_An_cal_AUC']:.4f} | "
        f"{item['h1_An_cal_AUC']:.4f} | {item['old_h1_An_cal_AUC']:.4f} | {item['h1_An_auc']:.4f} | "
        f"{display(item['surface_An_auc'])} | {display(item['deep_An_auc'])} | {display(item['surface_DLE_mm'])} | "
        f"{display(item['deep_DLE_mm'])} | {'是' if item['H0_converged'] and item['H1_converged'] else '否'} |")
lines += ["", "case03 的 H1 局部 AUC 从 0.6012 提高到 0.7382，深层 DLE 为 0 mm；"
    "但 case04 仍为 0.7329，表层 DLE 未检出。改善说明旧协议确实把混合深源压得过低，"
    "同时也说明当前自适应稀疏惩罚仍会漏掉部分表层活动区，不能作为最终算法。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
