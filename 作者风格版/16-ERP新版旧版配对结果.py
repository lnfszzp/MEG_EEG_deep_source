# %% 同一批观测的新旧版本比较；按配置聚类，不把重复 SNR 当独立受试者。
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import plot_erp_whole_head_results as plotting

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--reference", type=Path, default=root / "results/erp_whole_head/development_full_v3/v3_locked")
args = parser.parse_args()
output = args.input / "paired_v3_v4"
output.mkdir(parents=True, exist_ok=True)
with (args.input / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
    new_rows = [row for row in csv.DictReader(stream) if row["method"] == "OASTER-ERP-v4"]
with (args.reference / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
    old_rows = {row["case_id"]: row for row in csv.DictReader(stream) if row["method"] == "OASTER-ERP-v3"}
new_meta = json.loads((args.input / "metadata.json").read_text(encoding="utf-8"))
old_meta = json.loads((args.reference / "metadata.json").read_text(encoding="utf-8"))
assert new_meta["manifest_sha256"] == old_meta["manifest_sha256"]
assert new_meta["erp_seed_root"] == old_meta["erp_seed_root"]
assert new_meta["checkpoint_shared_fingerprint"] == old_meta["checkpoint_shared_fingerprint"]
assert new_rows and all(row["case_id"] in old_rows and row["status"] == "ok" for row in new_rows)

# %% 所有病例配对，不挑最好结果；深检统一使用当前 0.14 阈值。
metrics = ("auc_tie_corrected", "surface_auc_tie_corrected", "deep_auc_tie_corrected",
           "surface_sd_mm", "surface_dle_mm", "deep_sd_mm", "deep_dle_mm")
pairs = []
for new in new_rows:
    old = old_rows[new["case_id"]]
    pair = {name: new[name] for name in ("case_id", "configuration_id", "scenario", "eeg_snr_db", "meg_snr_db")}
    for version, row in (("v3", old), ("v4", new)):
        for metric in metrics:
            pair[f"{version}_{metric}"] = float(row[metric])
        has_deep = int(row["has_deep_true"])
        deep_positive = float(row["deep_score"]) >= .14
        deep_distance = float(row["deep_peak_distance_mm"])
        pair[f"{version}_deep_detected"] = int(has_deep and deep_positive and deep_distance <= 10. + 1e-6)
        pair[f"{version}_deep_false_positive"] = int(not has_deep and deep_positive)
    pairs.append(pair)
with (output / "paired_cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
    writer.writeheader()
    writer.writerows(pairs)

# %% 以配置为单位先平均各 SNR，再给配对改变量做聚类 bootstrap。
rng = np.random.default_rng(20260922)
summary = []
for scenario in ("ALL", *plotting.SCENARIOS):
    selected = pairs if scenario == "ALL" else [pair for pair in pairs if pair["scenario"] == scenario]
    for metric in metrics:
        valid = [pair for pair in selected if np.isfinite(pair[f"v3_{metric}"]) and np.isfinite(pair[f"v4_{metric}"])]
        if not valid:
            continue
        config_ids = sorted({pair["configuration_id"] for pair in valid})
        delta = np.array([np.mean([pair[f"v4_{metric}"] - pair[f"v3_{metric}"] for pair in valid
                                  if pair["configuration_id"] == key]) for key in config_ids])
        draws = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
        summary.append(dict(scenario=scenario, metric=metric, cases=len(valid), configurations=len(delta),
                            v3_mean=np.mean([pair[f"v3_{metric}"] for pair in valid]),
                            v4_mean=np.mean([pair[f"v4_{metric}"] for pair in valid]),
                            config_mean_change=delta.mean(), ci95_low=np.quantile(draws, .025),
                            ci95_high=np.quantile(draws, .975)))
with (output / "paired_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

# %% 图1：相同病例的新旧 AUC，图2：每场景49格差异；缺测格明确灰色。
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
colors = ("#4477AA", "#228833", "#CC6677", "#AA3377")
for axis, metric, title in zip(axes, metrics[:3], ("All-source An_auc", "Surface An_auc", "Deep An_auc")):
    for scenario, color in zip(plotting.SCENARIOS, colors):
        selected = [pair for pair in pairs if pair["scenario"] == scenario]
        axis.scatter([p[f"v3_{metric}"] for p in selected], [p[f"v4_{metric}"] for p in selected],
                     s=12, alpha=.55, c=color, label=scenario)
    axis.plot([0, 1], [0, 1], color="#999999", linewidth=1)
    axis.set(xlabel="v3 fixed templates", ylabel="v4 adaptive graph", title=title, xlim=(0, 1), ylim=(0, 1))
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(output / "paired_auc_scatter.png", dpi=180)
plt.close(fig)
levels = plotting.SNR_LEVELS
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for axis, scenario in zip(axes, plotting.SCENARIOS):
    grid = np.full((7, 7), np.nan)
    for i, eeg in enumerate(levels):
        for j, meg in enumerate(levels):
            selected = [p for p in pairs if p["scenario"] == scenario and int(p["eeg_snr_db"]) == eeg and int(p["meg_snr_db"]) == meg]
            if selected:
                grid[i, j] = np.mean([p["v4_auc_tie_corrected"] - p["v3_auc_tie_corrected"] for p in selected])
    axis.set_facecolor("#dddddd")
    shown = axis.imshow(grid, origin="lower", cmap="RdBu", vmin=-.3, vmax=.3)
    axis.set(xticks=range(7), xticklabels=levels, yticks=range(7), yticklabels=levels,
             xlabel="MEG SNR (dB)", ylabel="EEG SNR (dB)", title=scenario.replace("_", " "))
fig.colorbar(shown, ax=axes.tolist(), shrink=.7, label="v4 minus v3 An_auc; gray = not run")
fig.savefig(output / "paired_auc_snr_change.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(output)
