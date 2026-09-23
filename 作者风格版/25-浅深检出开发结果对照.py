"""Summarize completed ERP-v6 development experiments only; no inverse jobs."""

# %% 仅读取四个开发目录。未完成的批次明确跳过，允许刷新汇总，绝不覆盖原结果。
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import protocol
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive
import plot_erp_whole_head_results as style

base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_summary"
inputs = {
    "IRLS wide, 100": base / "dev_predictive_wide_5_5",
    "IRLS wide, 300": base / "dev_predictive_wide_300",
    "ADMM": base / "dev_predictive_admm_5_5",
    "IRLS near-convex": base / "dev_predictive_nearconvex_5_5",
}
completed, skipped = {}, []
for variant, directory in inputs.items():
    marker = directory / "completion.json"
    if not marker.exists():
        skipped.append(dict(variant=variant, directory=str(directory), reason="completion.json absent; run not completed"))
        print("SKIP", variant, "incomplete", flush=True)
        continue
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if not completion.get("complete", False):
        skipped.append(dict(variant=variant, directory=str(directory), reason="complete flag is false"))
        print("SKIP", variant, "incomplete", flush=True)
        continue
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["phase"] == "development", "Never read calibration/validation scores here"
    with (directory / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (directory / "evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
        evidence = list(csv.DictReader(stream))
    assert len(evidence) == completion["case_count"] == len(metadata["case_ids"])
    assert {row["case_id"] for row in evidence} == set(metadata["case_ids"])
    assert all(row["status"] == "ok" for row in rows)
    assert len({(row["case_id"], row["method"]) for row in rows}) == len(rows)
    assert all(row["deep_present_decision"] == "" for row in evidence), "These are ungated development results"
    assert all(not np.isfinite(float(row["p_value"])) for row in evidence)
    records = {case_id: json.loads((directory / (case_id + ".json")).read_text(encoding="utf-8"))
               for case_id in metadata["case_ids"]}
    completed[variant] = dict(directory=directory, completion=completion, metadata=metadata,
                              rows=rows, evidence=evidence, records=records)
assert completed, "No completed development experiment available"

# %% 固定五个5/5病例配对；源真值、噪声、白化及指标实现必须一致。
reference = completed["IRLS wide, 100"]
ref_meta = reference["metadata"]
manifest_path = Path(ref_meta["manifest"])
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == ref_meta["manifest_sha256"]
manifest = {case["case_id"]: case for case in json.loads(manifest_path.read_text(encoding="utf-8"))}
paired_ids = sorted(case_id for case_id in reference["metadata"]["case_ids"]
                    if manifest[case_id]["eeg_snr_db"] == manifest[case_id]["meg_snr_db"] == 5)
assert len(paired_ids) == 5 and len({manifest[key]["configuration_id"] for key in paired_ids}) == 5
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
assert original._shared_fingerprint(shared) == ref_meta["shared_fingerprint"]
penalty_mm = float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000)
provenance = {}
raw_rows, evidence_rows = [], []
for variant, item in completed.items():
    meta = item["metadata"]
    for key in ("manifest_sha256", "seed_root", "shared_fingerprint", "environment"):
        assert meta[key] == ref_meta[key], f"{variant}: incompatible {key}"
    for name in ("benchmark\\erp_replicates.py", "benchmark\\erp_protocol.py", "benchmark\\protocol.py",
                 "benchmark\\methods.py", "benchmark\\metrics.py", "metrics\\user_metrics\\An_auc.py"):
        assert meta["code_sha256"][name] == ref_meta["code_sha256"][name], f"{variant}: changed data/metric code {name}"
    assert set(paired_ids) <= set(meta["case_ids"])
    for case_id in paired_ids:
        assert item["records"][case_id]["simulation"] == reference["records"][case_id]["simulation"], "paired observation metadata differs"
    settings = item["records"][paired_ids[0]]["fitting"]
    evidence_map = {row["case_id"]: row for row in item["evidence"]}
    for case_id, record in item["records"].items():
        assert record["fitting"]["solver_settings"] == settings["solver_settings"]
        assert record["fitting"].get("structural_settings") == settings.get("structural_settings")
        assert {row["method"] for row in item["rows"] if row["case_id"] == case_id} >= {"v6-surface-only", "v6-full-ungated"}
        evidence = evidence_map[case_id]
        for model in ("null", "full"):
            assert int(evidence[model + "_converged"]) == record["fitting"][model + "_model"]["windows"][0]["solver"]["converged"]
        evidence_rows.append(dict(variant=variant, paired_5_5=int(case_id in paired_ids),
                                 configuration_kind=manifest[case_id]["configuration_kind"], **evidence))
    for row in item["rows"]:
        record = dict(variant=variant, paired_5_5=int(row["case_id"] in paired_ids), **row)
        evidence = evidence_map[row["case_id"]]
        record.update(null_converged=int(evidence["null_converged"]),
                      full_converged=int(evidence["full_converged"]), both_converged=int(evidence["all_converged"]),
                      prediction_score=float(evidence["score"]))
        for layer in ("surface", "deep"):
            for metric in ("sd_mm", "dle_mm"):
                name = layer + "_" + metric
                value = float(row[name])
                present = int(row["has_surface_true" if layer == "surface" else "has_deep_true"])
                record[name + "_penalized"] = (np.nan if not present else penalty_mm
                    if not np.isfinite(value) or (layer == "deep" and not int(row["deep_detected"])) else value)
        raw_rows.append(record)
    provenance[variant] = dict(directory=str(item["directory"]), completion=item["completion"],
        requested_solver_settings=meta["solver_settings"], effective_solver_settings=settings["solver_settings"],
        solver_kind=settings["null_model"]["solver_kind"], structural_settings=settings.get("structural_settings"),
        older_structural_metadata_note="wide-100 did not serialize structural_settings; original null/full model metadata retained in raw case JSON",
        changed_code_vs_reference=[name for name, digest in meta["code_sha256"].items()
                                   if digest != ref_meta["code_sha256"].get(name)],
        code_sha256=meta["code_sha256"],
        input_hashes={name: hashlib.sha256((item["directory"] / name).read_bytes()).hexdigest()
                      for name in ("metadata.json", "completion.json", "rows.csv", "evidence.csv")})

# %% 同时保存H0/H1原始指标、分层漏检罚值与收敛。不把幅度规则的TP/FP当校准后的检出性能。
summaries = []
for variant, item in completed.items():
    scopes = {"paired_5_5": paired_ids}
    if len(item["metadata"]["case_ids"]) > 5:
        scopes["all_20_development"] = item["metadata"]["case_ids"]
    for scope, case_ids in scopes.items():
        evid = [row for row in evidence_rows if row["variant"] == variant and row["case_id"] in case_ids]
        for method in sorted({row["method"] for row in item["rows"]}):
            rows = [row for row in raw_rows if row["variant"] == variant and row["case_id"] in case_ids and row["method"] == method]
            assert len(rows) == len(case_ids)
            values = archive._aggregate(rows, penalty_mm)
            for name in ("deep_sensitivity", "deep_specificity", "deep_balanced_accuracy"):
                values["raw_amplitude_" + name] = values.pop(name)
            summaries.append(dict(variant=variant, scope=scope, method=method,
                null_converged=sum(int(row["null_converged"]) for row in evid),
                full_converged=sum(int(row["full_converged"]) for row in evid),
                both_converged=sum(int(row["all_converged"]) for row in evid),
                raw_amplitude_deep_TP=sum(int(row["deep_detected"]) for row in rows),
                raw_amplitude_deep_FP=sum(int(row["deep_false_positive"]) for row in rows),
                true_deep_cases=sum(int(row["has_deep_true"]) for row in rows),
                true_surface_only_cases=sum(not int(row["has_deep_true"]) for row in rows), **values))
output.mkdir(parents=True, exist_ok=True)
for filename, table in (("raw_method_metrics.csv", raw_rows), ("raw_prediction_evidence.csv", evidence_rows),
                        ("method_summary.csv", summaries)):
    archive._atomic_csv(output / filename, table, table[0].keys())

# %% 分数图仅画零线，不设置或挑选检出阈值。颜色和形状区分无深源/有深源。
style._style()
plt = style.plt
figure, axes = plt.subplots(1, len(completed), figsize=(4 * len(completed), 4.8), squeeze=False)
labels = ("Single\ncortex", "Two\ncortices", "Deep\nonly", "Deep +\ncortex", "Deep +\n2 cortices")
for axis, (variant, item) in zip(axes[0], completed.items()):
    scores = {row["case_id"]: float(row["score"]) for row in item["evidence"]}
    values = np.array([scores[key] for key in paired_ids])
    spread = max(float(np.ptp(np.r_[values, 0])), .001)
    axis.axhline(0, color="#777777", linewidth=.8, linestyle="--")
    for index, case_id in enumerate(paired_ids):
        deep = manifest[case_id].get("deep_index") is not None
        axis.scatter(index, values[index], marker="D" if deep else "s", s=48,
                     color="#D55E00" if deep else "#0072B2", zorder=3)
        axis.annotate(f"{values[index]:.3g}", (index, values[index]), xytext=(0, 8),
                      textcoords="offset points", ha="center", fontsize=8)
    axis.set_xticks(range(5), labels)
    axis.set_ylim(min(0, values.min()) - .1 * spread, max(0, values.max()) + .22 * spread)
    evid = [row for row in item["evidence"] if row["case_id"] in paired_ids]
    axis.set_title(variant + f"\nH0/H1 converged: {sum(int(row['null_converged']) for row in evid)}/5, {sum(int(row['full_converged']) for row in evid)}/5", fontsize=10)
    axis.set_ylabel("Held-out prediction loss gain T")
axes[0, 0].scatter([], [], marker="s", color="#0072B2", label="Pure cortex: no deep truth")
axes[0, 0].scatter([], [], marker="D", color="#D55E00", label="Deep truth present")
figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.5, .93), frameon=False)
figure.suptitle("Same five development configurations | EEG/MEG 5/5 dB", y=.99)
figure.text(.5, .015, "Zero line: positive prediction gain; NOT a calibrated detection threshold. Panel scales differ.", ha="center", fontsize=9)
figure.tight_layout(rect=(0, .06, 1, .86))
figure.savefig(output / "prediction_scores_paired_5_5.png", dpi=200, bbox_inches="tight")
plt.close(figure)

# %% 描述性结论与来源。没有读取校准集或验证集，不报告p值或经过验收的误报率。
report = ["# ERP-v6 开发结果对照", "", "仅汇总已完成开发批次；未完成批次跳过。允许重跑本汇总脚本刷新图表，不改原始结果。",
    "", f"已完成批次：{len(completed)}/4；尚未完成：" + ("、".join(row["variant"] for row in skipped) or "无") + "。",
    "", "固定同五个5/5 dB配置进行配对；wide-300另保留全部20例（五配置×四SNR组合）。重复SNR不是独立空间样本；不计算CI或p值。",
    "", "## 预测分数：高空模型分数与弱混合源", "",
    "| 变体 | 单表层 T | 双表层 T | 纯深层 T | 深+单表层 T | 深+双表层 T | H0/H1均收敛 |",
    "|---|---:|---:|---:|---:|---:|---:|"]
gaps = []
for variant, item in completed.items():
    selected = [next(row for row in item["evidence"] if row["case_id"] == key) for key in paired_ids]
    values = [float(row["score"]) for row in selected]
    report.append(f"| {variant} | " + " | ".join(f"{value:.6g}" for value in values) + f" | {sum(int(row['all_converged']) for row in selected)}/5 |")
    gaps.append(dict(variant=variant, max_pure_surface_T=max(values[:2]),
        min_mixed_T=min(values[3:]), mixed_cases_above_max_surface=sum(value > max(values[:2]) for value in values[3:])))
report += ["", "正T仅说明完整模型在这份独立确认观测上降低平方损失，不等于检出深源。纯表层也可能因可重复的皮层拟合偏差获得正T；必须先冻结算法，再用独立纯表层病例校准。图中零线不是检出阈值。"]
for gap in gaps:
    report.append(f"- {gap['variant']}：纯表层最大T={gap['max_pure_surface_T']:.6g}；混合病例最小T={gap['min_mixed_T']:.6g}；仅{gap['mixed_cases_above_max_surface']}/2个混合病例高于这两例纯表层最大值。这是开发排序诊断，不是校准规则。")
report += ["", "## H0/H1指标与原始幅度规则", "",
    "| 变体 | 范围 | 模型 | An_auc | 表层An_auc | 深层An_auc | 原幅度TP | 原幅度FP |",
    "|---|---|---|---:|---:|---:|---:|---:|"]
for row in summaries:
    report.append(f"| {row['variant']} | {row['scope']} | {row['method']} | {row['auc_tie_corrected']:.4f} | {row['surface_auc_tie_corrected']:.4f} | {row['deep_auc_tie_corrected']:.4f} | {row['raw_amplitude_deep_TP']}/{row['true_deep_cases']} | {row['raw_amplitude_deep_FP']}/{row['true_surface_only_cases']} |")
report += ["", "这里TP/FP是未经预测门控的既有幅度比≥0.14规则（TP另要求峰距≤10mm），仅用于暴露原始定位问题，绝不能当作校准后v6性能。H0先验不含深源，因此其零误报同时伴随必然深源漏检，并不表示成功。",
    "", f"CSV保留An_cal_AUC字段auc、An_auc字段auc_tie_corrected及分层An_auc；源层不存在时分层指标NaN。原始SD/DLE和漏检罚分同时保留，罚值为源网格bbox对角线{penalty_mm:.6f}mm；深层罚分依照原幅度规则而非尚不存在的校准决策。未过滤未收敛病例。",
    "", "## 可比性与尚未完成的工作", "",
    "配对检查包含manifest、共享前向/噪声指纹、随机根种子、完整仿真元数据（实际噪声SNR/半组种子/白化权重）、环境、仿真与指标代码指纹。算法/运行器代码字节变化及每版实际求解参数另存metadata.json，不能把ADMM非光滑目标与IRLS平滑目标解释为纯数值实现的等价替换。",
    "", "validation_not_run=true：本脚本没有读取校准/验证分数，未做校准决策，不能宣称硬性误报验收通过或检出率达标；也没有证明真实数据优势。",
    "", "输出：raw_method_metrics.csv（全部已完成病例/方法原始及罚分指标）、raw_prediction_evidence.csv（全部预测分数与双模型收敛）、method_summary.csv（5/5配对及wide-300全部20例）、prediction_scores_paired_5_5.png、metadata.json。"]
(output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
(output / "metadata.json").write_text(json.dumps(dict(
    scope="Completed development runs only; descriptive matched comparison, not calibrated deep-presence performance",
    validation_not_run=True, calibration_scores_read=False, paired_case_ids=paired_ids,
    completed_variants=list(completed), skipped_incomplete=skipped,
    all_requested_batches_completed=not skipped, source_provenance=provenance,
    manifest_sha256=ref_meta["manifest_sha256"], shared_fingerprint=ref_meta["shared_fingerprint"],
    missing_distance_penalty_mm=penalty_mm, gaps=gaps,
    raw_detection_rule="amplitude ratio >=0.14; TP additionally peak distance <=10mm; not a calibrated decision",
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Completed variants:", list(completed), "raw rows:", len(raw_rows), "evidence rows:", len(evidence_rows), flush=True)
print(output, flush=True)
