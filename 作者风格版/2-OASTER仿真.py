# %%
# 这个脚本把 corrected-v2 的一个仿真 case 完整展开，便于逐块查看 OASTER。
# 主文件不定义函数：从参数、真值、正向投影一直运行到指标和保存结果。
#
# 注意：这里是“看算法流程”的可重复仿真，会根据 manifest 里的种子重新生成一次噪声。
# 正式 strict-blind 结果仍然使用冻结的 F_EEG/F_MEG，不能用这个脚本替代盲评入口。

from pathlib import Path
import hashlib
import json
from operator import itemgetter
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# %%
# ==================== 1. 参数区：主要改这里 ====================

project_root = Path(r"D:\博士\工作＆汇报\源定位\新建文件夹")
manifest_path = project_root / "results" / "corrected_v2" / "strict_blind" / "manifest.json"
geometry_root = project_root / "corrected_v2" / "generated"
save_root = project_root / "作者风格版" / "结果" / "OASTER仿真"

# 四种情况：
# "surface_only"                只有表层源
# "deep_only"                   只有深层源
# "deep_plus_surface"           一个表层源 + 一个深层源
# "deep_plus_two_surface"       两个表层源 + 一个深层源
scenario = "deep_plus_surface"

eeg_snr_db = 0
meg_snr_db = 0

# 同一个 scenario、EEG SNR、MEG SNR 单元格内的第几个 case，从 1 开始。
cell_case_number = 1

surface_scales_mm = (0.0, 4.0, 7.0)
show_figure = False

assert scenario in {
    "surface_only",
    "deep_only",
    "deep_plus_surface",
    "deep_plus_two_surface",
}
assert cell_case_number >= 1

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import protected_multilayer as protected
from benchmark import metrics as benchmark_metrics
from benchmark import protocol
from candidates import oaster_rebuilt as oaster

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# %%
# ==================== 2. 读取 corrected-v2 的 manifest ====================

manifest_bytes = manifest_path.read_bytes()
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")

if sidecar_path.exists():
    expected_sha256 = sidecar_path.read_text(encoding="ascii").split()[0]
    assert manifest_sha256 == expected_sha256, "manifest 的 SHA-256 校验失败"

manifest = json.loads(manifest_bytes)

cell_cases = [
    item for item in manifest
    if item["scenario"] == scenario
    and int(item["eeg_snr_db"]) == eeg_snr_db
    and int(item["meg_snr_db"]) == meg_snr_db
]
cell_cases = sorted(cell_cases, key=itemgetter("case_number"))

assert cell_cases, "没有找到这个 scenario 和 SNR 组合"
assert cell_case_number <= len(cell_cases), "cell_case_number 超出这个单元格的 case 数量"

case = cell_cases[cell_case_number - 1]

print("manifest SHA-256:", manifest_sha256)
print("scenario:", scenario)
print("EEG SNR:", eeg_snr_db, "dB")
print("MEG SNR:", meg_snr_db, "dB")
print("本单元 case:", cell_case_number, "/", len(cell_cases))
print("case_id:", case["case_id"])
print("表层中心点:", case["surface_centers"])
print("深层源点:", case["deep_index"])


# %%
# ==================== 3. 读取当前几何、forward 和噪声模型 ====================
# corrected_v2/generated 保存了和正式结果完全一致的混合源空间。
# 前 n_surf 个点是皮层表面，后 n_deep 个点是双侧丘脑深层源。

shared = protocol.load_shared(data_root=geometry_root)

gain_eeg = np.asarray(shared["gain_eeg"], dtype=float)
gain_meg = np.asarray(shared["gain_meg"], dtype=float)
src_vertices = np.asarray(shared["vertices"], dtype=float)
vert_conn = np.asarray(shared["adjacency"], dtype=np.uint8)
times = np.asarray(shared["times"], dtype=float)

n_surf = int(shared["n_surf"])
n_deep = int(shared["n_deep"])
n_sources = n_surf + n_deep
noise_samples = int(protocol.ACTIVE_START)
active_index = np.arange(noise_samples, times.size)

print("EEG Gain shape:", gain_eeg.shape)
print("MEG Gain shape:", gain_meg.shape)
print("表层源点数:", n_surf)
print("深层源点数:", n_deep)
print("时间点数:", times.size)
print("基线点数:", noise_samples)

assert gain_eeg.shape[1] == n_sources
assert gain_meg.shape[1] == n_sources
assert src_vertices.shape == (n_sources, 3)
assert vert_conn.shape == (n_sources, n_sources)
assert int(case["deep_index"] or n_surf) >= n_surf


# %%
# ==================== 4. 按 manifest 构造仿真真值 ====================
# 表层源不是单点，而是中心点周围两阶邻域组成的高斯 patch。
# 深层源目前是丘脑候选点中的一个点。
# active 段使用带半正弦包络的 11/14/17 Hz 等振荡波形。

s_true, true_groups, truth_information = protocol.truth_for_case(shared, case)

true_source_index = np.flatnonzero(np.sum(s_true[:, active_index] ** 2, axis=1) > 0)
true_surface_index = true_source_index[true_source_index < n_surf]
true_deep_index = true_source_index[true_source_index >= n_surf]

print("s_true shape:", s_true.shape)
print("真实源组:", [group.tolist() for group in true_groups])
print("真实表层源点数:", true_surface_index.size)
print("真实深层源点数:", true_deep_index.size)
print("设定的源间相关系数:", case["correlation"])
print("实际深层/表层范数比:", truth_information["deep_surface_ratio_actual"])

assert s_true.shape == (n_sources, times.size)
assert np.allclose(s_true[:, :noise_samples], 0.0)
assert true_source_index.size > 0


# %%
# ==================== 5. forward：真值投影到 EEG 和 MEG ====================

clean_eeg = gain_eeg @ s_true
clean_meg = gain_meg @ s_true

print("clean EEG shape:", clean_eeg.shape)
print("clean MEG shape:", clean_meg.shape)
print("clean EEG active energy:", np.sum(clean_eeg[:, active_index] ** 2))
print("clean MEG active energy:", np.sum(clean_meg[:, active_index] ** 2))

assert np.sum(clean_eeg[:, active_index] ** 2) > 0
assert np.sum(clean_meg[:, active_index] ** 2) > 0


# %%
# ==================== 6. 只加一次有色噪声 ====================
# 这里没有 trial 维度，也没有 epochs.average()。
# 即：一个确定性源信号 -> 一次 forward -> 一次有色高斯噪声。
# 所以这里不是“很多次单试次 ERP 再叠加平均”的仿真。

eeg_seed, meg_seed = np.random.SeedSequence(case["seed"]).spawn(2)
eeg_rng = np.random.default_rng(eeg_seed)
meg_rng = np.random.default_rng(meg_seed)

noise_eeg = np.asarray(shared["noise_factor_eeg"]) @ eeg_rng.standard_normal(clean_eeg.shape)
noise_meg = np.asarray(shared["noise_factor_meg"]) @ meg_rng.standard_normal(clean_meg.shape)

clean_eeg_energy = float(np.sum(clean_eeg[:, active_index] ** 2))
clean_meg_energy = float(np.sum(clean_meg[:, active_index] ** 2))
noise_eeg_energy = float(np.sum(noise_eeg[:, active_index] ** 2))
noise_meg_energy = float(np.sum(noise_meg[:, active_index] ** 2))

noise_eeg *= np.sqrt(clean_eeg_energy / (noise_eeg_energy * 10 ** (eeg_snr_db / 10.0)))
noise_meg *= np.sqrt(clean_meg_energy / (noise_meg_energy * 10 ** (meg_snr_db / 10.0)))

F_eeg = clean_eeg + noise_eeg
F_meg = clean_meg + noise_meg

actual_eeg_snr_db = 10 * np.log10(
    clean_eeg_energy / np.sum(noise_eeg[:, active_index] ** 2)
)
actual_meg_snr_db = 10 * np.log10(
    clean_meg_energy / np.sum(noise_meg[:, active_index] ** 2)
)

print("仿真 trial 数: 1（没有叠加平均）")
print("实际 EEG SNR:", actual_eeg_snr_db, "dB")
print("实际 MEG SNR:", actual_meg_snr_db, "dB")

assert np.isclose(actual_eeg_snr_db, eeg_snr_db, atol=1e-10)
assert np.isclose(actual_meg_snr_db, meg_snr_db, atol=1e-10)


# %%
# ==================== 7. EEG 和 MEG 分别白化，再上下拼接 ====================
# 白化矩阵只用前 200 个基线时间点估计。
# 两种模态分别白化以后再融合，避免 EEG 和 MEG 的物理单位直接混在一起。

whitener_eeg = protected.whitening_matrix(F_eeg, noise_samples)
whitener_meg = protected.whitening_matrix(F_meg, noise_samples)

F_eeg_white = whitener_eeg @ F_eeg
F_meg_white = whitener_meg @ F_meg
gain_eeg_white = whitener_eeg @ gain_eeg
gain_meg_white = whitener_meg @ gain_meg

F_white = np.vstack((F_eeg_white, F_meg_white))
gain_white = np.vstack((gain_eeg_white, gain_meg_white))

print("EEG 白化后 shape:", F_eeg_white.shape)
print("MEG 白化后 shape:", F_meg_white.shape)
print("融合数据 shape:", F_white.shape)
print("融合 Gain shape:", gain_white.shape)

assert F_white.shape[0] == gain_white.shape[0]
assert F_white.shape[1] == times.size
assert gain_white.shape[1] == n_sources


# %%
# ==================== 8. 建立 0/4/7 mm 表层空间模板 ====================
# 0 mm 是单点模板，4 mm 和 7 mm 是限制在皮层连通网格内的高斯 patch。

kernels = protected.connected_euclidean_surface_kernels(
    src_vertices,
    vert_conn,
    n_surf,
    scales_mm=surface_scales_mm,
)

kernel_by_scale = dict(kernels)

for scale_mm, kernel in kernels:
    print(scale_mm, "mm kernel shape:", kernel.shape, "非零数:", kernel.nnz)

assert set(kernel_by_scale) == set(surface_scales_mm)


# %%
# ==================== 9. active/baseline 频谱筛选 + SVD 时间基 ====================
# 这一步说明当前还原版 OASTER 确实具有明显的频谱/能量定位倾向。
# 它不是直接取 ERP 某一个峰值的有符号传感器拓扑。

temporal_basis = oaster._temporal_basis(F_white, noise_samples=noise_samples)

print("时间基 shape:", temporal_basis.shape)
print("保留的时间秩:", temporal_basis.shape[0])

assert temporal_basis.ndim == 2
assert temporal_basis.shape[1] == times.size


# %%
# ==================== 10. EBIC 逐个选择空间模板 ====================
# 候选包括 0/4/7 mm 表层模板和所有深层点。
# 每加入一个候选都重新计算 EBIC；没有改善就停止。

s_ebic, selected_templates = oaster._ebic_templates(
    F_white,
    gain_white,
    n_surf,
    kernels,
    temporal_basis,
    ridge_fraction=oaster.RIDGE_FRACTION,
)

print("EBIC 选择的模板数:", selected_templates)
print("EBIC 结果 shape:", s_ebic.shape)

assert s_ebic.shape == s_true.shape


# %%
# ==================== 11. deep rescue：检查遗漏的深层源 ====================
# 在 EBIC 已解释的传感器子空间之外，再尝试加入一个深层候选。
# 只有加入以后条件 EBIC 下降，才接受这个深层源。

s_deep_rescue, deep_information = oaster.deep_rescue_trial(
    F_white,
    gain_white,
    s_ebic,
    n_surf,
    temporal_basis,
)

deep_ebic_delta = float(deep_information["ebic_delta_without_tau"]) + oaster.DEEP_RESCUE_TAU
deep_rescue_accepted = deep_ebic_delta < 0.0

if deep_rescue_accepted:
    s_primary = s_ebic + s_deep_rescue
else:
    s_primary = s_ebic.copy()

print("deep rescue 候选（deep local index）:", deep_information["deep_local"])
print("deep rescue EBIC 变化:", deep_ebic_delta)
print("是否接受 deep rescue:", deep_rescue_accepted)

assert s_primary.shape == s_true.shape


# %%
# ==================== 12. 计算 4 mm 模板上的频谱证据 ====================
# 频谱证据只作为次要补充，最后峰值缩放为主结果的 5%。

s_spectral = oaster._multiscale_spectral_evidence_source(
    F_white,
    gain_white,
    n_surf,
    kernel_by_scale[4.0],
    noise_samples=noise_samples,
)

primary_peak = np.linalg.norm(s_primary[:, active_index], axis=1).max(initial=0.0)
spectral_peak = np.linalg.norm(s_spectral[:, active_index], axis=1).max(initial=0.0)

print("主结果 active peak:", primary_peak)
print("频谱证据 active peak:", spectral_peak)
print("频谱证据比例:", oaster.SPECTRAL_FRACTION)

assert s_spectral.shape == s_true.shape


# %%
# ==================== 13. 合成最终 OASTER 估计 ====================

s_oaster = oaster._add_scaled_evidence(
    s_primary,
    s_spectral,
    oaster.SPECTRAL_FRACTION,
    noise_samples=noise_samples,
)

print("最终 OASTER shape:", s_oaster.shape)
print("是否全部为有限值:", np.all(np.isfinite(s_oaster)))

assert s_oaster.shape == s_true.shape
assert np.all(np.isfinite(s_oaster))


# %%
# ==================== 14. 计算 AUC、SD、DLE 等指标 ====================
# evaluate_estimate 内部使用项目中保存的 An_auc / An_cal_AUC / SD / DLE_an。
# 表层和深层的 SD、DLE 分开给出；不存在的层会显示 nan。

metrics = benchmark_metrics.evaluate_estimate(
    s_oaster,
    s_true,
    src_vertices,
    true_groups,
    n_surf,
    active_index,
    shared["auc_cortex"],
)

print("AUC（原 An_cal_AUC）:", metrics["auc"])
print("AUC（An_auc tied-rank）:", metrics["auc_tie_corrected"])
print("RMSE:", metrics["rmse"])
print("表层 SD:", metrics["surface_sd_mm"], "mm")
print("表层 DLE:", metrics["surface_dle_mm"], "mm")
print("深层 SD:", metrics["deep_sd_mm"], "mm")
print("深层 DLE:", metrics["deep_dle_mm"], "mm")
print("深层 peak distance:", metrics["deep_peak_distance_mm"], "mm")
print("active source count:", metrics["active_count"])


# %%
# ==================== 15. 保存这个 case 的全部结果 ====================

snr_name = f"EEG{eeg_snr_db:+d}_MEG{meg_snr_db:+d}"
save_dir = save_root / scenario / snr_name / f"case_{cell_case_number:03d}"
save_dir.mkdir(parents=True, exist_ok=True)

metric_values = {
    name: int(value) if isinstance(value, (int, np.integer)) else float(value)
    for name, value in metrics.items()
}

result_information = {
    "case": case,
    "manifest_sha256": manifest_sha256,
    "simulation": "one colored-noise realization; no trials; no averaging",
    "actual_eeg_snr_db": float(actual_eeg_snr_db),
    "actual_meg_snr_db": float(actual_meg_snr_db),
    "temporal_rank": int(temporal_basis.shape[0]),
    "selected_templates": int(selected_templates),
    "deep_rescue_accepted": bool(deep_rescue_accepted),
    "deep_rescue_deep_local": int(deep_information["deep_local"]),
    "deep_rescue_ebic_delta": float(deep_ebic_delta),
    "metrics": metric_values,
}

np.savez_compressed(
    save_dir / "OASTER仿真结果.npz",
    times=times,
    src_vertices=src_vertices,
    s_true=s_true,
    s_oaster=s_oaster,
    clean_eeg=clean_eeg,
    clean_meg=clean_meg,
    F_eeg=F_eeg,
    F_meg=F_meg,
    true_source_index=true_source_index,
)

(save_dir / "指标.json").write_text(
    json.dumps(result_information, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
    encoding="utf-8",
)

print("结果保存到:", save_dir)


# %%
# ==================== 16. 简单查看传感器信号、真值和定位结果 ====================

truth_amplitude = benchmark_metrics.source_amplitude(s_true, active_index)
oaster_amplitude = benchmark_metrics.source_amplitude(s_oaster, active_index)

truth_amplitude /= max(float(truth_amplitude.max(initial=0.0)), 1e-30)
oaster_amplitude /= max(float(oaster_amplitude.max(initial=0.0)), 1e-30)

eeg_plot_scale = max(float(np.std(F_eeg[0])), 1e-30)
meg_plot_scale = max(float(np.std(F_meg[0])), 1e-30)

figure, axes = plt.subplots(2, 2, figsize=(13, 8))

axes[0, 0].plot(times, clean_eeg[0] / eeg_plot_scale, color="#0072B2", label="clean EEG")
axes[0, 0].plot(times, F_eeg[0] / eeg_plot_scale, color="#D55E00", alpha=0.65, label="noisy EEG")
axes[0, 0].axvline(times[noise_samples], color="0.35", linestyle="--", linewidth=1)
axes[0, 0].set_title("EEG 第一个通道：一次加噪，无叠加平均")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Normalized amplitude")
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(times, clean_meg[0] / meg_plot_scale, color="#009E73", label="clean MEG")
axes[0, 1].plot(times, F_meg[0] / meg_plot_scale, color="#CC79A7", alpha=0.65, label="noisy MEG")
axes[0, 1].axvline(times[noise_samples], color="0.35", linestyle="--", linewidth=1)
axes[0, 1].set_title("MEG 第一个通道：一次加噪，无叠加平均")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Normalized amplitude")
axes[0, 1].legend(frameon=False)

axes[1, 0].plot(truth_amplitude, color="#0072B2", linewidth=1.3, label="仿真真值")
axes[1, 0].plot(oaster_amplitude, color="#D55E00", linewidth=1.0, alpha=0.85, label="OASTER")
axes[1, 0].axvline(n_surf, color="0.25", linestyle="--", linewidth=1, label="表层/深层边界")
axes[1, 0].set_title("源空间结果")
axes[1, 0].set_xlabel("Source index")
axes[1, 0].set_ylabel("Normalized active amplitude")
axes[1, 0].legend(frameon=False)

metric_text = "\n".join(
    (
        f"scenario: {scenario}",
        f"EEG / MEG SNR: {eeg_snr_db} / {meg_snr_db} dB",
        f"temporal rank: {temporal_basis.shape[0]}",
        f"selected templates: {selected_templates}",
        f"AUC (An_auc): {metrics['auc_tie_corrected']:.4f}",
        f"surface SD / DLE: {metrics['surface_sd_mm']:.2f} / {metrics['surface_dle_mm']:.2f} mm",
        f"deep SD / DLE: {metrics['deep_sd_mm']:.2f} / {metrics['deep_dle_mm']:.2f} mm",
    )
)
axes[1, 1].axis("off")
axes[1, 1].text(0.03, 0.95, metric_text, va="top", fontsize=12, linespacing=1.7)

figure.suptitle("corrected-v2 单 case OASTER 仿真", fontsize=15)
figure.tight_layout()
figure.savefig(save_dir / "OASTER仿真检查图.png", dpi=220, bbox_inches="tight")

if show_figure:
    plt.show()
else:
    plt.close(figure)

print("检查图:", save_dir / "OASTER仿真检查图.png")
