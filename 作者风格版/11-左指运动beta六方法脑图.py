#%%
# 只读脚本 10 已保存的源图；不重新预处理、不重新运行逆解。
# 六种方法统一使用：正向 beta-ERD → 各自峰值归一化 → P95/峰值 5% 较高者。
# 因此脑图只能比较峰位/空间形态，不能比较绝对振幅或激活面积。

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


#%%
# ==================== 1. 路径和显示参数 ====================

project_root = Path(__file__).resolve().parents[1]
result_dir = project_root / "results" / "real_data" / "ds006035" / "sub-sm09_finger_beta_multimethod"
source_file = result_dir / "beta_multimethod_source_maps.npz"
subjects_dir = Path(r"D:\mne_data\MNE-sample-data\subjects")
subject = "sample"
methods = ("DICS", "LCMV", "MNE", "dSPM", "sLORETA", "eLORETA")
display_percentile = 95.0
camera_zoom = 0.72

assert source_file.is_file(), f"找不到六方法源图：{source_file}"
assert (subjects_dir / subject).is_dir(), f"找不到 sample pial 解剖：{subjects_dir}"
mne.set_log_level("warning")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


#%%
# ==================== 2. 同一可视化规则渲染六张完整 pial 俯视图 ====================

saved = np.load(source_file)
vertices = [saved["vertices_lh"], saved["vertices_rh"]]
n_sources = sum(len(item) for item in vertices)
image_files = []

for method in methods:
    source_map = np.maximum(np.asarray(saved[f"pooled__{method}"], dtype=float), 0.0)
    assert source_map.shape == (n_sources,) and np.isfinite(source_map).all()
    peak_db = float(np.max(source_map))
    assert peak_db > 0.0, f"{method} 没有正向 beta-ERD，不能渲染"
    normalized = source_map / peak_db
    # 这里正向 ERD 只占全部顶点的 1.5%~4.1%，全顶点 P95 恰为 0。
    # 因此统一加峰值 5% 的显示下限；仅影响渲染，不改变源图或 ROI 指标。
    threshold = max(float(np.percentile(normalized, display_percentile)), 0.05)
    display_map = np.where(normalized > threshold, normalized, 0.0)
    assert np.count_nonzero(display_map) > 0

    stc = mne.SourceEstimate(
        display_map[:, None], vertices=vertices, tmin=0.0, tstep=1.0, subject=subject
    )
    brain = stc.plot(
        subject=subject, subjects_dir=subjects_dir, surface="pial",
        hemi="both", views="dorsal", initial_time=0.0,
        time_viewer=False, show_traces=False, time_label=None,
        colormap="inferno",
        clim={"kind": "value", "lims": [threshold, (threshold + 1.0) / 2.0, 1.0]},
        smoothing_steps=5, transparent=True, background="white",
        foreground="black", cortex="classic", colorbar=False,
        size=(900, 720), backend="pyvistaqt",
        brain_kwargs={"show": False, "theme": "light"},
    )
    brain.plotter.camera.zoom(camera_zoom)
    brain.plotter.render()
    image_file = result_dir / f"brain_beta_ERD_{method}_pial_dorsal.png"
    brain.save_image(image_file)
    brain.close()
    image_files.append(image_file)
    print(method, "P95/峰值：", round(threshold, 4), "图：", image_file)


#%%
# ==================== 3. 带清楚标题的六方法对照图 ====================

figure, axes = plt.subplots(2, 3, figsize=(15.5, 10.0), layout="constrained")
for axis, method, image_file in zip(axes.ravel(), methods, image_files):
    axis.imshow(plt.imread(image_file))
    axis.set_title(f"{method} · beta-ERD · dorsal pial", fontsize=13)
    axis.axis("off")
figure.suptitle(
    "ds006035 sub-sm09 左指动作：15–30 Hz beta-ERD（基线 -1500~-1000 ms；活动 -400~200 ms）",
    fontsize=15,
)
figure.text(
    0.5, 0.01,
    "3 runs / 113 epochs；每图各自峰值归一化，显示阈值=max(P95, 峰值5%)；仅比较峰位与空间形态，不能跨方法比较振幅或范围。"
    " sample 模板解剖；无源位置真值。",
    ha="center", fontsize=10, color="#444444",
)
montage_file = result_dir / "beta_six_methods_pial_dorsal_montage.png"
figure.savefig(montage_file, dpi=190, bbox_inches="tight", facecolor="white")
plt.close(figure)

assert len(image_files) == 6 and all(path.is_file() for path in image_files)
assert montage_file.is_file()
print("六方法脑图：", montage_file)
