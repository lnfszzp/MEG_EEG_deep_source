#%%
# ds006035 的 Finger 不是自主敲击，而是右腕正中神经刺激后尽快抬起左食指。
# 因此这里把运动定位拆成两条互补链：
# 1. response-lock 的时域 MF / MEFI / MEFII（OASTER-ERP、dSPM、eLORETA）；
# 2. response-lock 的频域 mu/beta ERD 与 PMBR（共同滤波器 DICS）。
#
# 代码按作者习惯从上往下写，不定义自己的 def/class/lambda。
# 三个 run 的头位不同，所以每个 run 各自建 forward 和做逆解；只在共同的
# sample 皮层网格上按有效试次数聚合，不能把三个 run 的传感器数据直接拼起来
# 再套用 run-1 的 forward。

from pathlib import Path
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.forward import _merge_fwds
from mne.time_frequency import csd_morlet
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from candidates import oaster_rebuilt as oaster
import protected_multilayer as protected
from pipelines import run_ds006035_somatomotor as real_pipeline


mne.set_log_level("warning")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


#%%
# ==================== 1. 参数区：平时主要改这里 ====================

dataset_path = Path(r"D:\博士\工作＆汇报\源定位\开源数据\v1.0.0")
sample_data_path = Path(r"D:\mne_data\MNE-sample-data")
subjects_dir = sample_data_path / "subjects"
subject = "sm09"
runs = (1, 2, 3)
sample_subject = "sample"
spacing = "ico4"

trans_file = (
    project_root
    / "results"
    / "real_data"
    / "ds006035"
    / f"sub-{subject}_final"
    / f"sub-{subject}_to_sample-trans.fif"
)
save_dir = (
    project_root
    / "results"
    / "real_data"
    / "ds006035"
    / f"sub-{subject}_finger_dual_chain"
)
save_dir.mkdir(parents=True, exist_ok=True)

filter_low = 0.5
# 最高 DICS 频率是 30 Hz；40 Hz 低通足够，并减少高频肌电进入时域窗功率。
filter_high = 40.0
notch_frequency_fallback = 60.0
minimum_rt = 0.10
maximum_rt = 0.90

response_epoch = (-1.50, 1.10)
response_baseline = (-1.50, -1.00)
stimulus_epoch = (-0.50, 0.60)
stimulus_baseline = (-0.50, -0.05)

# MF 在动作前，MEFI/MEFII 在动作后。
time_windows = {
    "MF_response_-80_-20ms": ("response", -0.080, -0.020),
    "MEFI_response_20_60ms": ("response", 0.020, 0.060),
    "MEFII_response_120_180ms": ("response", 0.120, 0.180),
}
# stimulus 180~240 ms 与多数试次的 response-MF 是同一段物理时间，默认不当作独立证据。
include_stimulus_m1 = False
if include_stimulus_m1:
    time_windows["M1_stimulus_180_240ms"] = ("stimulus", 0.180, 0.240)

lambda2 = 1.0 / 9.0
depth = 0.8
# MF、MEFI、MEFII 和 stimulus-M1 各自作为单主导发生器检验；不是按 ROI 调参。
oaster_max_templates = 1

dics_erd_window = (-0.40, 0.20)
dics_pmbr_window = (0.50, 1.00)
dics_frequencies = {
    "mu_8_14Hz": np.arange(8.0, 15.0, 1.0),
    "beta_15_30Hz": np.arange(15.0, 31.0, 2.0),
}
dics_regularization = 0.05
dics_wavelet_seconds = 0.40
dics_sample_rate = 200.0

method_names = ("OASTER-ERP", "dSPM", "eLORETA")
method_colors = {
    "OASTER-ERP": "#0072B2",
    "dSPM": "#D55E00",
    "eLORETA": "#009E73",
}
draw_brain = False
brain_display_percentile = 95.0

print("结果保存到：", save_dir)
assert dataset_path.is_dir(), f"找不到 ds006035：{dataset_path}"
assert trans_file.is_file(), f"找不到已经保存的模板配准：{trans_file}"
assert (subjects_dir / sample_subject).is_dir(), f"找不到 sample 解剖：{subjects_dir}"
assert response_baseline == (-1.50, -1.00)
assert maximum_rt <= 0.90, "response 基线安全性依赖 RT 不超过 0.90 s"


#%%
# ==================== 2. 建立共同皮层候选网格，只建立一次 ====================
# 真正进入逆解的 analysis_src 要等 forward 的 mindist=5 mm 裁剪完成后再确定。
# 每个 run 后面仍会使用自己的 info/dev_head_t 重算 forward。

all_started = time.perf_counter()
trans = mne.read_trans(trans_file, verbose=False)
source_space = mne.setup_source_space(
    sample_subject,
    spacing=spacing,
    subjects_dir=subjects_dir,
    add_dist=False,
    n_jobs=1,
    verbose=False,
)
analysis_src = None

bem_folder = subjects_dir / sample_subject / "bem"
eeg_bem = bem_folder / "sample-5120-5120-5120-bem-sol.fif"
meg_bem = bem_folder / "sample-5120-bem-sol.fif"

print("共同候选源点数：", sum(len(space["vertno"]) for space in source_space))


#%%
# ==================== 3. 三个 run：严格 32→16 配对 ====================
# 规则：一个 32 只能配到下一个 32 之前的第一个 16；多余事件单独报告。
# RT 超出 0.10~0.90 s 的配对不进入定位。这个上限同时保证 response
# -1.5~-1.0 s 位于本试次 event32 前至少 0.1 s。

run_inputs = []
pair_rows = []
pair_summary_rows = []

for run in runs:
    stem = f"sub-{subject}_ses-meeg_task-somatomotor_run-{run}"
    data_folder = dataset_path / f"sub-{subject}" / "ses-meeg" / "meg"
    raw_file = data_folder / f"{stem}_meg.fif"
    events_file = data_folder / f"{stem}_events.tsv"

    assert raw_file.is_file(), f"找不到 raw：{raw_file}"
    assert events_file.is_file(), f"找不到 events：{events_file}"

    raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
    first_samp = int(raw.first_samp)
    event_table = pd.read_csv(events_file, sep="\t")
    event_table["value_numeric"] = pd.to_numeric(event_table["value"], errors="coerce")
    event_table = event_table[event_table["value_numeric"].isin([16, 32])].copy()
    event_table["sample_numeric"] = event_table["sample"].astype(float).astype(int)
    event_table = event_table.sort_values("sample_numeric")

    pending_stimulus = None
    paired_stimulus_samples = []
    paired_response_samples = []
    isolated_stimulus = 0
    isolated_response = 0
    invalid_rt_pairs = 0

    for event_row in event_table.itertuples(index=False):
        event_value = int(event_row.value_numeric)
        event_sample = int(event_row.sample_numeric) + first_samp

        if event_value == 32:
            if pending_stimulus is not None:
                isolated_stimulus += 1
            pending_stimulus = event_sample
        elif pending_stimulus is None:
            isolated_response += 1
        else:
            reaction_time = (event_sample - pending_stimulus) / float(raw.info["sfreq"])
            if minimum_rt <= reaction_time <= maximum_rt:
                paired_stimulus_samples.append(pending_stimulus)
                paired_response_samples.append(event_sample)
                pair_rows.append({
                    "subject": subject,
                    "run": run,
                    "pair_number": len(paired_stimulus_samples),
                    "stimulus_sample": pending_stimulus,
                    "response_sample": event_sample,
                    "reaction_time_ms": reaction_time * 1000.0,
                })
            else:
                invalid_rt_pairs += 1
            pending_stimulus = None

    if pending_stimulus is not None:
        isolated_stimulus += 1

    paired_stimulus_samples = np.asarray(paired_stimulus_samples, dtype=int)
    paired_response_samples = np.asarray(paired_response_samples, dtype=int)
    reaction_times = (
        paired_response_samples - paired_stimulus_samples
    ) / float(raw.info["sfreq"])

    paired_events = np.zeros((2 * len(reaction_times), 3), dtype=int)
    paired_events[0::2, 0] = paired_stimulus_samples
    paired_events[0::2, 2] = 32
    paired_events[1::2, 0] = paired_response_samples
    paired_events[1::2, 2] = 16
    paired_events = paired_events[np.argsort(paired_events[:, 0])]

    all_stimulus_samples = (
        event_table.loc[event_table["value_numeric"] == 32, "sample_numeric"]
        .to_numpy(dtype=int)
        + first_samp
    )
    all_stimulus_events = np.zeros((len(all_stimulus_samples), 3), dtype=int)
    all_stimulus_events[:, 0] = all_stimulus_samples
    all_stimulus_events[:, 2] = 32

    pair_summary_rows.append({
        "subject": subject,
        "run": run,
        "event32_total": int(np.sum(event_table["value_numeric"] == 32)),
        "event16_total": int(np.sum(event_table["value_numeric"] == 16)),
        "valid_pairs": len(reaction_times),
        "isolated_event32": isolated_stimulus,
        "isolated_event16": isolated_response,
        "invalid_rt_pairs": invalid_rt_pairs,
        "rt_min_ms": float(np.min(reaction_times) * 1000.0),
        "rt_median_ms": float(np.median(reaction_times) * 1000.0),
        "rt_mean_ms": float(np.mean(reaction_times) * 1000.0),
        "rt_max_ms": float(np.max(reaction_times) * 1000.0),
    })

    assert len(reaction_times) >= 30, f"run-{run} 的有效 32→16 配对太少"
    assert np.all((reaction_times >= minimum_rt) & (reaction_times <= maximum_rt))

    run_inputs.append({
        "run": run,
        "raw": raw,
        "paired_events": paired_events,
        "all_stimulus_events": all_stimulus_events,
        "reaction_times": reaction_times,
        "raw_file": raw_file,
        "events_file": events_file,
    })

pair_table = pd.DataFrame(pair_rows)
pair_summary_table = pd.DataFrame(pair_summary_rows)
reaction_time_seconds = pair_table["reaction_time_ms"].to_numpy(dtype=float) / 1000.0
stimulus_m1_overlaps_response_mf = (
    (reaction_time_seconds - 0.080 <= 0.240)
    & (reaction_time_seconds - 0.020 >= 0.180)
)
stimulus_m1_overlap_fraction = float(np.mean(stimulus_m1_overlaps_response_mf))
pair_table.to_csv(save_dir / "reaction_times.csv", index=False, encoding="utf-8-sig")
pair_summary_table.to_csv(
    save_dir / "event_pairing_report.csv",
    index=False,
    encoding="utf-8-sig",
)
print(pair_summary_table.to_string(index=False))
print(
    "stimulus 180~240 ms 与 response-MF 重叠：",
    int(stimulus_m1_overlaps_response_mf.sum()),
    "/",
    len(stimulus_m1_overlaps_response_mf),
    f"({100.0 * stimulus_m1_overlap_fraction:.1f}%)",
)


#%%
# ==================== 4. 每个 run：先插值 event32 脉冲，再滤波 ====================
# 明确保留 EEG、102 MAG、204 GRAD；只允许后面自动删除异常 EEG。

for run_input in run_inputs:
    raw = run_input["raw"]
    picks = mne.pick_types(
        raw.info,
        meg=True,
        eeg=True,
        eog=False,
        ecg=False,
        stim=False,
        ref_meg=False,
        exclude=[],
    )
    raw.pick(picks)
    raw.load_data(verbose=False)

    mag_count = len(mne.pick_types(raw.info, meg="mag", eeg=False, exclude=[]))
    grad_count = len(mne.pick_types(raw.info, meg="grad", eeg=False, exclude=[]))
    eeg_count = len(mne.pick_types(raw.info, meg=False, eeg=True, exclude=[]))
    assert mag_count == 102, f"run-{run_input['run']} MAG 不是 102 个"
    assert grad_count == 204, f"run-{run_input['run']} GRAD 不是 204 个"

    # 必须在零相位滤波前去掉电刺激方波，避免把振铃拖进时域和频域窗口。
    real_pipeline.interpolate_stimulation_artifacts(
        raw,
        run_input["all_stimulus_events"],
        tmin=-0.002,
        tmax=0.008,
    )
    line_frequency = float(raw.info.get("line_freq") or notch_frequency_fallback)
    if line_frequency < raw.info["sfreq"] / 2.0:
        raw.notch_filter([line_frequency], n_jobs=1, verbose=False)
    raw.filter(filter_low, filter_high, n_jobs=1, verbose=False)

    run_input["meg_channels"] = mag_count + grad_count
    run_input["eeg_channels_before_qc"] = eeg_count
    print(
        f"run-{run_input['run']}：EEG={eeg_count}，MAG={mag_count}，GRAD={grad_count}"
    )


#%%
# ==================== 5. 每个 run：EEG 坏道检查和双锁时分段 ====================
# response 基线固定为 -1.5~-1.0 s；stimulus 基线固定为 -0.5~-0.05 s。

for run_input in run_inputs:
    raw = run_input["raw"]
    paired_events = run_input["paired_events"]

    probe_epochs = mne.Epochs(
        raw,
        paired_events,
        event_id={"right_wrist": 32},
        tmin=stimulus_baseline[0],
        tmax=0.0,
        baseline=None,
        preload=True,
        proj=False,
        event_repeated="drop",
        verbose=False,
    )
    eeg_probe_picks = mne.pick_types(probe_epochs.info, meg=False, eeg=True, exclude=[])
    probe_window = (
        (probe_epochs.times >= stimulus_baseline[0])
        & (probe_epochs.times <= stimulus_baseline[1])
    )
    eeg_probe_data = probe_epochs.get_data(copy=False)[:, eeg_probe_picks][:, :, probe_window]
    median_ptp = np.median(np.ptp(eeg_probe_data, axis=2), axis=0)
    ptp_center = float(np.median(median_ptp))
    ptp_mad = float(np.median(np.abs(median_ptp - ptp_center)))
    ptp_cutoff = max(100e-6, ptp_center + 6.0 * 1.4826 * ptp_mad)
    auto_bad_eeg = [
        probe_epochs.ch_names[channel_index]
        for channel_index, channel_ptp in zip(eeg_probe_picks, median_ptp)
        if channel_ptp > ptp_cutoff
    ]
    if auto_bad_eeg:
        raw.drop_channels(auto_bad_eeg)

    assert len(mne.pick_types(raw.info, meg=True, eeg=False, exclude=[])) == 306

    reject = {"eeg": 200e-6, "mag": 6e-12, "grad": 4e-10}
    response_epochs = mne.Epochs(
        raw,
        paired_events,
        event_id={"left_finger": 16},
        tmin=response_epoch[0],
        tmax=response_epoch[1],
        baseline=response_baseline,
        reject=reject,
        preload=True,
        proj=False,
        event_repeated="drop",
        verbose=False,
    )
    stimulus_epochs = mne.Epochs(
        raw,
        paired_events,
        event_id={"right_wrist": 32},
        tmin=stimulus_epoch[0],
        tmax=stimulus_epoch[1],
        baseline=stimulus_baseline,
        reject=reject,
        preload=True,
        proj=False,
        event_repeated="drop",
        verbose=False,
    )
    response_epochs.set_eeg_reference("average", projection=True, verbose=False)
    stimulus_epochs.set_eeg_reference("average", projection=True, verbose=False)

    assert len(response_epochs) >= 20, f"run-{run_input['run']} response epoch 太少"
    assert len(stimulus_epochs) >= 20, f"run-{run_input['run']} stimulus epoch 太少"

    run_input["auto_bad_eeg"] = auto_bad_eeg
    run_input["response_epochs"] = response_epochs
    run_input["stimulus_epochs"] = stimulus_epochs
    run_input["response_evoked"] = response_epochs.average()
    run_input["stimulus_evoked"] = stimulus_epochs.average()
    run_input["response_cov"] = mne.compute_covariance(
        response_epochs,
        tmin=response_baseline[0],
        tmax=response_baseline[1],
        method="shrunk",
        rank="info",
        verbose=False,
    )
    run_input["stimulus_cov"] = mne.compute_covariance(
        stimulus_epochs,
        tmin=stimulus_baseline[0],
        tmax=stimulus_baseline[1],
        method="shrunk",
        rank="info",
        verbose=False,
    )
    print(
        f"run-{run_input['run']}：response={len(response_epochs)}，"
        f"stimulus={len(stimulus_epochs)}，异常EEG={auto_bad_eeg}"
    )


#%%
# ==================== 6. 每个 run：用自己的头位建立 EEG+306 MEG forward ====================

for run_input in run_inputs:
    info = run_input["response_evoked"].info
    eeg_picks = mne.pick_types(
        info, meg=False, eeg=True, ref_meg=False, exclude="bads"
    )
    meg_picks = mne.pick_types(
        info, meg=True, eeg=False, ref_meg=False, exclude="bads"
    )
    assert len(meg_picks) == 306

    eeg_info = mne.pick_info(info, eeg_picks, copy=True)
    meg_info = mne.pick_info(info, meg_picks, copy=True)

    eeg_forward = mne.make_forward_solution(
        eeg_info,
        trans,
        source_space,
        eeg_bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
        on_inside="raise",
        verbose=False,
    )
    meg_forward = mne.make_forward_solution(
        meg_info,
        trans,
        source_space,
        meg_bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=1,
        on_inside="raise",
        verbose=False,
    )
    eeg_forward_free = mne.convert_forward_solution(
        eeg_forward,
        surf_ori=True,
        force_fixed=False,
        use_cps=True,
        verbose=False,
    )
    meg_forward_free = mne.convert_forward_solution(
        meg_forward,
        surf_ori=True,
        force_fixed=False,
        use_cps=True,
        verbose=False,
    )
    eeg_forward_fixed = mne.convert_forward_solution(
        eeg_forward_free,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        verbose=False,
    )
    meg_forward_fixed = mne.convert_forward_solution(
        meg_forward_free,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        verbose=False,
    )
    mag_forward_fixed = mne.pick_types_forward(
        meg_forward_fixed,
        meg="mag",
        eeg=False,
        ref_meg=False,
        exclude=[],
    )
    grad_forward_fixed = mne.pick_types_forward(
        meg_forward_fixed,
        meg="grad",
        eeg=False,
        ref_meg=False,
        exclude=[],
    )

    for hemi in range(2):
        assert np.array_equal(
            eeg_forward_fixed["src"][hemi]["vertno"],
            meg_forward_fixed["src"][hemi]["vertno"],
        )
    if analysis_src is None:
        analysis_src = eeg_forward_fixed["src"].copy()
    else:
        for hemi in range(2):
            assert np.array_equal(
                analysis_src[hemi]["vertno"],
                eeg_forward_fixed["src"][hemi]["vertno"],
            ), "三个 run 经 mindist 裁剪后的皮层顶点必须完全一致"
    assert len(mag_forward_fixed["info"]["ch_names"]) == 102
    assert len(grad_forward_fixed["info"]["ch_names"]) == 204

    joint_forward = _merge_fwds(
        {"meg": meg_forward_free.copy(), "eeg": eeg_forward_free.copy()},
        verbose=False,
    )
    run_input["eeg_forward_fixed"] = eeg_forward_fixed
    run_input["mag_forward_fixed"] = mag_forward_fixed
    run_input["grad_forward_fixed"] = grad_forward_fixed
    run_input["joint_forward"] = joint_forward
    print(
        f"run-{run_input['run']} forward：EEG={len(eeg_forward_fixed['info']['ch_names'])}，"
        "MAG=102，GRAD=204"
    )

source_vertices = [
    np.asarray(space["vertno"], dtype=int)
    for space in analysis_src
]
source_xyz = np.vstack([
    space["rr"][space["vertno"]]
    for space in analysis_src
])
n_sources = len(source_xyz)
source_adjacency = mne.spatial_src_adjacency(analysis_src, verbose=False).toarray()
surface_kernels = protected.connected_euclidean_surface_kernels(
    source_xyz,
    source_adjacency,
    n_sources,
    scales_mm=oaster.SURFACE_SCALES_MM,
)
del source_adjacency

for run_input in run_inputs:
    assert run_input["eeg_forward_fixed"]["nsource"] == n_sources
    assert run_input["mag_forward_fixed"]["nsource"] == n_sources
    assert run_input["grad_forward_fixed"]["nsource"] == n_sources

print("mindist 裁剪后的共同皮层源点数：", n_sources)
print("OASTER 空间模板尺度：", [scale for scale, _ in surface_kernels])


#%%
# ==================== 7. 时域链：每个 run 分别运行 OASTER-ERP ====================
# EEG、MAG、GRAD 分开用安全基线白化，再联合进入同一个 OASTER 逆解。

time_run_maps = {}
time_run_available = {}
time_run_meta = []

for run_input in run_inputs:
    for lock_name in ("response", "stimulus"):
        lock_windows = [
            (component_name, window_start, window_stop)
            for component_name, (component_lock, window_start, window_stop)
            in time_windows.items()
            if component_lock == lock_name
        ]
        if not lock_windows:
            continue

        epochs = run_input[f"{lock_name}_epochs"]
        evoked = run_input[f"{lock_name}_evoked"]
        baseline = response_baseline if lock_name == "response" else stimulus_baseline
        trial_noise_data = epochs.get_data(
            tmin=baseline[0],
            tmax=baseline[1],
            copy=True,
            verbose=False,
        )
        trial_noise = trial_noise_data.transpose(1, 0, 2).reshape(
            len(evoked.ch_names),
            -1,
        )

        eeg_data, eeg_gain, eeg_noise = real_pipeline._aligned_modality(
            evoked.data,
            trial_noise,
            evoked.info,
            run_input["eeg_forward_fixed"],
            eeg=True,
        )
        mag_data, mag_gain, mag_noise = real_pipeline._aligned_modality(
            evoked.data,
            trial_noise,
            evoked.info,
            run_input["mag_forward_fixed"],
            eeg=False,
        )
        grad_data, grad_gain, grad_noise = real_pipeline._aligned_modality(
            evoked.data,
            trial_noise,
            evoked.info,
            run_input["grad_forward_fixed"],
            eeg=False,
        )

        eeg_whitener = protected.whitening_matrix(eeg_noise, eeg_noise.shape[1])
        mag_whitener = protected.whitening_matrix(mag_noise, mag_noise.shape[1])
        grad_whitener = protected.whitening_matrix(grad_noise, grad_noise.shape[1])
        eeg_white = eeg_whitener @ eeg_data
        mag_white = mag_whitener @ mag_data
        grad_white = grad_whitener @ grad_data
        joint_white = np.vstack((eeg_white, mag_white, grad_white))
        joint_gain_white = np.vstack((
            eeg_whitener @ eeg_gain,
            mag_whitener @ mag_gain,
            grad_whitener @ grad_gain,
        ))

        baseline_mask = (
            (evoked.times >= baseline[0])
            & (evoked.times <= baseline[1])
        )
        active_masks = tuple(
            (evoked.times >= window_start) & (evoked.times <= window_stop)
            for _, window_start, window_stop in lock_windows
        )
        window_channel_weights = []
        modality_weight_rows = []

        for (component_name, _, _), active_mask in zip(lock_windows, active_masks):
            modality_weights = real_pipeline.modality_evidence_weights(
                (eeg_data, mag_data, grad_data),
                baseline_mask,
                active_mask,
            )
            window_channel_weights.append(np.r_[
                np.full(eeg_white.shape[0], modality_weights[0]),
                np.full(mag_white.shape[0], modality_weights[1]),
                np.full(grad_white.shape[0], modality_weights[2]),
            ])
            modality_weight_rows.append({
                "run": run_input["run"],
                "component": component_name,
                "EEG_weight": float(modality_weights[0]),
                "MAG_weight": float(modality_weights[1]),
                "GRAD_weight": float(modality_weights[2]),
            })

        oaster_source, oaster_information = oaster.reconstruct_evoked_oaster_from_whitened(
            joint_white,
            joint_gain_white,
            n_sources,
            surface_kernels,
            baseline=baseline_mask,
            active_windows=active_masks,
            window_channel_weights=window_channel_weights,
            max_templates=oaster_max_templates,
            require_one=False,
        )
        run_input[f"{lock_name}_sensor_white"] = joint_white
        run_input[f"{lock_name}_oaster"] = oaster_source
        run_input[f"{lock_name}_oaster_information"] = oaster_information
        run_input.setdefault("modality_weight_rows", []).extend(modality_weight_rows)

        assert oaster_source.shape == (n_sources, len(evoked.times))
        assert np.isfinite(oaster_source).all()
        print(
            f"run-{run_input['run']} {lock_name} OASTER-ERP：",
            oaster_source.shape,
            oaster_information,
        )


#%%
# ==================== 8. 时域链：同一数据运行 dSPM 和 eLORETA ====================

for run_input in run_inputs:
    for lock_name in ("response", "stimulus"):
        lock_windows = [
            item
            for item in time_windows.items()
            if item[1][0] == lock_name
        ]
        if not lock_windows:
            continue

        evoked = run_input[f"{lock_name}_evoked"]
        noise_cov = run_input[f"{lock_name}_cov"]
        joint_forward = run_input["joint_forward"].copy()
        joint_names = joint_forward["sol"]["row_names"]
        inverse_evoked = evoked.copy().reorder_channels(joint_names)
        joint_forward["info"] = inverse_evoked.info
        inverse_cov = mne.pick_channels_cov(
            noise_cov,
            include=joint_names,
            exclude=[],
            ordered=True,
            copy=True,
            verbose=False,
        )
        inverse = mne.minimum_norm.make_inverse_operator(
            inverse_evoked.info,
            joint_forward,
            inverse_cov,
            loose=0.0,
            fixed=True,
            depth=depth,
            rank="info",
            use_cps=True,
            verbose=False,
        )

        dspm = mne.minimum_norm.apply_inverse(
            inverse_evoked,
            inverse,
            lambda2=lambda2,
            method="dSPM",
            use_cps=True,
            verbose=False,
        )
        eloreta = mne.minimum_norm.apply_inverse(
            inverse_evoked,
            inverse,
            lambda2=lambda2,
            method="eLORETA",
            use_cps=True,
            verbose=False,
        )
        run_input[f"{lock_name}_dspm"] = dspm.data
        run_input[f"{lock_name}_eloreta"] = eloreta.data

        assert dspm.data.shape == run_input[f"{lock_name}_oaster"].shape
        assert eloreta.data.shape == dspm.data.shape
        print(f"run-{run_input['run']} {lock_name}：dSPM/eLORETA 完成")


#%%
# ==================== 9. 时域链：形成每个 run 的四个源图（此时仍不读取 ROI） ====================

for run_input in run_inputs:
    for component_name, (lock_name, window_start, window_stop) in time_windows.items():
        evoked = run_input[f"{lock_name}_evoked"]
        baseline = response_baseline if lock_name == "response" else stimulus_baseline
        active_mask = (
            (evoked.times >= window_start)
            & (evoked.times <= window_stop)
        )
        baseline_mask = (
            (evoked.times >= baseline[0])
            & (evoked.times <= baseline[1])
        )
        source_by_method = {
            "OASTER-ERP": run_input[f"{lock_name}_oaster"],
            "dSPM": run_input[f"{lock_name}_dspm"],
            "eLORETA": run_input[f"{lock_name}_eloreta"],
        }
        lock_component_names = [
            name for name, (item_lock, _, _) in time_windows.items()
            if item_lock == lock_name
        ]
        oaster_window_index = lock_component_names.index(component_name)
        oaster_available = bool(
            run_input[f"{lock_name}_oaster_information"]["windows"]
            [oaster_window_index]["selected_templates"]
        )
        sensor_white = run_input[f"{lock_name}_sensor_white"]
        active_indices = np.flatnonzero(active_mask)
        sensor_gfp = np.linalg.norm(sensor_white[:, active_mask], axis=0)
        sensor_peak_index = int(active_indices[int(np.argmax(sensor_gfp))])
        sensor_peak_ms = float(evoked.times[sensor_peak_index] * 1000.0)

        for method_name, source in source_by_method.items():
            localization_available = method_name != "OASTER-ERP" or oaster_available
            active_power = np.mean(source[:, active_mask] ** 2, axis=1)
            baseline_power = np.mean(source[:, baseline_mask] ** 2, axis=1)
            source_map = np.sqrt(np.maximum(active_power - baseline_power, 0.0))
            time_run_maps[(run_input["run"], component_name, method_name)] = source_map
            time_run_available[(run_input["run"], component_name, method_name)] = (
                localization_available
            )
            time_run_meta.append({
                "subject": subject,
                "run": run_input["run"],
                "component": component_name,
                "lock": lock_name,
                "method": method_name,
                "n_trials": len(run_input[f"{lock_name}_epochs"]),
                "baseline_start_ms": baseline[0] * 1000.0,
                "baseline_stop_ms": baseline[1] * 1000.0,
                "window_start_ms": window_start * 1000.0,
                "window_stop_ms": window_stop * 1000.0,
                "sensor_gfp_peak_ms": sensor_peak_ms,
                "localization_available": localization_available,
            })


#%%
# ==================== 10. 频域链：单试次共同滤波 DICS ====================
# 同一频带用 baseline+movement+postmovement 整段数据建一个 common filter；
# 随后同一 filter 分别投影 baseline、ERD 和 PMBR，不能每个条件各建一套滤波器。

dics_run_maps = {}
dics_run_meta = []

for run_input in run_inputs:
    response_epochs = run_input["response_epochs"].copy().apply_proj(verbose=False)
    # DICS 最高只分析 30 Hz；仅频域副本降到 200 Hz，时域 ERP 保留原采样率。
    response_epochs.resample(dics_sample_rate, npad="auto", n_jobs=1, verbose=False)
    joint_forward = run_input["joint_forward"].copy()
    joint_names = joint_forward["sol"]["row_names"]
    response_epochs.reorder_channels(joint_names)
    joint_forward["info"] = response_epochs.info

    for band_name, frequencies in dics_frequencies.items():
        n_cycles = frequencies * dics_wavelet_seconds
        common_csd = csd_morlet(
            response_epochs,
            frequencies=frequencies,
            tmin=response_epoch[0],
            tmax=dics_pmbr_window[1],
            n_cycles=n_cycles,
            decim=1,
            n_jobs=1,
            verbose=False,
        )
        baseline_csd = csd_morlet(
            response_epochs,
            frequencies=frequencies,
            tmin=response_baseline[0],
            tmax=response_baseline[1],
            n_cycles=n_cycles,
            decim=1,
            n_jobs=1,
            verbose=False,
        )
        erd_csd = csd_morlet(
            response_epochs,
            frequencies=frequencies,
            tmin=dics_erd_window[0],
            tmax=dics_erd_window[1],
            n_cycles=n_cycles,
            decim=1,
            n_jobs=1,
            verbose=False,
        )
        pmbr_csd = csd_morlet(
            response_epochs,
            frequencies=frequencies,
            tmin=dics_pmbr_window[0],
            tmax=dics_pmbr_window[1],
            n_cycles=n_cycles,
            decim=1,
            n_jobs=1,
            verbose=False,
        )

        filters = mne.beamformer.make_dics(
            response_epochs.info,
            joint_forward,
            common_csd.mean(),
            reg=dics_regularization,
            noise_csd=baseline_csd.mean(),
            pick_ori="max-power",
            rank="info",
            weight_norm="unit-noise-gain",
            reduce_rank=True,
            depth=depth,
            real_filter=True,
            verbose=False,
        )
        baseline_stc, _ = mne.beamformer.apply_dics_csd(
            baseline_csd.mean(), filters, verbose=False
        )
        erd_stc, _ = mne.beamformer.apply_dics_csd(
            erd_csd.mean(), filters, verbose=False
        )
        pmbr_stc, _ = mne.beamformer.apply_dics_csd(
            pmbr_csd.mean(), filters, verbose=False
        )

        tiny = np.finfo(float).tiny
        baseline_power = np.maximum(baseline_stc.data[:, 0], tiny)
        erd_power = np.maximum(erd_stc.data[:, 0], tiny)
        pmbr_power = np.maximum(pmbr_stc.data[:, 0], tiny)
        erd_db = 10.0 * np.log10(baseline_power / erd_power)
        pmbr_db = 10.0 * np.log10(pmbr_power / baseline_power)

        dics_run_maps[(run_input["run"], band_name, "ERD")] = erd_db
        dics_run_maps[(run_input["run"], band_name, "PMBR")] = pmbr_db
        for contrast_name in ("ERD", "PMBR"):
            dics_run_meta.append({
                "subject": subject,
                "run": run_input["run"],
                "band": band_name,
                "contrast": contrast_name,
                "n_trials": len(response_epochs),
                "baseline_start_ms": response_baseline[0] * 1000.0,
                "baseline_stop_ms": response_baseline[1] * 1000.0,
                "active_start_ms": (
                    dics_erd_window[0] if contrast_name == "ERD" else dics_pmbr_window[0]
                ) * 1000.0,
                "active_stop_ms": (
                    dics_erd_window[1] if contrast_name == "ERD" else dics_pmbr_window[1]
                ) * 1000.0,
                "definition": (
                    "10log10(baseline/movement)"
                    if contrast_name == "ERD"
                    else "10log10(postmovement/baseline)"
                ),
            })
        print(f"run-{run_input['run']} {band_name}：ERD/PMBR DICS 完成")


#%%
# ==================== 11. 所有逆解完成后，才读取右侧运动/感觉 ROI ====================

labels = mne.read_labels_from_annot(
    sample_subject,
    parc="aparc",
    subjects_dir=subjects_dir,
    verbose=False,
)
source_label_names = np.full(n_sources, "unknown", dtype=object)
roi_masks = {
    "right_precentral": np.zeros(n_sources, dtype=bool),
    "right_postcentral": np.zeros(n_sources, dtype=bool),
}
left_source_count = len(source_vertices[0])

for label in labels:
    hemisphere_index = 0 if label.hemi == "lh" else 1
    source_offset = 0 if hemisphere_index == 0 else left_source_count
    label_source_indices = (
        np.flatnonzero(np.isin(source_vertices[hemisphere_index], label.vertices))
        + source_offset
    )
    source_label_names[label_source_indices] = label.name
    if label.name == "precentral-rh":
        roi_masks["right_precentral"][label_source_indices] = True
    elif label.name == "postcentral-rh":
        roi_masks["right_postcentral"][label_source_indices] = True

assert roi_masks["right_precentral"].any()
assert roi_masks["right_postcentral"].any()
print("ROI 只在逆解后读取：", {name: int(mask.sum()) for name, mask in roi_masks.items()})


#%%
# ==================== 12. 三个 run 在共同源网格聚合并保存时域指标 ====================
# 时域图是按有效试次数加权的 RMS 功率聚合，不直接平均不同头位的传感器波形。

time_pooled_maps = {}
time_metric_rows = []

for component_name in time_windows:
    for method_name in method_names:
        maps = []
        weights = []
        available_runs = []
        for run_input in run_inputs:
            key = (run_input["run"], component_name, method_name)
            if not time_run_available[key]:
                continue
            maps.append(time_run_maps[key])
            lock_name = time_windows[component_name][0]
            weights.append(len(run_input[f"{lock_name}_epochs"]))
            available_runs.append(run_input["run"])
        assert maps, f"{component_name} {method_name} 没有可聚合的定位结果"
        maps = np.asarray(maps)
        weights = np.asarray(weights, dtype=float)
        pooled_map = np.sqrt(np.average(maps**2, axis=0, weights=weights))
        time_pooled_maps[(component_name, method_name)] = pooled_map
        time_run_available[("pooled_1_2_3", component_name, method_name)] = available_runs

for meta in time_run_meta:
    source_map = time_run_maps[(meta["run"], meta["component"], meta["method"])]
    localization_available = bool(meta["localization_available"])
    if localization_available:
        peak_index = int(np.argmax(source_map))
        source_power = source_map**2
        total_power = float(source_power.sum())
        peak_label = str(source_label_names[peak_index])
        peak_hemi = "left" if peak_index < left_source_count else "right"
    else:
        peak_index = -1
        source_power = np.zeros_like(source_map)
        total_power = 0.0
        peak_label = "not_localized"
        peak_hemi = "none"
    for roi_name, roi_mask in roi_masks.items():
        roi_power = float(source_power[roi_mask].sum()) if localization_available else np.nan
        roi_fraction = float(roi_mask.mean())
        peak_distance_mm = float(
            np.min(np.linalg.norm(source_xyz[roi_mask] - source_xyz[peak_index], axis=1))
            * 1000.0
        ) if localization_available else np.nan
        time_metric_rows.append({
            **meta,
            "aggregation": "single_run",
            "roi": roi_name,
            "roi_vertices": int(roi_mask.sum()),
            "roi_mass_pct": 100.0 * roi_power / total_power if total_power > 0 else np.nan,
            "roi_enrichment": (
                (roi_power / total_power) / roi_fraction if total_power > 0 else np.nan
            ),
            "peak_distance_to_roi_mm": peak_distance_mm,
            "peak_label": peak_label,
            "peak_hemi": peak_hemi,
            "roi_used_before_inverse": False,
        })

for component_name, (lock_name, window_start, window_stop) in time_windows.items():
    baseline = response_baseline if lock_name == "response" else stimulus_baseline
    pooled_n_trials = int(sum(
        len(run_input[f"{lock_name}_epochs"])
        for run_input in run_inputs
    ))
    for method_name in method_names:
        source_map = time_pooled_maps[(component_name, method_name)]
        available_runs = time_run_available[("pooled_1_2_3", component_name, method_name)]
        peak_index = int(np.argmax(source_map))
        source_power = source_map**2
        total_power = float(source_power.sum())
        for roi_name, roi_mask in roi_masks.items():
            roi_power = float(source_power[roi_mask].sum())
            roi_fraction = float(roi_mask.mean())
            peak_distance_mm = float(
                np.min(np.linalg.norm(source_xyz[roi_mask] - source_xyz[peak_index], axis=1))
                * 1000.0
            )
            time_metric_rows.append({
                "subject": subject,
                "run": "pooled_1_2_3",
                "component": component_name,
                "lock": lock_name,
                "method": method_name,
                "n_trials": int(sum(
                    len(run_input[f"{lock_name}_epochs"])
                    for run_input in run_inputs
                    if run_input["run"] in available_runs
                )),
                "baseline_start_ms": baseline[0] * 1000.0,
                "baseline_stop_ms": baseline[1] * 1000.0,
                "window_start_ms": window_start * 1000.0,
                "window_stop_ms": window_stop * 1000.0,
                "sensor_gfp_peak_ms": np.nan,
                "aggregation": "trial_weighted_source_power_RMS",
                "localization_available": True,
                "available_runs": ";".join(str(run) for run in available_runs),
                "roi": roi_name,
                "roi_vertices": int(roi_mask.sum()),
                "roi_mass_pct": 100.0 * roi_power / total_power if total_power > 0 else 0.0,
                "roi_enrichment": (
                    (roi_power / total_power) / roi_fraction if total_power > 0 else 0.0
                ),
                "peak_distance_to_roi_mm": peak_distance_mm,
                "peak_label": str(source_label_names[peak_index]),
                "peak_hemi": "left" if peak_index < left_source_count else "right",
                "roi_used_before_inverse": False,
            })

time_metric_table = pd.DataFrame(time_metric_rows)
time_metric_file = save_dir / "time_domain_method_comparison.csv"
time_metric_table.to_csv(time_metric_file, index=False, encoding="utf-8-sig")

modality_weight_table = pd.DataFrame([
    row
    for run_input in run_inputs
    for row in run_input["modality_weight_rows"]
])
modality_weight_table.to_csv(
    save_dir / "oaster_modality_weights.csv",
    index=False,
    encoding="utf-8-sig",
)

time_npz = {
    f"{component_name}__{method_name}": source_map
    for (component_name, method_name), source_map in time_pooled_maps.items()
}
np.savez_compressed(
    save_dir / "time_domain_pooled_source_maps.npz",
    vertices_lh=source_vertices[0],
    vertices_rh=source_vertices[1],
    **time_npz,
)


#%%
# ==================== 13. 保存 DICS 指标，明确输出 beta PMBR ====================

dics_pooled_maps = {}
dics_metric_rows = []

for band_name in dics_frequencies:
    for contrast_name in ("ERD", "PMBR"):
        maps = np.asarray([
            dics_run_maps[(run_input["run"], band_name, contrast_name)]
            for run_input in run_inputs
        ])
        weights = np.asarray([
            len(run_input["response_epochs"])
            for run_input in run_inputs
        ], dtype=float)
        pooled_map = np.average(maps, axis=0, weights=weights)
        dics_pooled_maps[(band_name, contrast_name)] = pooled_map

for meta in dics_run_meta:
    source_map_signed = dics_run_maps[(meta["run"], meta["band"], meta["contrast"])]
    source_map = np.maximum(source_map_signed, 0.0)
    peak_index = int(np.argmax(source_map))
    source_power = source_map**2
    total_power = float(source_power.sum())
    for roi_name, roi_mask in roi_masks.items():
        roi_power = float(source_power[roi_mask].sum())
        roi_fraction = float(roi_mask.mean())
        peak_distance_mm = float(
            np.min(np.linalg.norm(source_xyz[roi_mask] - source_xyz[peak_index], axis=1))
            * 1000.0
        )
        dics_metric_rows.append({
            **meta,
            "aggregation": "single_run",
            "roi": roi_name,
            "roi_mass_pct": 100.0 * roi_power / total_power if total_power > 0 else 0.0,
            "roi_enrichment": (
                (roi_power / total_power) / roi_fraction if total_power > 0 else 0.0
            ),
            "peak_distance_to_roi_mm": peak_distance_mm,
            "peak_positive_db": float(source_map[peak_index]),
            "peak_label": str(source_label_names[peak_index]),
            "peak_hemi": "left" if peak_index < left_source_count else "right",
            "roi_used_before_inverse": False,
        })

for band_name in dics_frequencies:
    for contrast_name in ("ERD", "PMBR"):
        source_map_signed = dics_pooled_maps[(band_name, contrast_name)]
        source_map = np.maximum(source_map_signed, 0.0)
        peak_index = int(np.argmax(source_map))
        source_power = source_map**2
        total_power = float(source_power.sum())
        for roi_name, roi_mask in roi_masks.items():
            roi_power = float(source_power[roi_mask].sum())
            roi_fraction = float(roi_mask.mean())
            peak_distance_mm = float(
                np.min(np.linalg.norm(source_xyz[roi_mask] - source_xyz[peak_index], axis=1))
                * 1000.0
            )
            dics_metric_rows.append({
                "subject": subject,
                "run": "pooled_1_2_3",
                "band": band_name,
                "contrast": contrast_name,
                "n_trials": int(sum(len(item["response_epochs"]) for item in run_inputs)),
                "baseline_start_ms": response_baseline[0] * 1000.0,
                "baseline_stop_ms": response_baseline[1] * 1000.0,
                "active_start_ms": (
                    dics_erd_window[0] if contrast_name == "ERD" else dics_pmbr_window[0]
                ) * 1000.0,
                "active_stop_ms": (
                    dics_erd_window[1] if contrast_name == "ERD" else dics_pmbr_window[1]
                ) * 1000.0,
                "definition": (
                    "10log10(baseline/movement)"
                    if contrast_name == "ERD"
                    else "10log10(postmovement/baseline)"
                ),
                "aggregation": "trial_weighted_dB",
                "roi": roi_name,
                "roi_mass_pct": 100.0 * roi_power / total_power if total_power > 0 else 0.0,
                "roi_enrichment": (
                    (roi_power / total_power) / roi_fraction if total_power > 0 else 0.0
                ),
                "peak_distance_to_roi_mm": peak_distance_mm,
                "peak_positive_db": float(source_map[peak_index]),
                "peak_label": str(source_label_names[peak_index]),
                "peak_hemi": "left" if peak_index < left_source_count else "right",
                "roi_used_before_inverse": False,
            })

dics_metric_table = pd.DataFrame(dics_metric_rows)
dics_metric_file = save_dir / "dics_mu_beta_erd_pmbr_metrics.csv"
dics_metric_table.to_csv(dics_metric_file, index=False, encoding="utf-8-sig")

dics_npz = {
    f"{band_name}__{contrast_name}": source_map
    for (band_name, contrast_name), source_map in dics_pooled_maps.items()
}
np.savez_compressed(
    save_dir / "dics_pooled_source_maps.npz",
    vertices_lh=source_vertices[0],
    vertices_rh=source_vertices[1],
    beta_pmbr_db=dics_pooled_maps[("beta_15_30Hz", "PMBR")],
    **dics_npz,
)

beta_pmbr_source_table = pd.DataFrame({
    "source_index": np.arange(n_sources),
    "hemisphere": np.where(np.arange(n_sources) < left_source_count, "left", "right"),
    "vertex": np.r_[source_vertices[0], source_vertices[1]],
    "x_m": source_xyz[:, 0],
    "y_m": source_xyz[:, 1],
    "z_m": source_xyz[:, 2],
    "label": source_label_names,
    "beta_pmbr_db": dics_pooled_maps[("beta_15_30Hz", "PMBR")],
})
beta_pmbr_source_table.to_csv(
    save_dir / "dics_beta_pmbr_pooled_sources.csv",
    index=False,
    encoding="utf-8-sig",
)


#%%
# ==================== 14. 画时域三方法 × 双 ROI 比较图 ====================

pooled_time_table = time_metric_table[
    time_metric_table["run"].astype(str) == "pooled_1_2_3"
].copy()
component_order = list(time_windows)
component_display = ["MF\n-80~-20", "MEFI\n20~60", "MEFII\n120~180"]
if include_stimulus_m1:
    component_display.append("Stim-M1\n180~240")

figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), sharex=True, layout="constrained")
x = np.arange(len(component_order), dtype=float)
roi_order = ("right_precentral", "right_postcentral")
roi_titles = ("右侧 precentral（M1）", "右侧 postcentral（S1）")

for roi_column, (roi_name, roi_title) in enumerate(zip(roi_order, roi_titles)):
    for method_name in method_names:
        selected = pooled_time_table[
            (pooled_time_table["roi"] == roi_name)
            & (pooled_time_table["method"] == method_name)
        ].set_index("component").loc[component_order]
        axes[0, roi_column].plot(
            x,
            selected["roi_enrichment"],
            marker="o",
            linewidth=2.0,
            color=method_colors[method_name],
            label=method_name,
        )
        axes[1, roi_column].plot(
            x,
            selected["peak_distance_to_roi_mm"],
            marker="o",
            linewidth=2.0,
            color=method_colors[method_name],
        )
    axes[0, roi_column].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[0, roi_column].set_title(roi_title)
    axes[0, roi_column].set_ylabel("ROI 富集")
    axes[1, roi_column].set_ylabel("峰距 ROI（mm）")
    axes[1, roi_column].set_xticks(x, component_display)

for axis in axes.ravel():
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
figure.suptitle(
    f"sub-{subject} 三 run 左指运动：时域源定位比较\n"
    "各 run 独立 forward/inverse，按有效试次数在共同源网格聚合",
    fontsize=15,
    fontweight="bold",
)
time_figure_file = save_dir / "time_domain_method_comparison.png"
figure.savefig(time_figure_file, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(figure)


#%%
# ==================== 15. 画 DICS mu/beta ERD 与 PMBR 比较图 ====================

pooled_dics_table = dics_metric_table[
    dics_metric_table["run"].astype(str) == "pooled_1_2_3"
].copy()
dics_labels = ["mu ERD", "mu PMBR", "beta ERD", "beta PMBR"]
dics_keys = [
    ("mu_8_14Hz", "ERD"),
    ("mu_8_14Hz", "PMBR"),
    ("beta_15_30Hz", "ERD"),
    ("beta_15_30Hz", "PMBR"),
]

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), layout="constrained")
x = np.arange(len(dics_keys), dtype=float)
for roi_name, roi_title, color in zip(
    roi_order,
    roi_titles,
    ("#0072B2", "#CC79A7"),
):
    enrichment_values = []
    distance_values = []
    for band_name, contrast_name in dics_keys:
        selected = pooled_dics_table[
            (pooled_dics_table["roi"] == roi_name)
            & (pooled_dics_table["band"] == band_name)
            & (pooled_dics_table["contrast"] == contrast_name)
        ].iloc[0]
        enrichment_values.append(selected["roi_enrichment"])
        distance_values.append(selected["peak_distance_to_roi_mm"])
    axes[0].plot(x, enrichment_values, marker="o", linewidth=2.0, color=color, label=roi_title)
    axes[1].plot(x, distance_values, marker="o", linewidth=2.0, color=color, label=roi_title)

axes[0].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
axes[0].set_ylabel("正 ERD/PMBR 的 ROI 富集")
axes[0].set_title("共同滤波器 DICS：ROI 富集")
axes[1].set_ylabel("正峰距 ROI（mm）")
axes[1].set_title("共同滤波器 DICS：峰位置")
for axis in axes:
    axis.set_xticks(x, dics_labels)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)

dics_figure_file = save_dir / "dics_mu_beta_erd_pmbr_comparison.png"
figure.savefig(dics_figure_file, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(figure)


#%%
# ==================== 16. 可选：渲染 pooled pial 俯视图 ====================
# 不画星号、圆或 ROI；P95 只控制显示，不参与数值评价。

if draw_brain:
    brain_jobs = []
    for component_name in component_order:
        for method_name in method_names:
            brain_jobs.append((
                f"time_{component_name}_{method_name}",
                time_pooled_maps[(component_name, method_name)],
            ))
    brain_jobs.append((
        "dics_beta_PMBR",
        np.maximum(dics_pooled_maps[("beta_15_30Hz", "PMBR")], 0.0),
    ))

    for brain_name, brain_map in brain_jobs:
        brain_peak = max(float(np.max(brain_map)), np.finfo(float).eps)
        brain_normalized = brain_map / brain_peak
        brain_floor = float(np.percentile(brain_normalized, brain_display_percentile))
        if brain_floor >= 1.0:
            brain_floor = 0.0
        brain_display = np.where(
            brain_normalized >= brain_floor,
            brain_normalized,
            0.0,
        )
        brain_stc = mne.SourceEstimate(
            brain_display[:, None],
            vertices=source_vertices,
            tmin=0.0,
            tstep=1.0,
            subject=sample_subject,
        )
        brain = brain_stc.plot(
            subject=sample_subject,
            subjects_dir=subjects_dir,
            surface="pial",
            hemi="both",
            views="dorsal",
            initial_time=0.0,
            time_viewer=False,
            show_traces=False,
            time_label=None,
            colormap="inferno",
            clim={
                "kind": "value",
                "lims": [brain_floor, (brain_floor + 1.0) / 2.0, 1.0],
            },
            smoothing_steps=5,
            transparent=True,
            background="white",
            foreground="black",
            cortex="classic",
            colorbar=True,
            size=(1000, 760),
            backend="pyvistaqt",
            brain_kwargs={"show": False, "theme": "light"},
        )
        brain.plotter.camera.zoom(0.80)
        brain.plotter.render()
        brain.save_image(save_dir / f"brain_{brain_name}_pial_dorsal.png")
        brain.close()


#%%
# ==================== 17. 保存运行说明并做最后检查 ====================

oaster_diagnostic_rows = []
for run_input in run_inputs:
    for lock_name in ("response", "stimulus"):
        information_key = f"{lock_name}_oaster_information"
        if information_key not in run_input:
            continue
        lock_components = [
            name for name, (item_lock, _, _) in time_windows.items()
            if item_lock == lock_name
        ]
        for component_name, window_diagnostic in zip(
            lock_components,
            run_input[information_key]["windows"],
        ):
            oaster_diagnostic_rows.append({
                "subject": subject,
                "run": run_input["run"],
                "component": component_name,
                "temporal_rank": window_diagnostic["temporal_rank"],
                "selected_template_count": len(window_diagnostic["selected_templates"]),
                "selected_templates_json": json.dumps(
                    window_diagnostic["selected_templates"], ensure_ascii=False
                ),
                "surface_ebic_deltas": ";".join(
                    f"{value:.12g}"
                    for value in window_diagnostic["surface_ebic_deltas"]
                ),
                "require_one": run_input[information_key]["require_one"],
            })
oaster_diagnostic_table = pd.DataFrame(oaster_diagnostic_rows)
oaster_diagnostic_table.to_csv(
    save_dir / "oaster_window_diagnostics.csv",
    index=False,
    encoding="utf-8-sig",
)

run_qc_table = pd.DataFrame([
    {
        "subject": subject,
        "run": run_input["run"],
        "raw_file": str(run_input["raw_file"]),
        "meg_channels": run_input["meg_channels"],
        "eeg_channels_before_qc": run_input["eeg_channels_before_qc"],
        "auto_bad_eeg": ";".join(run_input["auto_bad_eeg"]),
        "response_epochs": len(run_input["response_epochs"]),
        "stimulus_epochs": len(run_input["stimulus_epochs"]),
        "dev_head_t_translation_m": ";".join(
            f"{value:.8f}"
            for value in run_input["response_evoked"].info["dev_head_t"]["trans"][:3, 3]
        ),
    }
    for run_input in run_inputs
])
run_qc_table.to_csv(save_dir / "run_qc.csv", index=False, encoding="utf-8-sig")

summary = {
    "dataset": "ds006035",
    "subject": subject,
    "runs": list(runs),
    "task": "right median nerve stimulus followed by left index-finger lift",
    "pairing": "strict one-to-one event32 -> first event16 before next event32",
    "accepted_reaction_time_seconds": [minimum_rt, maximum_rt],
    "artifact_interpolation_seconds": [-0.002, 0.008],
    "channels": "EEG + 102 magnetometers + 204 gradiometers",
    "run_policy": (
        "run-specific forward/inverse because dev_head_t differs; "
        "trial-weighted aggregation only after projection to the shared sample source grid"
    ),
    "response_baseline_seconds": list(response_baseline),
    "stimulus_baseline_seconds": list(stimulus_baseline),
    "time_domain_methods": list(method_names),
    "oaster_require_one": False,
    "time_windows": time_windows,
    "stimulus_m1_overlap_with_response_mf_fraction": stimulus_m1_overlap_fraction,
    "dics_bands_hz": {
        name: [float(frequencies.min()), float(frequencies.max())]
        for name, frequencies in dics_frequencies.items()
    },
    "dics_erd_window_seconds": list(dics_erd_window),
    "dics_pmbr_window_seconds": list(dics_pmbr_window),
    "dics_sample_rate_hz": dics_sample_rate,
    "roi_policy": "right precentral and right postcentral read only after every inverse",
    "anatomy_limit": "sample template anatomy; not individual anatomical localization",
    "elapsed_seconds": float(time.perf_counter() - all_started),
}
(save_dir / "run_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

assert len(run_inputs) == 3
assert all(run_input["meg_channels"] == 306 for run_input in run_inputs)
assert response_baseline == (-1.50, -1.00)
assert not time_metric_table["roi_used_before_inverse"].any()
assert not dics_metric_table["roi_used_before_inverse"].any()
assert ("beta_15_30Hz", "PMBR") in dics_pooled_maps

print("事件配对：", save_dir / "event_pairing_report.csv")
print("反应时：", save_dir / "reaction_times.csv")
print("时域指标：", time_metric_file)
print("DICS 指标：", dics_metric_file)
print("OASTER 窗诊断：", save_dir / "oaster_window_diagnostics.csv")
print("时域比较图：", time_figure_file)
print("DICS 比较图：", dics_figure_file)
print("总耗时（秒）：", round(summary["elapsed_seconds"], 1))
