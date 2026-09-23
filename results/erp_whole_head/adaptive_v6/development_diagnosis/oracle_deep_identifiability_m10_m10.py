"""只读 oracle 诊断：开发集 -10/-10 dB 深源病例，不读取校准或验证集。"""

# %% 固定环境与输入。真值只在生成观测后用于事后可辨识性上限分析。
import hashlib
import json
import os
from pathlib import Path
from statistics import NormalDist
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
import run_erp_whole_head_matrix as original

manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_manifest.json"
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
assert manifest.with_suffix(".json.sha256").read_text(encoding="ascii").split()[0] == manifest_hash
all_cases = json.loads(manifest.read_text(encoding="utf-8"))
cases = [case for case in all_cases if case["panel"] == "erp_v6_development"
         and case["eeg_snr_db"] == case["meg_snr_db"] == -10
         and case["configuration_number"] in (2, 3, 4)]
assert [case["configuration_number"] for case in cases] == [2, 3, 4]
assert all(case["deep_index"] is not None for case in cases)
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
n_surf = int(shared["n_surf"])
z_95 = NormalDist().inv_cdf(0.95)
matched_rows, geometry_rows = [], []

# %% 每例仅用试次基线协方差预处理；half20 与 combined40 共用同一训练白化器。
for case in cases:
    observation = prepare_trial_covariance_case(shared, case, seed_root=2026092206)
    truth = np.asarray(observation["truth"], dtype=float)
    deep_index = int(case["deep_index"])
    deep_wave = truth[deep_index, observation["active"]]
    assert deep_index >= n_surf and np.linalg.norm(deep_wave) > 0
    surface_truth = truth.copy()
    surface_truth[n_surf:] = 0
    surface_groups = [np.asarray(group, dtype=int) for group in observation["groups"]
                      if np.all(np.asarray(group) < n_surf)]
    counts = list(map(int, observation["metadata"]["retained_channels"]))
    assert len(counts) == 2 and sum(counts) == observation["gain"].shape[0]
    splits = np.cumsum([0, *counts])

    # EEG、MEG 单独看；联合结果使用算法训练阶段已确定的通道权重。
    representations = (
        ("eeg", np.arange(splits[0], splits[1]), np.ones(counts[0])),
        ("meg", np.arange(splits[1], splits[2]), np.ones(counts[1])),
        ("joint_weighted", np.arange(splits[2]), np.asarray(observation["channel_weights"], dtype=float)),
    )
    for representation, indices, weights in representations:
        gain = observation["gain"][indices] * weights[:, None]
        lead = gain[:, deep_index]
        lead_norm = float(np.linalg.norm(lead))
        assert lead_norm > 0

        # 空间条件角：真实表层 patch 的实际秩一空间权重产生的传感器 lead 子空间。
        surface_leads, surface_templates = [], []
        for group in surface_groups:
            source_component = truth[group][:, observation["active"]]
            left, singular, _right = np.linalg.svd(source_component, full_matrices=False)
            assert singular[0] > 0 and (len(singular) == 1 or singular[1] <= singular[0] * 1e-10)
            surface_leads.append(gain[:, group] @ left[:, 0])
            surface_templates.append((gain[:, group] @ source_component).ravel())

        deep_template = (lead[:, None] * deep_wave[None, :]).ravel()
        if surface_leads:
            surface_space = np.column_stack(surface_leads)
            basis = np.linalg.svd(surface_space, full_matrices=False)[0][:, :np.linalg.matrix_rank(surface_space)]
            spatial_residual = lead - basis @ (basis.T @ lead)
            spatial_ratio = float(np.linalg.norm(spatial_residual) / lead_norm)
            normalized = [column / np.linalg.norm(column) for column in surface_leads]
            spatial_condition = float(np.linalg.cond(np.column_stack([*normalized, lead / lead_norm])))

            time_space = np.column_stack(surface_templates)
            time_basis = np.linalg.svd(time_space, full_matrices=False)[0][:, :np.linalg.matrix_rank(time_space)]
            time_residual = deep_template - time_basis @ (time_basis.T @ deep_template)
            time_ratio = float(np.linalg.norm(time_residual) / np.linalg.norm(deep_template))
            normalized_time = [column / np.linalg.norm(column) for column in surface_templates]
            time_condition = float(np.linalg.cond(np.column_stack(
                [*normalized_time, deep_template / np.linalg.norm(deep_template)])))
            surface_energy = float(np.linalg.norm(gain @ surface_truth[:, observation["active"]]) ** 2)
        else:
            spatial_ratio = time_ratio = 1.0
            spatial_condition = time_condition = 1.0
            surface_energy = 0.0
        spatial_ratio = float(np.clip(spatial_ratio, 0, 1))
        time_ratio = float(np.clip(time_ratio, 0, 1))
        deep_energy = float(np.linalg.norm(deep_template) ** 2)
        geometry_rows.append({
            "case_id": case["case_id"], "scenario": case["scenario"],
            "representation": representation, "surface_component_count": len(surface_groups),
            "spatial_residual_ratio": spatial_ratio,
            "spatial_conditional_angle_deg": float(np.degrees(np.arcsin(spatial_ratio))),
            "spatial_normalized_design_condition": spatial_condition,
            "spatiotemporal_residual_ratio": time_ratio,
            "spatiotemporal_conditional_angle_deg": float(np.degrees(np.arcsin(time_ratio))),
            "spatiotemporal_normalized_design_condition": time_condition,
            "deep_sensor_active_energy": deep_energy,
            "surface_sensor_active_energy": surface_energy,
            "deep_to_surface_sensor_energy_ratio": deep_energy / surface_energy if surface_energy else None,
        })

        # 已知真实表层后消除；匹配滤波只检验真实深点和真实波形的 oracle 功效。
        observations = (
            ("half20_train", observation["training"][indices] * weights[:, None]),
            ("half20_check", observation["confirmation"][indices] * weights[:, None]),
            ("combined40", (observation["training"][indices] + observation["confirmation"][indices])
             * weights[:, None] / 2),
        )
        surface_signal = gain @ surface_truth
        q_energy = float(deep_wave @ deep_wave)
        q_sum = float(deep_wave.sum())
        baseline_count = int(observation["baseline"].sum())
        effective_q_norm = float(np.sqrt(q_energy + q_sum ** 2 / baseline_count))
        for observation_name, data in observations:
            residual = data - surface_signal
            projected_baseline = (lead / lead_norm) @ residual[:, observation["baseline"]]
            projected_noise_sd = float(np.std(projected_baseline, ddof=1))
            assert projected_noise_sd > 0 and np.isfinite(projected_noise_sd)
            numerator = float(np.sum((lead[:, None] * deep_wave[None, :])
                                     * residual[:, observation["active"]]))
            denominator = lead_norm * projected_noise_sd * effective_q_norm
            observed_z = numerator / denominator
            dprime = lead_norm * q_energy / (projected_noise_sd * effective_q_norm)
            channel_variance = np.var(residual[:, observation["baseline"]], axis=1, ddof=1)
            active_noise_energy = float(observation["active"].size * (1 + 1 / baseline_count)
                                        * channel_variance.sum())
            matched_rows.append({
                "case_id": case["case_id"], "scenario": case["scenario"],
                "representation": representation, "observation": observation_name,
                "matched_filter_noncentrality_lambda": float(dprime ** 2),
                "oracle_expected_dprime": float(dprime), "observed_matched_z": float(observed_z),
                "observed_amplitude_ratio": float(numerator / deep_energy),
                "baseline_projected_noise_sd": projected_noise_sd,
                "deep_active_energy_snr_db": float(10 * np.log10(deep_energy / active_noise_energy)),
                "gaussian_one_sided_power_at_alpha_0_05": float(NormalDist().cdf(dprime - z_95)),
                "observed_passes_one_sided_z_0_05": bool(observed_z > z_95),
                "deep_wave_active_sum": q_sum, "deep_wave_active_norm": float(np.sqrt(q_energy)),
            })

# %% 机器可读结论与自检；不把 oracle 阈值接回算法。
assert len(matched_rows) == 27 and len(geometry_rows) == 9
assert all(np.isfinite(row["oracle_expected_dprime"]) for row in matched_rows)
mixed_combined = [row for row in matched_rows if row["scenario"] != "deep_only"
                  and row["observation"] == "combined40"]
mixed_geometry = [row for row in geometry_rows if row["scenario"] != "deep_only"]
power_by_representation = {}
for representation in ("eeg", "meg", "joint_weighted"):
    combined_rows = [row for row in mixed_combined if row["representation"] == representation]
    check_rows = [row for row in matched_rows if row["scenario"] != "deep_only"
                  and row["representation"] == representation and row["observation"] == "half20_check"]
    power_by_representation[representation] = {
        "combined40_min_dprime": min(row["oracle_expected_dprime"] for row in combined_rows),
        "combined40_min_power": min(row["gaussian_one_sided_power_at_alpha_0_05"] for row in combined_rows),
        "half20_check_min_dprime": min(row["oracle_expected_dprime"] for row in check_rows),
        "half20_check_min_power": min(row["gaussian_one_sided_power_at_alpha_0_05"] for row in check_rows),
    }
joint_power = power_by_representation["joint_weighted"]
min_time_angle = min(row["spatiotemporal_conditional_angle_deg"] for row in mixed_geometry)
joint_geometry = [row for row in mixed_geometry if row["representation"] == "joint_weighted"]
payload = {
    "scope": {
        "phase": "development only", "manifest": str(manifest.relative_to(root)),
        "manifest_sha256": manifest_hash, "case_ids": [case["case_id"] for case in cases],
        "snr_pair_db": [-10, -10], "calibration_read": False, "validation_read": False,
        "preprocessing": "prepare_trial_covariance_case; same training whitener for half20 and combined40",
        "oracle_use": "truth used only after observation generation to subtract exact surface and test the true deep point/waveform",
    },
    "formulas": {
        "matched_z": "<l*q, Y-G*J_surface> / (||l|| sigma_baseline sqrt(||q||^2+(sum q)^2/B))",
        "noncentrality": "lambda=dprime^2, dprime=||l|| ||q||^2/(sigma_baseline sqrt(||q||^2+(sum q)^2/B))",
        "energy_snr": "10log10(||l*q||^2 / (T_active*(1+1/B)*sum channel baseline variances))",
        "conditional_angle": "asin(||(I-P_surface)l||/||l||); spatiotemporal version vectorizes sensor-by-active-time templates",
    },
    "summary": {
        "mixed_power_by_representation": power_by_representation,
        "minimum_mixed_spatiotemporal_conditional_angle_deg": min_time_angle,
        "minimum_joint_spatial_conditional_angle_deg": min(
            row["spatial_conditional_angle_deg"] for row in joint_geometry),
        "joint_deep_to_surface_sensor_energy_ratio_range": [
            min(row["deep_to_surface_sensor_energy_ratio"] for row in joint_geometry),
            max(row["deep_to_surface_sensor_energy_ratio"] for row in joint_geometry)],
        "combined40_joint_oracle_low_power": bool(joint_power["combined40_min_power"] < 0.5),
        "half20_check_joint_below_0_8_power": bool(joint_power["half20_check_min_power"] < 0.8),
        "interpretation": "联合 combined40 oracle 并非低功效，但独立 half20 确认功效不足 0.8；EEG 单独接近低功效。深源与真实表层时空子空间可分，困难主要来自深源传感器能量仅约为表层的百分之一，以及分半、盲搜索和正则化损失。",
        "warning": "oracle 是上限诊断，不是可部署算法；不得据此选择测试阈值或宣称盲检性能。",
    },
    "matched_filter_rows": matched_rows,
    "geometry_rows": geometry_rows,
}
output = Path(__file__).with_suffix(".json")
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
print(output)
