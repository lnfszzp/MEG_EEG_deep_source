"""# %% 生成四组SNR的分量平衡开发清单；正式校准和验证清单不改。"""

# %% 读取原20例开发设计，病例位置、噪声种子和SNR保持不变。
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
source = protocol_dir / "development_manifest.json"
cases = [dict(case) for case in json.loads(source.read_text(encoding="utf-8"))]
assert len(cases) == 20 and len({case["case_id"] for case in cases}) == 20

# %% 混合源按白化后EEG+MEG联合传感器幅度平衡；两个表层分量也取相同传感器能量。
for case in cases:
    if case["deep_index"] is not None and case["surface_centers"]:
        case["deep_surface_sensor_amplitude_ratio"] = case["deep_surface_ratio"]
    if len(case["surface_centers"]) == 2:
        case["surface_component_sensor_balance"] = True
        case["component_balance_revision"] = "surface_components_and_layers_joint_sensor_v2"
    else:
        case["component_balance_revision"] = "joint_eeg_meg_noise_normalized_sensor_amplitude_v1"

outputs = {
    "development_component_balanced_all_snr.json": cases,
    "development_component_balanced_all_snr_mixed_only.json":
        [case for case in cases if case["deep_index"] is not None and case["surface_centers"]],
    "development_component_balanced_new_snr_mixed_only.json":
        [case for case in cases if case["deep_index"] is not None and case["surface_centers"]
         and (case["eeg_snr_db"], case["meg_snr_db"]) != (-10, -10)],
}
for name, subset in outputs.items():
    output = protocol_dir / name
    payload = json.dumps(subset, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if output.exists():
        assert output.read_text(encoding="utf-8") == payload, "已有清单与固定规则不一致"
    else:
        output.write_text(payload, encoding="utf-8")
    print(name, len(subset), hashlib.sha256(output.read_bytes()).hexdigest())
