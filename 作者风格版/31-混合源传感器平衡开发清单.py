"""# %% 只生成新的开发清单；旧压力测试、校准和验证清单都不改。"""

# %% 路径和固定规则。
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "results/erp_whole_head/adaptive_v6/protocol/development_manifest.json"
protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
cases = json.loads(source.read_text(encoding="utf-8"))
selected = [dict(case) for case in cases
            if case["eeg_snr_db"] == -10 and case["meg_snr_db"] == -10]
assert len(selected) == 5 and len({case["case_id"] for case in selected}) == 5

# %% 混合病例把原来的“源空间幅度比”改为“噪声归一化传感器幅度比”。
# 纯表层、纯深层病例不需要比例，因此观测保持原定义。
for case in selected:
    if case["deep_index"] is not None and case["surface_centers"]:
        case["deep_surface_sensor_amplitude_ratio"] = case["deep_surface_ratio"]
    case["component_balance_revision"] = "joint_eeg_meg_noise_normalized_sensor_amplitude_v1"

outputs = {
    "development_sensor_balanced_m10_m10.json": selected,
    "development_sensor_balanced_m10_m10_cases00_01.json": selected[:2],
    "development_sensor_balanced_m10_m10_cases02_04.json": selected[2:],
    "development_sensor_balanced_m10_m10_case01.json": selected[1:2],
    "development_sensor_balanced_m10_m10_case04.json": selected[4:5],
}
records = []
for name, subset in outputs.items():
    output = protocol_dir / name
    payload = json.dumps(subset, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if output.exists():
        assert output.read_text(encoding="utf-8") == payload, "已有清单与固定规则不一致"
    else:
        output.write_text(payload, encoding="utf-8")
    records.append({"output": str(output), "cases": len(subset),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
print(json.dumps(records, ensure_ascii=False, indent=2))
