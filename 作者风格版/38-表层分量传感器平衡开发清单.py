"""# %% 只生成双表层源的“分量公平”开发清单；旧压力测试不改。"""

# %% 读取已经完成深浅传感器平衡的 -10/-10 dB 清单。
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
source = protocol_dir / "development_sensor_balanced_m10_m10.json"
cases = [dict(case) for case in json.loads(source.read_text(encoding="utf-8"))]
assert len(cases) == 5
selected = [case for case in cases if len(case["surface_centers"]) == 2]
assert [case["case_id"].split("-")[3] for case in selected] == ["01", "04"]

# %% 两个表层分量在生成噪声协方差白化后的 EEG+MEG 联合空间取相同能量。
for case in selected:
    case["surface_component_sensor_balance"] = True
    case["component_balance_revision"] = "surface_components_and_layers_joint_sensor_v2"

outputs = {
    "development_component_balanced_m10_m10.json": cases,
    "development_component_balanced_m10_m10_cases00_02_03.json":
        [case for case in cases if len(case["surface_centers"]) < 2],
    "development_component_balanced_m10_m10_cases01_04.json": selected,
    "development_component_balanced_m10_m10_case00.json": cases[:1],
    "development_component_balanced_m10_m10_case01.json": selected[:1],
    "development_component_balanced_m10_m10_case02.json": cases[2:3],
    "development_component_balanced_m10_m10_case03.json": cases[3:4],
    "development_component_balanced_m10_m10_case04.json": selected[1:],
}
for name, subset in outputs.items():
    output = protocol_dir / name
    payload = json.dumps(subset, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if output.exists():
        assert output.read_text(encoding="utf-8") == payload, "已有清单与固定规则不一致"
    else:
        output.write_text(payload, encoding="utf-8")
    print(name, len(subset), hashlib.sha256(output.read_bytes()).hexdigest())
