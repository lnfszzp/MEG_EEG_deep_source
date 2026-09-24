"""# %% 冻结四病例开发探针；只检验多轮 MM 稳定化，不用于正式验收。"""

# %% 1. 固定病例和预声明验收，不读取反演结果。
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()

protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
source = protocol_dir / "development_localization_v5_new_positions.json"
manifest_path = protocol_dir / "development_localization_v5_mm_floor_probe.json"
report_path = protocol_dir / "V5_MM_FLOOR_PROBE.md"
source_cases = json.loads(source.read_text(encoding="utf-8"))
selected_numbers = [0, 2, 4, 10]
manifest = [case for case in source_cases if case["case_number"] in selected_numbers]
if [case["case_number"] for case in manifest] != selected_numbers:
    raise ValueError("四病例探针必须依次为 00、02、04、10")
if [case["scenario"] for case in manifest] != [
        "surface_only", "deep_only", "deep_plus_two_surface", "surface_only"]:
    raise ValueError("四病例探针角色已变化")

settings = {
    "solver_kind": "admm",
    "mrf_strength": .8,
    "outer_iterations": 20,
    "max_inner_retries": 20,
    "epsilon_fraction": .05,
    "tolerance": .001,
    "outer_tolerance": .01,
    "surface_reweight_floor": .5,
    "deep_reweight_floor": .5,
    "edge_weight_floor": .5,
}
source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
report_text = f"""# v5 多轮 MM 四病例开发探针

本探针在 v5 新位置开发失败后按角色选取，只用于根因验证，不能作为校准或正式验收。

- 病例：00/10 两个纯表层误报位置，02 真深源，04 深层+双表层漏报位置。
- 源清单 SHA256：`{source_sha256}`。
- 唯一求解设置：`{json.dumps(settings, ensure_ascii=False, sort_keys=True)}`。
- 不运行参数网格，不复用旧正式 validation 阈值作为验收线。
- 预声明主条件：`max(T00, T10) < min(T02, T04)`。
- 数值条件：四例 H0/H1 均完成多轮 MM，最终新权重 stationarity gap `<= 0.001`。
- 若主条件失败，停止该候选；不得逐病例选择设置。
"""


# %% 2. 首次写入后只能核对，主文件和旁车一起冻结。
outputs = {manifest_path: manifest_text.encode("utf-8"), report_path: report_text.encode("utf-8")}
if args.check:
    for path, payload in outputs.items():
        digest = hashlib.sha256(payload).hexdigest()
        sidecar = Path(str(path) + ".sha256")
        expected = f"{digest}  {path.name}\n".encode("ascii")
        if not path.is_file() or path.read_bytes() != payload or \
                not sidecar.is_file() or sidecar.read_bytes() != expected:
            raise ValueError(f"冻结探针已变化：{path}")
else:
    for path in outputs:
        if path.exists() or Path(str(path) + ".sha256").exists():
            raise FileExistsError(f"拒绝覆盖：{path}")
    for path, payload in outputs.items():
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        Path(str(path) + ".sha256").write_bytes(f"{digest}  {path.name}\n".encode("ascii"))

print(json.dumps({"manifest": str(manifest_path), "case_numbers": selected_numbers,
                  "settings": settings,
                  "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()},
                 ensure_ascii=False, indent=2))
