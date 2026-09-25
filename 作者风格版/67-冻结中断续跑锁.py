"""# %% 冻结一次基础设施中断后的严格病例前缀；不运行或重算任何病例。"""

# %% 1. 固定旧锁、中断输出、快照和新恢复锁。
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if not __debug__:
    raise RuntimeError("中断续跑锁禁止使用 python -O")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import protocol
import run_erp_whole_head_matrix as original

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="只逐字节核验，不创建或覆盖")
args = parser.parse_args()

adaptive_root = root / "results/erp_whole_head/adaptive_v6"
protocol_dir = adaptive_root / "protocol"
old_lock_path = protocol_dir / "hierarchical_v1_execution_lock.json"
old_lock_sidecar = Path(str(old_lock_path) + ".sha256")
output_path = protocol_dir / "hierarchical_v1_recovery_execution_lock.json"
output_sidecar = Path(str(output_path) + ".sha256")
gate_output = adaptive_root / "formal_hierarchical_v1_gate_calibration"
marker_path = protocol_dir / "hierarchical_v1_gate_calibration_manifest_consumed.json"
metadata_path = gate_output / "metadata.json"
evidence_path = gate_output / "evidence.csv"
metadata_snapshot = gate_output / "metadata.interrupted_prefix.json"
evidence_snapshot = gate_output / "evidence.interrupted_prefix.csv"
snapshot_paths = {"metadata": metadata_snapshot, "evidence": evidence_snapshot}


# %% 2. 旧执行锁、marker、28例完整前缀和“尚无阈值/封存”必须逐项成立。
old_lock_bytes = old_lock_path.read_bytes()
old_lock_sha256 = hashlib.sha256(old_lock_bytes).hexdigest()
if not old_lock_sidecar.is_file() or old_lock_sidecar.read_bytes() != \
        f"{old_lock_sha256}  {old_lock_path.name}\n".encode("ascii"):
    raise ValueError("旧执行锁或旁车已变化")
old_lock = json.loads(old_lock_bytes)
if old_lock.get("schema_version") != 1 or \
        old_lock.get("protocol") != "erp-v6-hierarchical-v1-execution" or \
        old_lock.get("status") != "algorithm_frozen_calibration_allowed" or \
        old_lock.get("formal_cases_run_by_generator") is not False:
    raise ValueError("前驱不是未修改的 hierarchical-v1 执行锁")

gate_stage = old_lock.get("manifests", {}).get("gate-calibration", {})
manifest_lock = gate_stage.get("manifest", {})
manifest_path = root / manifest_lock.get("path", "")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if len(manifest) != 57 or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != \
        manifest_lock.get("sha256"):
    raise ValueError("gate manifest 已变化或病例数不是57")
if gate_output.resolve() != (root / gate_stage.get("output_dir", "")).resolve() or \
        marker_path.resolve() != (root / gate_stage.get("consumed_marker", "")).resolve():
    raise ValueError("中断输出或 marker 不是旧锁登记路径")

terminal_paths = [
    gate_output / "completion.json",
    gate_output / "frozen_gate_calibration.json",
    gate_output / "frozen_gate_calibration.json.sha256",
    gate_output / "calibration_seal.json",
    gate_output / "calibration_seal.json.sha256",
    gate_output / "invalid_calibration.json",
]
if any(path.exists() for path in terminal_paths):
    raise FileExistsError("已有阈值、completion、invalid 或 seal，禁止建立中断续跑锁")
for later_stage in ("surface-calibration", "validation"):
    specification = old_lock["manifests"][later_stage]
    later_marker = root / specification["consumed_marker"]
    later_output = root / specification["output_dir"]
    if later_marker.exists() or later_output.exists():
        raise FileExistsError(f"后续 stage 已开始，禁止恢复 gate：{later_stage}")

marker_bytes = marker_path.read_bytes()
marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
marker = json.loads(marker_bytes)
if marker.get("stage") != "gate-calibration" or \
        marker.get("execution_lock_sha256") != old_lock_sha256 or \
        marker.get("manifest_sha256") != manifest_lock.get("sha256") or \
        Path(marker.get("output", "")).resolve() != gate_output.resolve() or \
        marker.get("prior_calibration_sha256") != {}:
    raise ValueError("中断 marker 没有绑定旧锁/gate manifest/output")

metadata_bytes = metadata_path.read_bytes()
metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
metadata = json.loads(metadata_bytes)
if metadata.get("stage") != "gate-calibration" or \
        metadata.get("execution_lock_sha256") != old_lock_sha256 or \
        metadata.get("manifest_sha256") != manifest_lock.get("sha256") or \
        metadata.get("consumed_marker_sha256") != marker_sha256 or \
        metadata.get("case_ids") != [case["case_id"] for case in manifest]:
    raise ValueError("中断 metadata 没有完整绑定旧锁/marker/manifest")

evidence_bytes = evidence_path.read_bytes()
evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
with evidence_path.open("r", encoding="utf-8-sig", newline="") as stream:
    evidence_reader = csv.DictReader(stream)
    evidence = list(evidence_reader)
expected_columns = [
    "case_id", "scenario", "eeg_snr_db", "meg_snr_db", "gate_score",
    "gate_eeg_score", "gate_meg_score", "gate_threshold",
    "deep_present_decision", "selected_deep_index", "secondary_surface_score",
    "surface_threshold", "surface_present_decision", "null_converged",
    "full_converged", "selection_surface_converged", "final_surface_converged",
    "all_converged", "status", "error", "elapsed_seconds",
]
prefix_count = len(evidence)
if evidence_reader.fieldnames != expected_columns or prefix_count != 28 or \
        [row.get("case_id") for row in evidence] != [
            case["case_id"] for case in manifest[:prefix_count]] or \
        any(row.get("status") != "ok" or row.get("all_converged") != "1" or
            row.get("error") not in {"", None} for row in evidence):
    raise ValueError("只允许冻结当前28例全部ok、全部收敛的严格manifest前缀")

case_details = {}
for row, case in zip(evidence, manifest):
    detail_path = gate_output / (case["case_id"] + ".json")
    detail_bytes = detail_path.read_bytes()
    detail = json.loads(detail_bytes)
    if detail.get("case_id") != case["case_id"] or \
            detail.get("stage") != "gate-calibration" or \
            detail.get("complete") is not True or \
            detail.get("truth_used_by_decisions") is not False or \
            float(detail.get("gate_score")) != float(row["gate_score"]):
        raise ValueError(f"中断病例详情不可复核：{case['case_id']}")
    case_details[case["case_id"]] = {
        "path": detail_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(detail_bytes).hexdigest(),
    }
case_json_names = {path.name for path in gate_output.glob("erp-v6-*.json")}
if case_json_names != {case["case_id"] + ".json" for case in manifest[:prefix_count]}:
    raise ValueError("中断目录含前缀之外的病例JSON或缺少已完成病例")


# %% 3. 新runner只能改变续跑编排；其余全部旧锁代码、环境和forward必须不变。
git_status = subprocess.run(
    ["git", "status", "--porcelain=v1", "--untracked-files=no"],
    cwd=root, capture_output=True, text=True, encoding="utf-8")
if git_status.returncode or git_status.stdout:
    raise RuntimeError("冻结恢复锁前 tracked 工作树和 index 必须完全 clean")
git_head_result = subprocess.run(
    ["git", "rev-parse", "--verify", "HEAD"], cwd=root,
    capture_output=True, text=True, encoding="ascii")
if git_head_result.returncode:
    raise RuntimeError("无法解析当前Git HEAD")
git_head = git_head_result.stdout.strip()

runner_relative = "作者风格版/65-层级盲定位正式运行.py"
generator_relative = "作者风格版/67-冻结中断续跑锁.py"
code_sha256 = dict(old_lock.get("code_sha256", {}))
for relative, expected in code_sha256.items():
    path = root / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if relative != runner_relative and actual != expected:
        raise ValueError(f"除续跑runner外的冻结代码发生变化：{relative}")
code_sha256[runner_relative] = hashlib.sha256(
    (root / runner_relative).read_bytes()).hexdigest()
code_sha256[generator_relative] = hashlib.sha256(
    (root / generator_relative).read_bytes()).hexdigest()
tracked = subprocess.run(
    ["git", "ls-files", "--error-unmatch", "--", runner_relative, generator_relative],
    cwd=root, capture_output=True, text=True, encoding="utf-8")
if tracked.returncode:
    raise RuntimeError("65/67号脚本必须先提交到当前Git HEAD")

current_environment = original._environment_versions()
current_environment["python_debug"] = __debug__
if current_environment != old_lock.get("environment"):
    raise ValueError("恢复环境与旧执行锁不一致")
shared_lock = old_lock.get("shared", {})
shared = protocol.load_shared(root / shared_lock["generated_dir"],
                              Path(shared_lock["sample_path"]))
if original._shared_fingerprint(shared) != shared_lock.get("fingerprint"):
    raise ValueError("恢复forward与旧执行锁不一致")


# %% 4. 原字节快照、病例哈希和前驱锁共同授权只追加剩余29例。
snapshot_payloads = {
    metadata_snapshot: metadata_bytes,
    evidence_snapshot: evidence_bytes,
}
snapshots = {
    name: {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(snapshot_payloads[path]).hexdigest(),
        "source_path": (metadata_path if name == "metadata" else evidence_path
                        ).relative_to(root).as_posix(),
    }
    for name, path in snapshot_paths.items()
}

recovery_lock = json.loads(old_lock_bytes)
recovery_lock.update({
    "status": "interrupted_prefix_recovery_allowed",
    "repository": {
        "git_head": git_head,
        "tracked_worktree_and_index_clean": True,
        "all_frozen_inputs_and_code_tracked_at_head": True,
    },
    "code_sha256": code_sha256,
    "generator_sha256": code_sha256[generator_relative],
    "interrupted_prefix_recovery": {
        "stage": "gate-calibration",
        "completed_prefix_count": prefix_count,
        "remaining_case_count": len(manifest) - prefix_count,
        "predecessor_execution_lock": old_lock_path.relative_to(root).as_posix(),
        "predecessor_execution_lock_sha256": old_lock_sha256,
        "consumed_marker": marker_path.relative_to(root).as_posix(),
        "consumed_marker_sha256": marker_sha256,
        "output_dir": gate_output.relative_to(root).as_posix(),
        "snapshots": snapshots,
        "case_details": case_details,
        "terminal_outputs_present": False,
        "algorithm_or_manifest_changed": False,
        "partial_scores_used_to_change_algorithm": False,
        "rule": "preserve the verified 28-case prefix byte-for-byte and append only manifest cases 28..56 with unchanged seeds and algorithm",
    },
})
output_payload = (json.dumps(
    recovery_lock, ensure_ascii=False, sort_keys=True,
    separators=(",", ":")) + "\n").encode("utf-8")
output_sha256 = hashlib.sha256(output_payload).hexdigest()
sidecar_payload = f"{output_sha256}  {output_path.name}\n".encode("ascii")

all_payloads = {
    **snapshot_payloads,
    output_path: output_payload,
    output_sidecar: sidecar_payload,
}
for snapshot_path, payload in snapshot_payloads.items():
    digest = hashlib.sha256(payload).hexdigest()
    all_payloads[Path(str(snapshot_path) + ".sha256")] = \
        f"{digest}  {snapshot_path.name}\n".encode("ascii")

if args.check:
    for path, payload in all_payloads.items():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"恢复锁或中断快照已变化：{path}")
else:
    occupied = [str(path) for path in all_payloads if path.exists()]
    if occupied:
        raise FileExistsError(f"拒绝覆盖已有恢复锁/快照：{occupied}")
    for path, payload in all_payloads.items():
        with path.open("xb") as stream:
            stream.write(payload)

print(json.dumps({
    "checked": bool(args.check),
    "path": str(output_path),
    "sha256": output_sha256,
    "predecessor_sha256": old_lock_sha256,
    "completed_prefix_count": prefix_count,
    "remaining_case_count": len(manifest) - prefix_count,
    "formal_threshold_present": False,
}, ensure_ascii=False, indent=2))
