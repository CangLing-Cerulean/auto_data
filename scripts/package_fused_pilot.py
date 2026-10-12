"""Package only this no-API pilot's reproducible research assets."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(root):
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    core = Path(protocol["base_root"]) / "scripts/run_research.py"
    if sha(core) != protocol["core_sha256"]:
        raise ValueError("Original model source identity changed")
    (root / "core_reference.py").write_bytes(core.read_bytes())
    files = sorted([p for p in root.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".jsonl", ".md")
                    and p.name not in ("artifact_manifest.json", "archive_receipt.json")]
                   + list((root / "runs").glob("*/*")) + list((root / "tasks").glob("*/audit.json")))
    files = [p for p in files if p.is_file()]
    if any(p.suffix not in (".py", ".json", ".jsonl", ".md", ".npz", ".pt") for p in files):
        raise ValueError("Unexpected archive member")
    manifest = {"scope": "this_pilot_only; no_API_logs_credentials_or_prepared_sensor_arrays",
        "base_dataset_id": protocol["dataset_id"], "base_manifest_sha256": protocol["manifest_sha256"],
        "prepared_arrays": "reconstruct with frozen runner and original base dataset; source mappings included in run npz files",
        "files": [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]}
    (root / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = root / "fused-pilot-20261011-sol-high-r1.tar.gz"
    if archive.exists():
        raise ValueError("Existing archive will not be replaced")
    with tarfile.open(archive, "w:gz") as tar:
        for p in files + [root / "artifact_manifest.json"]:
            tar.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
    receipt = {"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive), "member_count": len(files) + 1}
    (root / "archive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
