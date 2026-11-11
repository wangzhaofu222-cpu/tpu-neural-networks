"""Run the existing training entry point and synchronize new evidence afterwards.

The training process uses the original interpreter; MLflow uses its own environment.
All arguments are forwarded to src.train, including --config, --set and --resume.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    training = ROOT / ".conda-env" / "python.exe"
    tracking = ROOT / ".mlflow-env" / "Scripts" / "python.exe"
    if not training.exists() or not tracking.exists():
        raise SystemExit("Both the original training environment and isolated MLflow environment are required.")
    before = {p.parent for p in (ROOT / "runs").rglob("manifest.json")}
    result = subprocess.run([str(training), "-X", "utf8", "-m", "src.train", *sys.argv[1:]], cwd=ROOT)
    after = {p.parent for p in (ROOT / "runs").rglob("manifest.json")}
    new = sorted(after - before)
    args = [str(tracking), "-X", "utf8", str(ROOT / "scripts/mlflow_sync.py"), "--no-aggregate"]
    for path in new:
        args.extend(["--run", str(path)])
    if new:
        synced = subprocess.run(args, cwd=ROOT)
        if synced.returncode:
            print("Training records remain saved; MLflow sync failed. Run Sync_MLflow.cmd to retry.", file=sys.stderr)
            if not result.returncode:
                raise SystemExit(synced.returncode)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
