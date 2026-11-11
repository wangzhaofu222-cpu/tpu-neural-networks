"""Check intended Git contents without displaying credential values."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    names = [n for n in names if n]
    secret = re.compile(r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    findings = []
    sizes = []
    forbidden = []
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        sizes.append((path.stat().st_size, name))
        if name.split("/")[0] in {".private", ".conda-env", ".conda-pkgs", ".mlflow-env", "mlflow_store", "data"}:
            forbidden.append(name)
        if path.suffix in {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".env"}:
            if secret.search(path.read_text("utf-8-sig", errors="replace")):
                findings.append(name)
    # Historical object sizes matter as well as the current working tree.
    objects = subprocess.run(["git", "rev-list", "--objects", "--all"], cwd=ROOT, capture_output=True, check=True).stdout
    history = subprocess.run(["git", "cat-file", "--batch-check=%(objecttype) %(objectsize) %(rest)"],
                             cwd=ROOT, input=objects, capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
    history_large = []
    for line in history.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[0] == "blob" and int(parts[1]) >= 100 * 1024**2:
            history_large.append({"bytes": int(parts[1]), "path": parts[2]})
    large = [{"bytes": n, "path": p} for n, p in sizes if n >= 100 * 1024**2]
    result = {"tracked_files": len(names), "tracked_bytes": sum(n for n, p in sizes),
              "credential_pattern_findings_paths_only": findings, "excluded_directory_violations": forbidden,
              "files_at_or_above_100MiB": large, "historical_blobs_at_or_above_100MiB": history_large,
              "scope": "High-confidence credential patterns in text source/config/docs; not a full historical secret audit."}
    output = ROOT / "docs/environment/git_delivery_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(findings or forbidden or large or history_large))


if __name__ == "__main__":
    main()
