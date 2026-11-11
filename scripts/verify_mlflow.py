"""Independently compare MLflow values and copied artifacts with source evidence."""
from collections import Counter
import json
from pathlib import Path
import sys

from mlflow_sync import Importer, ROOT, read_json, scalar_metrics, sha, state_status


def main():
    importer = Importer(ROOT / "mlflow_store")
    checks = 0
    failures = []
    counts = Counter()
    curves = {}
    archived_incomplete = []

    def check(value, message):
        nonlocal checks
        checks += 1
        if not value:
            failures.append(message)

    for key, entry in importer.index.items():
        rid = entry["mlflow_run_id"]
        run = importer.client.get_run(rid)
        if key.endswith("_aggregate"):
            check(run.info.status == "FINISHED", f"aggregate status {key}")
            continue
        path = ROOT / key
        summary = read_json(path / "summary.json", {})
        manifest = read_json(path / "manifest.json", {})
        counts[entry["lab"]] += 1
        check(run.data.tags.get("source.RUN_ID") == path.name, f"RUN_ID {key}")
        check(run.data.tags.get("import.complete") == "true", f"complete {key}")
        status = summary.get("status", manifest.get("status", "UNKNOWN"))
        check(run.info.status == state_status(status), f"status {key}")
        check(run.data.tags.get("source.git_commit") == str(manifest.get("git_commit", "unrecorded")), f"commit {key}")
        for name, value in scalar_metrics(summary).items():
            actual = run.data.metrics.get("result." + name)
            check(actual == value, f"metric {key}/{name}: {actual} != {value}")
        # MLflow copied bytes must match the immutable originals, including full logs.
        provenance = Path(importer.client.download_artifacts(rid, "provenance/source_files.json"))
        expected = read_json(provenance)["files"]
        for name, digest in expected.items():
            copy = Path(importer.client.download_artifacts(rid, "original/" + name))
            check(sha(copy) == sha(path / name) == digest, f"artifact checksum {key}/{name}")
        lab = entry["lab"]
        if lab not in curves and (path / "steps.jsonl").exists():
            rows = [json.loads(x) for x in (path / "steps.jsonl").read_text("utf-8-sig").splitlines() if x.strip()]
            if rows and all("train_loss" in r for r in [rows[0], rows[-1]]):
                history = importer.client.get_metric_history(rid, "step.train_loss")
                pairs = {(m.step, m.value) for m in history}
                for row in [rows[0], rows[-1]]:
                    check((row["global_step"], row["train_loss"]) in pairs, f"curve endpoints {key}")
                curves[lab] = {"source": key, "original": len(rows), "displayed": len(history)}
    source_count = len(list((ROOT / "runs").rglob("summary.json")))
    check(sum(counts.values()) == source_count, "all source summaries accounted for")
    # Reversible cleanup of failed import attempts, not original failed experiments.
    for exp in importer.client.search_experiments():
        for run in importer.client.search_runs([exp.experiment_id], max_results=5000):
            if run.data.tags.get("import.complete") == "false" and run.info.status == "FAILED":
                importer.client.set_tag(run.info.run_id, "import.failure_scope", "import_attempt_only")
                importer.client.delete_run(run.info.run_id)
                archived_incomplete.append(run.info.run_id)
    audit = {"checks": checks, "failures": failures, "runs_by_lab": dict(sorted(counts.items())),
             "source_summary_count": source_count, "aggregate_count": sum(k.endswith("_aggregate") for k in importer.index),
             "curve_endpoint_checks": curves, "archived_failed_import_attempts": archived_incomplete,
             "note": "Original failed/interrupted experiments retained; only incomplete import attempts soft-deleted."}
    out = ROOT / "docs/environment/mlflow_verification.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
