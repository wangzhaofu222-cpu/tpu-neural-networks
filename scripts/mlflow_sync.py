"""Read-only import of authentic course evidence into a separate MLflow store.

Run with the isolated .mlflow-env interpreter; never imports the training stack.
Historical timestamps and statuses remain source metadata, not new training claims.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
TITLES = ["Reproducibility", "Trainability", "CNN vs ViT", "Attention and positions",
          "Self-supervised learning", "Graph networks", "EBM and Flow", "Diffusion",
          "Calibration and robustness", "Multimodal learning"]
KEY = re.compile(r"[^\w./ -]", re.UNICODE)
TEXT_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".csv", ".txt", ".md"}


def read_json(path, default=None):
    return json.loads(path.read_text("utf-8-sig")) if path.is_file() else default


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flatten(value, prefix="", depth=0):
    """Keep scalar metrics; arrays remain intact in the original artifact."""
    if isinstance(value, dict) and depth < 12:
        for key, child in value.items():
            name = KEY.sub("_", str(key))
            yield from flatten(child, f"{prefix}.{name}".strip("."), depth + 1)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        yield prefix[:240], value


def scalar_metrics(value, prefix=""):
    return {k: float(v) for k, v in flatten(value, prefix)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def sample_rows(rows, maximum):
    if len(rows) <= maximum:
        return rows
    # Select actual records evenly, including endpoints; no averaging/interpolation.
    return [rows[round(i * (len(rows) - 1) / (maximum - 1))] for i in range(maximum)]


def timestamp(value):
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except (TypeError, ValueError, AttributeError):
        return None


def source_files(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix in TEXT_SUFFIXES)


def run_lab(path, config, manifest):
    for part in path.relative_to(ROOT / "runs").parts[:-1]:
        if re.fullmatch(r"lab\d+(?:_verification)?", part):
            return int(re.search(r"\d+", part).group())
    return int(config.get("experiment", {}).get("lab", manifest.get("lab", 0)))


def report_aliases():
    aliases = {}
    for lab in range(10):
        base = ROOT / "reports" / f"lab{lab}"
        if lab:
            base /= "formal"
        for p in [base / "report_body.tex", *sorted((base / "tables").glob("*.tex"))]:
            if p.is_file():
                for line in p.read_text("utf-8-sig").splitlines():
                    m = re.search(r"\\hypertarget\{([^}]+)\}.*?([0-9a-f]{20}-[0-9a-f]{32})", line)
                    if m:
                        aliases.setdefault(m[2], []).append(f"Lab{lab}:{m[1]}")
    return {k: sorted(set(v)) for k, v in aliases.items()}


def state_status(original):
    upper = str(original).upper()
    if upper.startswith(("COMPLETED", "SUCCESS", "FINISHED")):
        return "FINISHED"
    if "FAIL" in upper or "ERROR" in upper:
        return "FAILED"
    return "KILLED"  # Interrupted or unknown completion; never assume success.


class Importer:
    def __init__(self, root, max_points=1000):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "tmp").mkdir(exist_ok=True)
        tempfile.tempdir = str(self.root / "tmp")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        from mlflow.tracking import MlflowClient
        self.uri = "sqlite:///" + (self.root / "mlflow.db").as_posix()
        self.client = MlflowClient(tracking_uri=self.uri)
        self.max_points = max_points
        self.aliases = report_aliases()
        self.experiments = {}
        self.index_path = self.root / "import_index.json"
        self.index = read_json(self.index_path, {})

    def experiment(self, lab):
        if lab not in self.experiments:
            name = f"Lab {lab} - {TITLES[lab]}"
            existing = self.client.get_experiment_by_name(name)
            self.experiments[lab] = existing.experiment_id if existing else self.client.create_experiment(
                name, artifact_location=(self.root / "artifacts" / f"lab{lab}").as_uri(),
                tags={"course": "TPU Neural Networks", "lab": str(lab)})
        return self.experiments[lab]

    def save_index(self):
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.index, ensure_ascii=False, indent=2), "utf-8")
        temporary.replace(self.index_path)

    def batch(self, rid, metrics):
        for start in range(0, len(metrics), 1000):
            self.client.log_batch(rid, metrics=metrics[start:start + 1000])

    def import_run(self, path):
        from mlflow.entities import Metric, Param
        path = path.resolve()
        config_path = path / "config.resolved.yaml"
        config = yaml.safe_load(config_path.read_text("utf-8-sig")) if config_path.exists() else {}
        config = config or {}
        manifest = read_json(path / "manifest.json", {})
        summary = read_json(path / "summary.json", {})
        lab = run_lab(path, config, manifest)
        source_id = path.name
        files = source_files(path)
        fingerprint = hashlib.sha256(json.dumps({p.name: sha(p) for p in files}, sort_keys=True).encode()).hexdigest()
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        old = self.index.get(key)
        if old and old["source_sha256"] == fingerprint and old.get("complete"):
            return {**old, "skipped": True}
        expid = self.experiment(lab)
        # Recover a completed import even if the process stopped before saving the index.
        matches = self.client.search_runs([expid],
            filter_string=f"tags.`source.path` = '{key}' AND tags.`source.sha256` = '{fingerprint}'",
            max_results=10)
        completed = next((r for r in matches if r.data.tags.get("import.complete") == "true"), None)
        if completed:
            result = {"mlflow_run_id": completed.info.run_id, "source_sha256": fingerprint,
                      "lab": lab, "complete": True, "skipped": True}
            self.index[key] = result
            self.save_index()
            return result
        started = timestamp(manifest.get("started_at"))
        original_status = summary.get("status", manifest.get("status", "UNKNOWN"))
        purpose = summary.get("purpose", manifest.get("purpose", config.get("experiment", {}).get("purpose", "unspecified")))
        model = config.get("model", {}).get("name", summary.get("model", "evaluation"))
        if isinstance(model, dict):
            model = model.get("name", "evaluation")
        aliases = self.aliases.get(source_id, [])
        label = aliases[0].split(":", 1)[1] if aliases else source_id[:12]
        kind = "training" if (path / "steps.jsonl").exists() else "evaluation_or_diagnostic"
        tags = {"source.RUN_ID": source_id, "source.path": key, "source.sha256": fingerprint,
                "source.status": str(original_status), "source.purpose": str(purpose),
                "source.kind": kind, "source.lab": str(lab), "source.model": str(model),
                "source.seed": str(manifest.get("seed", config.get("experiment", {}).get("seed", "unknown"))),
                "source.git_commit": str(manifest.get("git_commit", "unrecorded")),
                "source.started_at": str(manifest.get("started_at", "unrecorded")),
                "report.referenced": str(bool(aliases)).lower(), "report.aliases": ", ".join(aliases),
                "import.mode": "historical_evidence", "import.complete": "false",
                "import.timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "curve.policy": f"actual records, at most {self.max_points} per stream; full original JSONL in artifacts",
                "curve.timestamps": "source start time only; per-step wall times not reconstructed"}
        for name in ["source_RUN_ID", "parent_RUN_ID", "encoder_RUN_ID", "parent_run_id"]:
            value = summary.get(name, manifest.get(name))
            if value:
                tags[f"source.{name}"] = str(value)
        rid = self.client.create_run(expid, start_time=started, tags=tags,
            run_name=f"{label} | {model} | {purpose}").info.run_id
        try:
            params = [Param(k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v[:5900])
                      for k, v in flatten(config) if v is not None]
            for start in range(0, len(params), 100):
                self.client.log_batch(rid, params=params[start:start + 100])
            now = started if started is not None else int(time.time() * 1000)
            metrics = [Metric("result." + k, v, now, 0) for k, v in scalar_metrics(summary).items()]
            self.batch(rid, metrics)
            count = len(metrics)
            for name, prefix in [("steps.jsonl", "step"), ("metrics.jsonl", "epoch")]:
                f = path / name
                if not f.exists():
                    continue
                rows = [json.loads(line) for line in f.read_text("utf-8-sig").splitlines() if line.strip()]
                selected = sample_rows(rows, self.max_points)
                batch = []
                for i, row in enumerate(selected):
                    step = int(row.get("global_step", row.get("epoch", i)))
                    batch.extend(Metric(prefix + "." + k, v, now, step)
                                 for k, v in scalar_metrics(row).items()
                                 if k not in {"global_step", "epoch", "batch"})
                self.batch(rid, batch)
                count += len(batch)
                self.client.set_tag(rid, f"curve.{prefix}.original_records", str(len(rows)))
                self.client.set_tag(rid, f"curve.{prefix}.displayed_records", str(len(selected)))
            for f in files:
                self.client.log_artifact(rid, str(f), "original")
            inventory = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "bytes": p.stat().st_size,
                          "sha256": sha(p)} for p in sorted((path / "checkpoints").glob("*.pt"))]
            self.client.log_dict(rid, {"checkpoints": inventory,
                "note": "Original local model files; not duplicated into the tracking store."}, "provenance/checkpoints.json")
            self.client.log_dict(rid, {"files": {p.name: sha(p) for p in files},
                "source_path": key, "source_sha256": fingerprint}, "provenance/source_files.json")
            self.client.set_tag(rid, "import.complete", "true")
            end = timestamp(manifest.get("completed_at"))
            wall = manifest.get("wall_seconds", summary.get("training_seconds"))
            if end is None and started is not None and isinstance(wall, (int, float)) and math.isfinite(wall):
                end = started + int(wall * 1000)
            self.client.set_tag(rid, "source.end_time_basis", "recorded_or_recorded_duration" if end else "unrecorded")
            self.client.set_terminated(rid, status=state_status(original_status), end_time=end or started or now)
        except Exception:
            self.client.set_tag(rid, "import.failed", "true")
            self.client.set_terminated(rid, status="FAILED")
            raise
        result = {"mlflow_run_id": rid, "source_sha256": fingerprint, "lab": lab,
                  "complete": True, "metrics_logged": count, "source_status": original_status,
                  "report_referenced": bool(aliases), "checkpoint_count": len(inventory)}
        self.index[key] = result
        self.save_index()
        return result

    def import_lab_artifacts(self, lab):
        from mlflow.entities import Metric
        base = ROOT / "experiments" / f"lab{lab}"
        report = ROOT / "reports" / f"lab{lab}"
        plots = report / "formal" / "figures" if lab else report / "figures"
        files = [p for p in base.glob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES]
        files += [p for p in plots.glob("*") if p.suffix.lower() in {".png", ".svg"}]
        pdf = report / f"Lab{lab}_Report.pdf"
        if pdf.exists():
            files.append(pdf)
        if lab == 0:
            files += [p for p in [report / "run_registry.json", report / "acceptance.json"] if p.exists()]
        key = f"lab{lab}_aggregate"
        digest = hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in files}, sort_keys=True).encode()).hexdigest()
        if self.index.get(key, {}).get("source_sha256") == digest:
            return self.index[key]
        rid = self.client.create_run(self.experiment(lab), tags={"source.kind": "aggregate_analysis",
            "import.mode": "historical_evidence", "source.lab": str(lab), "report.referenced": "true",
            "mlflow.note.content": "Сводные результаты, графики и отчёт. 汇总、图表与报告；非新增训练。"},
            run_name=f"00 - Lab {lab} results, figures and report").info.run_id
        for f in files:
            folder = "figures" if f.parent == plots else "report" if f.suffix == ".pdf" else "analysis"
            self.client.log_artifact(rid, str(f), folder)
        self.client.log_metric(rid, "artifact_files", len(files))
        self.client.set_terminated(rid)
        result = {"mlflow_run_id": rid, "source_sha256": digest, "lab": lab, "files": len(files), "complete": True}
        self.index[key] = result
        self.save_index()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / "mlflow_store")
    parser.add_argument("--run", action="append", type=Path)
    parser.add_argument("--lab", type=int, action="append")
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()
    if args.max_points < 2:
        parser.error("max-points must be at least 2")
    importer = Importer(args.store, args.max_points)
    paths = args.run or sorted({p.parent for p in (ROOT / "runs").rglob("summary.json")})
    labs = set(args.lab) if args.lab else set(range(10))
    if args.lab:
        paths = [p for p in paths if run_lab(p.resolve(), yaml.safe_load((p / "config.resolved.yaml").read_text("utf-8-sig")) or {}, {}) in labs]
    if args.limit:
        paths = paths[:args.limit]
    failures = []
    begin = time.monotonic()
    counts = {}
    for i, path in enumerate(paths, 1):
        try:
            result = importer.import_run(path)
            counts[result["lab"]] = counts.get(result["lab"], 0) + 1
            print(f"[{i}/{len(paths)}] Lab{result['lab']} {'unchanged' if result.get('skipped') else 'imported'} {path.name}", flush=True)
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
            print(f"ERROR {path}: {exc}", flush=True)
    if not args.no_aggregate:
        for lab in sorted(labs):
            importer.import_lab_artifacts(lab)
    audit = {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.monotonic() - begin,
             "requested_runs": len(paths), "successful_by_lab": counts, "failures": failures,
             "tracking_uri": importer.uri, "max_curve_points_per_stream": args.max_points,
             "policy": "Original records preserved; historical import, no retraining; original statuses retained."}
    (args.store / "last_sync.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
