"""Start the local MLflow presentation server without changing training environments."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    store = ROOT / "mlflow_store"
    if not (store / "mlflow.db").exists():
        raise SystemExit("Run scripts/mlflow_sync.py first to import the original experiment records.")
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    url = f"http://127.0.0.1:{args.port}"
    try:
        with urlopen(url + "/health", timeout=2) as r:
            healthy = r.status == 200 and r.read().strip() == b"OK"
    except Exception:
        healthy = False
    if healthy:
        print(f"MLflow already available: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url + "/#/experiments")
        return
    command = [sys.executable, "-m", "mlflow", "server", "--backend-store-uri",
               "sqlite:///" + (store / "mlflow.db").as_posix(), "--host", "127.0.0.1",
               "--port", str(args.port), "--workers", "1", "--no-serve-artifacts"]
    server = subprocess.Popen(command, cwd=ROOT)
    try:
        for _ in range(80):
            if server.poll() is not None:
                raise SystemExit(server.returncode)
            try:
                with urlopen(url + "/health", timeout=1) as response:
                    if response.status == 200:
                        print(f"MLflow ready: {url}", flush=True)
                        if not args.no_browser:
                            webbrowser.open(url + "/#/experiments")
                        break
            except Exception:
                time.sleep(.25)
        else:
            server.terminate()
            server.wait()
            raise RuntimeError("MLflow server did not become ready; see the error above.")
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
