import csv
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

app = Flask(__name__)

# --- Global state ---
_process: subprocess.Popen | None = None
_process_lock = threading.Lock()
_log_queue: queue.Queue = queue.Queue()
_metrics: list[dict] = []
_json_buf: list[str] | None = None
_status = {"state": "idle", "pid": None, "type": None, "started": None}
_status_lock = threading.Lock()


def _build_clip_fine_tune_args(params: dict) -> list[str]:
    args = [sys.executable, str(SRC / "clip_fine_tune.py")]
    args += ["--dataset", params.get("dataset", "FashionIQ")]
    args += ["--clip-model-name", params.get("clip_model_name", "RN50x4")]
    args += ["--encoder", params.get("encoder", "both")]
    args += ["--batch-size", str(params.get("batch_size", 24))]
    args += ["--num-epochs", str(params.get("num_epochs", 100))]
    args += ["--learning-rate", str(params.get("learning_rate", 2e-6))]
    args += ["--transform", params.get("transform", "targetpad")]
    args += ["--target-ratio", str(params.get("target_ratio", 1.25))]
    args += ["--validation-frequency", str(params.get("validation_frequency", 1))]
    args += ["--fiq-dress-type", params.get("fiq_dress_type", "all")]
    args += ["--relevance", params.get("relevance", "dot")]
    args += ["--prob-lr", str(params.get("prob_lr", 1e-4))]
    args += ["--free-energy-scale", str(params.get("free_energy_scale", 10.0))]
    if params.get("save_training"):
        args.append("--save-training")
    if params.get("save_best"):
        args.append("--save-best")
    return args


def _build_combiner_train_args(params: dict) -> list[str]:
    args = [sys.executable, str(SRC / "combiner_train.py")]
    args += ["--dataset", params.get("dataset", "FashionIQ")]
    args += ["--clip-model-name", params.get("clip_model_name", "RN50x4")]
    args += ["--clip-model-path", params.get("clip_model_path", "")]
    args += ["--projection-dim", str(params.get("projection_dim", 2560))]
    args += ["--hidden-dim", str(params.get("hidden_dim", 5120))]
    args += ["--batch-size", str(params.get("batch_size", 1024))]
    args += ["--clip-bs", str(params.get("clip_bs", 16))]
    args += ["--num-epochs", str(params.get("num_epochs", 300))]
    args += ["--combiner-lr", str(params.get("combiner_lr", 2e-5))]
    args += ["--transform", params.get("transform", "targetpad")]
    args += ["--target-ratio", str(params.get("target_ratio", 1.25))]
    args += ["--validation-frequency", str(params.get("validation_frequency", 3))]
    args += ["--fiq-dress-type", params.get("fiq_dress_type", "all")]
    args += ["--relevance", params.get("relevance", "dot")]
    args += ["--prob-lr", str(params.get("prob_lr", 1e-4))]
    args += ["--free-energy-scale", str(params.get("free_energy_scale", 10.0))]
    if params.get("save_training"):
        args.append("--save-training")
    if params.get("save_best"):
        args.append("--save-best")
    return args


def _build_validate_args(params: dict) -> list[str]:
    args = [sys.executable, str(SRC / "validate.py")]
    args += ["--dataset", params.get("dataset", "FashionIQ")]
    args += ["--combining-function", params.get("combining_function", "sum")]
    args += ["--clip-model-name", params.get("clip_model_name", "RN50x4")]
    args += ["--transform", params.get("transform", "targetpad")]
    args += ["--target-ratio", str(params.get("target_ratio", 1.25))]
    if params.get("combining_function") == "combiner" and params.get("combiner_path"):
        args += ["--combiner-path", params["combiner_path"]]
    if params.get("clip_model_path"):
        args += ["--clip-model-path", params["clip_model_path"]]
    if params.get("projection_dim"):
        args += ["--projection-dim", str(params["projection_dim"])]
    if params.get("hidden_dim"):
        args += ["--hidden-dim", str(params["hidden_dim"])]
    if params.get("prob_head_path"):
        args += ["--prob-head-path", params["prob_head_path"]]
    return args


def _runner(args: list[str], task_type: str):
    global _process, _metrics, _json_buf
    _metrics = []
    _json_buf = None
    _log_queue.put({"type": "info", "msg": f"Starting: {' '.join(args)}", "ts": time.time()})

    try:
        _process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        with _status_lock:
            _status.update(state="running", pid=_process.pid, type=task_type,
                           started=time.strftime("%Y-%m-%d %H:%M:%S"))

        for line in _process.stdout:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            _log_queue.put({"type": "stdout", "msg": line, "ts": time.time()})
            _maybe_extract_metrics(line.strip())
            progress = _parse_tqdm(line.strip())
            if progress:
                _log_queue.put({"type": "progress", "data": progress, "ts": time.time()})

        _process.wait()
        rc = _process.returncode
        _log_queue.put({"type": "info", "msg": f"Process exited with code {rc}", "ts": time.time()})
        with _status_lock:
            _status.update(state="finished" if rc == 0 else "error", pid=None, type=None)
    except Exception as exc:
        _log_queue.put({"type": "error", "msg": f"Runner error: {exc}", "ts": time.time()})
        with _status_lock:
            _status.update(state="error", pid=None, type=None)
    finally:
        with _process_lock:
            _process = None


def _parse_tqdm(line: str) -> dict | None:
    """Parse tqdm training progress line like: [0/5] train loss: 1.108 :   1%|...| 6/702 [03:28<3:57:31, 20.45s/it]"""
    m = re.match(r'\[(\d+)/(\d+)\]\s*train loss:\s*([\d.]+)\s*:.*?(\d+)/(\d+)\s*\[', line)
    if m:
        return {
            "epoch": int(m.group(1)),
            "total_epochs": int(m.group(2)),
            "loss": float(m.group(3)),
            "batch": int(m.group(4)),
            "total_batches": int(m.group(5)),
        }
    return None


def _maybe_extract_metrics(line: str):
    global _json_buf
    stripped = line.strip()
    # Try single-line JSON first (preferred format)
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and "average_recall" in data:
                data["_ts"] = time.time()
                _metrics.append(data)
                _log_queue.put({"type": "metrics", "data": data, "ts": time.time()})
                return
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: accumulate multi-line JSON
    if stripped == '{':
        _json_buf = [line]
    elif _json_buf is not None:
        _json_buf.append(line)
        if stripped == '}':
            try:
                data = json.loads('\n'.join(_json_buf))
                if isinstance(data, dict) and "average_recall" in data:
                    data["_ts"] = time.time()
                    _metrics.append(data)
                    _log_queue.put({"type": "metrics", "data": data, "ts": time.time()})
            except (json.JSONDecodeError, TypeError):
                pass
            finally:
                _json_buf = None


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    global _process
    with _process_lock:
        if _process is not None and _process.poll() is None:
            return jsonify({"ok": False, "error": "Training already running"}), 409

    params = request.get_json(force=True) or {}
    task_type = params.get("task_type", "clip_fine_tune")

    if task_type == "clip_fine_tune":
        args = _build_clip_fine_tune_args(params)
    elif task_type == "combiner_train":
        args = _build_combiner_train_args(params)
    elif task_type == "validate":
        args = _build_validate_args(params)
    else:
        return jsonify({"ok": False, "error": f"Unknown task_type: {task_type}"}), 400

    # Clear old log entries before starting
    while not _log_queue.empty():
        _log_queue.get()

    threading.Thread(target=_runner, args=(args, task_type), daemon=True).start()
    return jsonify({"ok": True, "args": " ".join(args)})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with _process_lock:
        p = _process
    if p is None or p.poll() is not None:
        return jsonify({"ok": False, "error": "No training running"}), 409

    _log_queue.put({"type": "warn", "msg": "Stopping training...", "ts": time.time()})
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    _log_queue.put({"type": "warn", "msg": "Training stopped", "ts": time.time()})
    with _status_lock:
        _status.update(state="stopped", pid=None, type=None)
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(_status)


@app.route("/api/clip-models")
def api_clip_models():
    """Return a list of available tuned_clip_best.pt paths."""
    models = []
    for d in sorted(ROOT.glob("models/clip_finetuned_*"), reverse=True):
        pt = d / "saved_models" / "tuned_clip_best.pt"
        if pt.exists():
            models.append(str(pt.relative_to(ROOT)))
    return jsonify(models)


@app.route("/api/log")
def api_log():
    def generate():
        yield f"data: {json.dumps({'type': 'connected', 'msg': 'SSE connected'})}\n\n"
        while True:
            try:
                entry = _log_queue.get(timeout=2)
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/history")
def api_history():
    """Scan models/ directory and return all historical training records."""
    models_dir = ROOT / "models"
    if not models_dir.exists():
        return jsonify({"runs": []})

    runs = []
    for entry in sorted(models_dir.iterdir(), reverse=True):
        if not entry.is_dir():
            continue

        run = _read_run_dir(entry)
        if run:
            runs.append(run)

    return jsonify({"runs": runs})


def _read_run_dir(run_dir: Path) -> dict | None:
    """Read a single training run directory and return structured data."""
    hp_file = run_dir / "training_hyperparameters.json"
    train_csv = run_dir / "train_metrics.csv"
    val_csv = run_dir / "validation_metrics.csv"

    if not hp_file.exists():
        return None

    try:
        with open(hp_file, "r", encoding="utf-8") as f:
            hyperparameters = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    train_metrics = _read_csv(train_csv)
    val_metrics = _read_csv(val_csv)

    run_type = ""
    name = run_dir.name
    if "clip_finetuned" in name:
        run_type = "clip_fine_tune"
    elif "combiner_trained" in name:
        run_type = "combiner_train"

    best_epoch = None
    best_avg_recall = None
    if val_metrics:
        best = max(val_metrics, key=lambda r: r.get("average_recall", 0))
        best_epoch = best.get("epoch")
        best_avg_recall = best.get("average_recall")

    return {
        "name": name,
        "type": run_type,
        "hyperparameters": hyperparameters,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "best_epoch": best_epoch,
        "best_avg_recall": best_avg_recall,
        "num_epochs_completed": len(train_metrics),
    }


def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return list of dicts with numeric values parsed."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k.strip()] = float(v)
                    except (ValueError, TypeError):
                        parsed[k.strip()] = v
                rows.append(parsed)
            return rows
    except (OSError, csv.Error):
        return []


def _cleanup_subprocess():
    """Kill any running training subprocess on server shutdown."""
    with _process_lock:
        p = _process
    if p is not None and p.poll() is None:
        print("Terminating training subprocess...", flush=True)
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force killing training subprocess...", flush=True)
            p.kill()
            p.wait()


def _signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...", flush=True)
    _cleanup_subprocess()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    print(f"Starting CLIP4Cir training server at http://127.0.0.1:5000", flush=True)
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        _cleanup_subprocess()
