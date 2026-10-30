"""One bounded Modal invocation; all dependency installation is inside its budget."""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
SCRIPTS = ROOT / "artifact/scripts"
RAW = ROOT / "results/v8/preparation/raw/rel-f1"
VENDOR = ROOT / "artifact/relbench_vendor"
app = modal.App("equischema-v8-bounded")
image = modal.Image.debian_slim(python_version="3.11")
if modal.is_local():
    image = (
        image.add_local_dir(
            VENDOR, "/root/vendor", ignore=[".git/**", "**/__pycache__/**"]
        )
        .add_local_dir(SCRIPTS, "/root/scripts", ignore=["**/__pycache__/**"])
        .add_local_dir(RAW, "/root/data")
    )


@app.function(
    image=image,
    gpu="T4",
    cpu=(1, 2),
    memory=(4096, 8192),
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    retries=0,
    scaledown_window=2,
    startup_timeout=60,
    timeout=2400,
    single_use_containers=True,
)
def matrix() -> bytes:
    """Install Linux binaries, fit and export evidence within one resource cap."""
    began = time.monotonic()
    deadline = began + 2300
    pip = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--only-binary=:all:",
    ]
    subprocess.run(
        [*pip, "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"],
        check=True,
        timeout=500,
    )
    subprocess.run(
        [
            *pip,
            "numpy<2",
            "pandas==2.2.3",
            "duckdb",
            "pyarrow==19.0.1",
            "scikit-learn==1.6.1",
            "pyyaml",
            "huggingface-hub==0.28.1",
            "torch-geometric==2.6.1",
            "pytorch-frame==0.3.0",
            "sentence-transformers==3.4.1",
            "transformers==4.48.3",
            "https://data.pyg.org/whl/torch-2.6.0%2Bcu124/pyg_lib-0.4.0%2Bpt26cu124-cp311-cp311-linux_x86_64.whl",
        ],
        check=True,
        timeout=500,
    )
    out = Path("/tmp/evidence")
    out.mkdir(parents=True, exist_ok=True)
    # Fresh interpreter prevents dependency/version caches surviving pip installs.
    code = (
        "import sys,time,traceback; from pathlib import Path; "
        "sys.path[:0]=['/root/scripts','/root/vendor']; "
        "from relbench_engine_v8 import execute\n"
        "try:\n"
        f"    execute(Path('/root/data'),Path('/root/vendor'),Path('/tmp/evidence'),{deadline!r})\n"
        "except Exception:\n"
        "    Path('/tmp/evidence/failure.txt').write_text(traceback.format_exc())\n"
        "    raise\n"
    )
    try:
        subprocess.run(
            [sys.executable, "-u", "-c", code],
            check=True,
            timeout=max(1, int(deadline - time.monotonic())),
        )
    except subprocess.CalledProcessError as exc:
        (out / "exit_code.json").write_text(
            json.dumps({"returncode": exc.returncode, "status": "FAILED"})
        )
    (out / "requirements-linux.txt").write_bytes(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    )
    (out / "invocation_seconds.json").write_text(
        json.dumps({"seconds_including_installation": time.monotonic() - began})
    )
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(out, arcname="evidence")
    return stream.getvalue()


def main(out_dir: Path | None = None, budget_usd: float = 1.0) -> None:
    """Reserve worst-case cost, execute once, and never retry uncertain attempts."""
    from budget_v8 import reserve

    if not math.isfinite(budget_usd) or not 0.85 <= budget_usd <= 1.0:
        raise ValueError(
            "Bounded invocation requires an authorized budget between USD0.85 and USD1"
        )
    if (ROOT / "results/v8/relbench/analysis_verified.json").exists():
        raise FileExistsError(
            "Registered audit already complete; refusing another paid invocation"
        )
    actual = subprocess.check_output(
        ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != "91c1f6420a03feb61c279aaeb194d7268435f278":
        raise ValueError("Vendor revision differs from registered protocol")
    out = out_dir if out_dir is not None else ROOT / "results/v8/relbench"
    out.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / "results/v8/relbench/bounded_budget.json"
    # Covers a 3000-second upper envelope at capped CPU/RAM; setup stays reserved too.
    reserve(
        ledger,
        0.60,
        "single T4 invocation, startup/execution/shutdown envelope",
        cap_usd=budget_usd,
    )
    existing = json.loads(ledger.read_text())
    if not any(
        row["purpose"] == "platform setup, transfer and uncertainty reserve"
        for row in existing["reservations"]
    ):
        reserve(
            ledger,
            0.15,
            "platform setup, transfer and uncertainty reserve",
            cap_usd=budget_usd,
        )
    started = time.monotonic()
    data: bytes | None = None
    try:
        with modal.enable_output(), app.run():
            call = matrix.spawn()
            try:
                data = call.get(timeout=2550)
            except BaseException:
                call.cancel(terminate_containers=True)
                raise
        if data is None:
            raise RuntimeError(
                "Remote invocation ended without evidence; no automatic retry"
            )
        (out / "evidence.tar.gz").write_bytes(data)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            archive.extractall(out, filter="data")
        if (out / "evidence/failure.txt").exists():
            raise RuntimeError((out / "evidence/failure.txt").read_text())
    finally:
        (out / "client_timing.json").write_text(
            json.dumps(
                {
                    "seconds": time.monotonic() - started,
                    "reserved_usd": json.loads(ledger.read_text())["reserved_usd"],
                    "note": "reservation is an upper bound, not a measured billing receipt; no automatic refunds or retries",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
