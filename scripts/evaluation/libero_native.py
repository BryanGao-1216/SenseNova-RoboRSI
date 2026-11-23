#!/usr/bin/env python3
# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prepare and serve standard LIBERO inside an existing Linux container."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSET_REVISION = "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def native_pins() -> dict[str, str]:
    """Reuse the source revisions and hashes from the canonical Dockerfile."""
    recipe = (ROOT / "benchmarks/libero/Dockerfile").read_text()
    commit = re.search(r"ENV ROBORSI_LIBERO_COMMIT=([0-9a-f]{40})", recipe)
    source = re.search(
        r"ADD --checksum=sha256:([0-9a-f]{64})[\s\\]+"
        r"(https://codeload\.github\.com/[^\s]+)",
        recipe,
    )
    wheel = re.search(
        r"'(https://files\.pythonhosted\.org/[^']+rpent_rlinf[^']+)'", recipe
    )
    if not commit or not source or not wheel or "#sha256=" not in wheel[1]:
        raise ValueError("Cannot read pinned LIBERO/RLinf sources from Dockerfile")
    return {
        "commit": commit[1],
        "source_sha256": source[1],
        "source_url": source[2].replace("${ROBORSI_LIBERO_COMMIT}", commit[1]),
        "rlinf_wheel": wheel[1],
    }


def _run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def _download(url: str, target: Path, expected_sha256: str) -> None:
    """Publish a source archive only after verifying its pinned checksum."""
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("LIBERO source archive checksum mismatch")
            output.close()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def native_environment(work: Path) -> dict[str, str]:
    """Configure a CPU renderer and caches without changing the caller's env."""
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        PYTHONDONTWRITEBYTECODE="1",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LIBERO_TYPE="standard",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        LIBERO_CONFIG_PATH=str(work / "config"),
        MPLCONFIGDIR=str(work / "matplotlib"),
        NUMBA_CACHE_DIR=str(work / "numba"),
        ROBORSI_LIBERO_COMMIT=native_pins()["commit"],
    )
    return env


def configure(work: Path) -> None:
    """Bind verified assets inside the dedicated simulator Python environment."""
    from roborsi.benchmarks.registry import directory_sha256

    asset_root = work / "assets"
    asset = json.loads((asset_root / "asset.json").read_text())
    snapshot = (asset_root / asset["path"]).resolve()
    if (
        asset.get("version") != ASSET_REVISION
        or not snapshot.is_relative_to(asset_root.resolve())
        or not snapshot.is_dir()
        or directory_sha256(snapshot) != asset.get("sha256")
    ):
        raise ValueError("LIBERO asset version, path or checksum mismatch")
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("LIBERO is not installed in the simulator environment")
    package = Path(next(iter(spec.submodule_search_locations))) / "libero"
    if not (package / "bddl_files").is_dir() or not (package / "init_files").is_dir():
        raise ValueError(
            "Installed LIBERO is missing task definitions or initial states"
        )
    link = package / "assets"
    if link.is_symlink():
        if link.resolve() != snapshot:
            raise ValueError("Existing LIBERO assets link points to another snapshot")
    elif link.exists():
        raise ValueError("Refusing to replace existing LIBERO package assets")
    else:
        link.symlink_to(snapshot, target_is_directory=True)
    config_dir = work / "config"
    config_dir.mkdir(exist_ok=True)
    paths = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(package.parent / "datasets"),
        "assets": str(snapshot),
    }
    # JSON is valid YAML and avoids importing LIBERO before its paths exist.
    (config_dir / "config.yaml").write_text(json.dumps(paths, indent=2) + "\n")
    (work / "native-runtime.json").write_text(
        json.dumps({"pins": native_pins(), "paths": paths}, indent=2) + "\n"
    )
    print(f"LIBERO native environment ready: {work}", flush=True)


def prepare(work: Path, python: str) -> None:
    """Install the locked simulator stack in its own virtual environment."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ValueError("Native preparation requires Linux x86_64")
    version = subprocess.check_output(
        [python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        text=True,
    ).strip()
    if version != "3.11":
        raise ValueError(
            "The locked simulator requires Python 3.11; select --python PATH"
        )
    work.mkdir(parents=True, exist_ok=True)
    (work / "config").mkdir(exist_ok=True)
    env = native_environment(work)
    venv = work / "venv"
    native_python = venv / "bin/python"
    if not native_python.exists():
        _run([python, "-m", "venv", str(venv)])
    installed_version = subprocess.check_output(
        [str(native_python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        text=True,
    ).strip()
    if installed_version != "3.11":
        raise ValueError(
            "Existing simulator venv is not Python 3.11; use a new work directory"
        )
    _run(
        [
            str(native_python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(ROOT / "benchmarks/libero/requirements.txt"),
        ],
        env=env,
    )
    pins = native_pins()
    _run(
        [str(native_python), "-m", "pip", "install", "--no-deps", pins["rlinf_wheel"]],
        env=env,
    )
    with tempfile.TemporaryDirectory(prefix="libero-source-", dir=work) as temporary:
        stage = Path(temporary)
        archive = stage / "libero.tar.gz"
        source = stage / "source"
        source.mkdir()
        _download(pins["source_url"], archive, pins["source_sha256"])
        _run(["tar", "-xzf", str(archive), "-C", str(source), "--strip-components=1"])
        _run(
            [str(native_python), "-m", "pip", "install", "--no-deps", str(source)],
            env=env,
        )
    if not (work / "assets/asset.json").is_file():
        _run(
            [
                str(native_python),
                "-m",
                "benchmarks.libero.download_assets",
                "--output-dir",
                str(work / "assets"),
                "--revision",
                ASSET_REVISION,
            ],
            cwd=ROOT,
            env=env,
        )
    _run(
        [
            str(native_python),
            str(Path(__file__).resolve()),
            "configure",
            "--work-dir",
            str(work),
        ],
        env=env,
    )


def serve(work: Path, args) -> None:
    """Start an ordinary Python HTTP server, without a Docker daemon."""
    record = json.loads((work / "native-runtime.json").read_text())
    if record["pins"] != native_pins():
        raise ValueError("Native source pins changed; rerun prepare")
    native_python = str(work / "venv/bin/python")
    env = native_environment(work)
    _run(
        [
            native_python,
            str(Path(__file__).resolve()),
            "configure",
            "--work-dir",
            str(work),
        ],
        env=env,
    )
    record = json.loads((work / "native-runtime.json").read_text())
    env.update(
        ROBORSI_LIBERO_ASSETS_ROOT=record["paths"]["assets"],
        ROBORSI_LIBERO_SUITE=args.suite,
        ROBORSI_LIBERO_SEEDS=json.dumps(args.seed or [0]),
        ROBORSI_LIBERO_MAX_STEPS=str(args.max_steps),
    )
    command = [
        native_python,
        "-B",
        "-m",
        "roborsi.evaluation.service_host",
        "--kind",
        "benchmark",
        "--factory",
        "robots.libero.benchmark_service:make_libero_benchmark",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    os.execvpe(native_python, command, env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "serve", "configure"))
    parser.add_argument(
        "--work-dir", type=Path, default=Path.home() / "roborsi-work/libero-native"
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python 3.11 used to create the simulator venv",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18800)
    parser.add_argument("--suite", choices=SUITES, default="libero_spatial")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.port <= 65535
        or args.max_steps <= 0
        or any(seed < 0 for seed in args.seed or [])
    ):
        parser.error("port, max-steps or seed is out of range")
    # The dedicated simulator environment imports this checkout directly.
    sys.path.insert(0, str(ROOT))
    work = args.work_dir.expanduser().resolve()
    try:
        if args.action == "prepare":
            prepare(work, args.python)
        elif args.action == "configure":
            configure(work)
        else:
            serve(work, args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"LIBERO native {args.action} failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
