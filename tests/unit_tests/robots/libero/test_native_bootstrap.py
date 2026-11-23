# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Offline checks for container-local LIBERO installation and service startup."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.evaluation import libero_native as native


class NativeBootstrapTests(unittest.TestCase):
    def test_pins_reuse_the_docker_sources(self):
        pins = native.native_pins()
        self.assertNotIn("${", pins["source_url"])
        self.assertTrue(pins["source_url"].endswith(pins["commit"]))
        self.assertEqual(len(pins["source_sha256"]), 64)
        self.assertIn("rpent_rlinf-0.3.0", pins["rlinf_wheel"])
        self.assertIn("#sha256=", pins["rlinf_wheel"])

    def test_download_rejects_bad_hash_without_replacing_existing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "archive.tar.gz"
            target.write_bytes(b"existing")
            with patch.object(
                native.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")
            ):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    native._download("https://example.invalid/source", target, "0" * 64)
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_verified_download_is_published(self):
        payload = b"pinned archive"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "archive.tar.gz"
            with patch.object(
                native.urllib.request, "urlopen", return_value=io.BytesIO(payload)
            ):
                native._download(
                    "https://example.invalid/source",
                    target,
                    hashlib.sha256(payload).hexdigest(),
                )
            self.assertEqual(target.read_bytes(), payload)

    def test_prepare_rejects_wrong_python_before_creating_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "new-runtime"
            with (
                patch.object(native.platform, "system", return_value="Linux"),
                patch.object(native.platform, "machine", return_value="x86_64"),
                patch.object(native.subprocess, "check_output", return_value="3.12\n"),
                patch.object(native, "_run") as run,
            ):
                with self.assertRaisesRegex(ValueError, "requires Python 3.11"):
                    native.prepare(work, "python")
            self.assertFalse(work.exists())
            run.assert_not_called()

    def _fixture(self, root: Path):
        root = root.resolve()
        work = root / "runtime"
        snapshot = work / "assets/snapshots/selected"
        snapshot.mkdir(parents=True)
        (snapshot / "scene.xml").write_text("synthetic scene")
        asset = {
            "version": native.ASSET_REVISION,
            "path": "snapshots/selected",
            "sha256": "a" * 64,
        }
        (work / "assets/asset.json").write_text(json.dumps(asset))
        package = root / "site/libero/libero"
        (package / "bddl_files").mkdir(parents=True)
        (package / "init_files").mkdir()
        # Directory hashing has independent tests; these cases test path binding.
        registry = types.ModuleType("roborsi.benchmarks.registry")
        registry.directory_sha256 = Mock(return_value="a" * 64)
        spec = types.SimpleNamespace(submodule_search_locations=[str(package.parent)])
        return work, snapshot, package, registry, spec

    def test_configure_creates_local_paths_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            work, snapshot, package, registry, spec = self._fixture(Path(directory))
            with (
                patch.dict("sys.modules", {"roborsi.benchmarks.registry": registry}),
                patch.object(native.importlib.util, "find_spec", return_value=spec),
            ):
                native.configure(work)
                native.configure(work)
            paths = json.loads((work / "config/config.yaml").read_text())
            self.assertEqual(paths["assets"], str(snapshot))
            self.assertEqual(paths["init_states"], str(package / "init_files"))
            self.assertEqual((package / "assets").resolve(), snapshot)
            self.assertEqual(
                json.loads((work / "native-runtime.json").read_text())["pins"],
                native.native_pins(),
            )

    def test_configure_rejects_asset_paths_outside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, _, package, registry, spec = self._fixture(root)
            outside = root / "outside"
            outside.mkdir()
            receipt = work / "assets/asset.json"
            asset = json.loads(receipt.read_text())
            asset["path"] = str(outside)
            receipt.write_text(json.dumps(asset))
            with (
                patch.dict("sys.modules", {"roborsi.benchmarks.registry": registry}),
                patch.object(native.importlib.util, "find_spec", return_value=spec),
            ):
                with self.assertRaisesRegex(ValueError, "path or checksum mismatch"):
                    native.configure(work)
            self.assertFalse((package / "assets").exists())

    def test_configure_preserves_existing_package_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            work, _, package, registry, spec = self._fixture(Path(directory))
            (package / "assets").mkdir()
            with (
                patch.dict("sys.modules", {"roborsi.benchmarks.registry": registry}),
                patch.object(native.importlib.util, "find_spec", return_value=spec),
            ):
                with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                    native.configure(work)
            self.assertFalse((package / "assets").is_symlink())

    def test_serve_executes_python_http_service_with_selected_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            record = {
                "pins": native.native_pins(),
                "paths": {"assets": "/selected/assets"},
            }
            (work / "native-runtime.json").write_text(json.dumps(record))
            args = types.SimpleNamespace(
                suite="libero_goal",
                seed=[2, 4],
                max_steps=3,
                host="127.0.0.1",
                port=18800,
            )
            with (
                patch.object(native, "_run"),
                patch.object(native.os, "execvpe") as execute,
            ):
                native.serve(work, args)
            python, command, env = execute.call_args.args
            self.assertEqual(python, str(work / "venv/bin/python"))
            self.assertNotIn("docker", command)
            self.assertIn("roborsi.evaluation.service_host", command)
            self.assertEqual(env["MUJOCO_GL"], "osmesa")
            self.assertEqual(env["ROBORSI_LIBERO_ASSETS_ROOT"], "/selected/assets")
            self.assertEqual(env["ROBORSI_LIBERO_SEEDS"], "[2, 4]")
            self.assertEqual(env["ROBORSI_LIBERO_MAX_STEPS"], "3")


if __name__ == "__main__":
    unittest.main()
